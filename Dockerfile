# syntax=docker/dockerfile:1

# The one Python pin. The builder's venv links to this interpreter and is copied
# into the runtime, so both stages must run the same Python; building both from
# this stage makes that structural rather than two tags kept in step by hand.
FROM python:3.12-slim AS base

# uv comes from its own image, which holds only the static binaries. A named
# stage rather than an inline COPY --from keeps the tag on a FROM line, where
# Dependabot looks for it.
FROM ghcr.io/astral-sh/uv:0.9.30 AS uv

# ---------------------------------------------------------------- build stage
FROM base AS builder

COPY --from=uv /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    VIRTUAL_ENV=/opt/venv

WORKDIR /build
COPY pyproject.toml uv.lock README.md ./

# CPU-only torch. The default wheel drags in GPU-only packages this image can
# never use: ~15 nvidia-* CUDA libraries, triton (813MB, a GPU kernel compiler)
# and the cuda-* bindings. Nothing in the installed set declares them as a
# dependency once torch comes from PyTorch's CPU index.
#
# `uv sync` has no --torch-backend flag and ignores UV_TORCH_BACKEND, so the
# locked versions are exported, the GPU-only packages dropped, and the rest
# installed with `uv pip install --torch-backend=cpu`. Versions still come from
# uv.lock, so the image stays reproducible.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv export --no-dev --no-hashes --no-emit-project -o /tmp/requirements.txt \
    && grep -vE '^(nvidia-|triton|cuda-)' /tmp/requirements.txt \
        > /tmp/requirements.cpu.txt \
    && uv venv "$VIRTUAL_ENV" \
    && uv pip install --torch-backend=cpu -r /tmp/requirements.cpu.txt

# The project itself, in its own layer: editing src/ does not reinstall deps.
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv uv pip install --no-deps .

# -------------------------------------------------------------- runtime stage
FROM base

RUN useradd --create-home --uid 10001 app

# HF_HOME is where checkpoints land; mount a volume there so the ~843MB
# download survives container restarts.
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/app/.cache/huggingface \
    LAYA_SERVER_DEVICE=cpu

COPY --from=builder --chown=app:app /opt/venv /opt/venv

USER app
WORKDIR /home/app
RUN mkdir -p "$HF_HOME"

EXPOSE 8000

# /ready takes no inference lock, so a container busy on a forward pass still
# answers. Note that checkpoints are preloaded in the lifespan hook, which
# uvicorn awaits before opening the socket: during the first download the probe
# gets connection-refused, not a 503. start-period covers that window.
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready').read()"

CMD ["uvicorn", "laya_server.app:app", "--host", "0.0.0.0", "--port", "8000"]
