# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An HTTP server that makes local [Laya](https://github.com/NandhaKishorM/laya) checkpoints a
drop-in replacement for [TypeSafe's Jev API](https://docs.typesafe.ai/api). A client sets
`TYPESAFE_BASE_URL` to this server and needs no other change. See `README.md` for the API surface
and the known divergences from Jev.

## Commands

`make help` lists everything. The common ones:

```bash
make install
make dev                        # first start downloads ~843MB per checkpoint
make test                       # fast suite: stubs Laya, no network, <1s
make test-slow                  # loads a real checkpoint (~20s)
make lint                       # ruff check + format check
make format                     # apply autofixes and reformat
make ci                         # what CI runs: lockfile check + lint + fast suite
uv run pytest tests/test_engine.py::TestConcurrency   # a single test target
```

`make ci` mirrors `.github/workflows/tests.yml`; keep them in step when either changes.

`make docker-build` / `make docker-run` build and run the container. The Dockerfile installs
CPU-only torch by exporting the locked versions, dropping the GPU-only packages
(`nvidia-*`, `triton`, `cuda-*`) and installing with `uv pip install --torch-backend=cpu` —
`uv sync` has no such flag and silently ignores `UV_TORCH_BACKEND`. That filter is what keeps
the image at ~1.6GB instead of ~2.7GB; triton alone is 813MB of GPU kernel compiler.

`make docker-push` publishes multi-arch to Docker Hub under `NAMESPACE` (default `pambrose`),
tagged with the `pyproject.toml` version and `latest`. It confirms before pushing. Version
bumps come from `pyproject.toml`; the Makefile reads it, so there is no second place to edit.

## Architecture

Request flow through `src/laya_server/`:

```
app.py      auth -> schema validation -> engine -> Jev-shaped JSON + X-Laya-Checkpoint
engine.py   validate questions -> resolve model -> route -> validate budget -> lock -> threadpool
translate.py  Laya answer dict -> Jev answer dict
```

The two non-obvious constraints that shape all of it:

**Laya validates nothing.** It has no schema checks and no exception types, so malformed input
becomes a raw `KeyError`/`AttributeError`/numpy `ValueError` — i.e. a client-triggerable 500.
`validation.py` pre-empts each of those and raises `ValidationFailed` (422) instead. Every check
there maps to a specific crash site in the installed library, cited in the comments. If you add a
question field, add its validation there first; `tests/test_validation.py` has one case per crash
mode.

**Laya is not thread-safe.** No locks anywhere: `Router.load` is check-then-act, `_touch` can
raise from a concurrent evict, and the OOM fallback reassigns `self.device` mid-request. `engine.py`
is therefore the sole owner of the Router and the only place allowed to touch it. It preloads
every checkpoint, sizes `max_loaded` to cover all of them so eviction never runs, holds one
`asyncio.Lock` per checkpoint, and offloads the blocking forward pass. Routing happens *outside*
the lock because `Router.route` is pure — that is what lets different checkpoints run concurrently.
Don't add a Router call anywhere else.

## Conventions

- The response body must contain exactly `model`, `answers`, `usage` — nothing Jev would not send.
  Laya's extra `action` blocks, its noul `confidence`, and the `routing` key are stripped in
  `translate.py`. Server-specific data goes in a header instead (`X-Laya-Checkpoint`).
- Wire-shape changes need a matching case in `tests/test_sdk_compat.py`, which parses our output
  with TypeSafe's real `SystemOneResponse` model. Note the SDK ignores unknown keys, so that suite
  proves parseability, not exactness — `test_translate.py` covers exactness.
- The fast suite must never download a checkpoint. Use the fakes in `tests/conftest.py`.
- Warnings are errors (`filterwarnings = ["error", ...]` in `pyproject.toml`), so a new
  deprecation fails the suite rather than scrolling past. Silence one only with a message-specific
  `ignore:` entry plus a comment saying what to upgrade to and when the entry can go — never a
  blanket `ignore::DeprecationWarning`.
- `TestClient` needs `httpx2`; with plain `httpx` starlette emits a deprecation that, under the
  rule above, fails the build. `httpx2` is therefore an explicit dev dependency rather than
  something inherited from `typesafe-sdk`.
- Ruff's rule selection is pinned explicitly in `pyproject.toml` rather than inherited, because
  ruff's defaults shift between releases and CI must lint what a developer's machine does.
  Linting covers `src/`, `tests/` and `scripts/`.
- Laya reports device downgrades and OOM fallbacks via bare `print()` to stdout, never `logging`,
  so a silent `cuda -> cpu` downgrade will not appear in structured logs.

## Reading the library

`laya` has no online API docs. Read the installed source at
`.venv/lib/python3.12/site-packages/laya/` (`router.py`, `agent.py`, `common.py`, `presets.py`)
rather than guessing. Note the yes/no primitive is spelled `noul` — it looks like a typo and is not.

`docs/running-laya-outside-python.md` already documents the model itself: a ModernBERT encoder
plus a custom decision head, the `[MASK]`-marker sequence layout that lets one forward pass answer
a whole schema, the per-checkpoint length budgets, and a verified ONNX export (`scripts/`). Read
it before re-deriving any of that from the library source.

The one thing in there worth knowing up front: scoring is scaled by `temperature_by_options`, a
per-checkpoint lookup keyed `"<type>:<bucket>"`, and the flat `temperature` list is only the
fallback. `typed-decisions` has near-neutral base temperatures but inherits `english`'s full
table, so reading only the list gives plausible but wrong probabilities.

## Other agent configs

An OpenAI Codex config (`~/.codex/config.toml`) and a Gemini CLI config (`~/.gemini/settings.json`)
exist on this machine. Reply `/import` to scan and list what is importable, then
`/import --yes=<digest>` to apply the user-level items.
