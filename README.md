# laya-server

[![tests](https://github.com/pambrose/laya-server/actions/workflows/tests.yml/badge.svg)](https://github.com/pambrose/laya-server/actions/workflows/tests.yml)
[![python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue)](pyproject.toml)
[![license](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

An proof of concept implementation of [TypeSafe's Jev API](https://docs.typesafe.ai/api), backed by local
[Laya](https://github.com/NandhaKishorM/laya) checkpoints.

## Contents

<!-- toc -->

- [Running the Server](#running-the-server)
- [Using the Server](#using-the-server)
- [Endpoints](#endpoints)
- [Models](#models)
- [Health and readiness](#health-and-readiness)
- [Configuration](#configuration)
- [Known divergences from Jev](#known-divergences-from-jev)
- [Verified against the real SDK](#verified-against-the-real-sdk)
- [Docker](#docker)
  - [With the repo (Makefile targets)](#with-the-repo-makefile-targets)
  - [Publishing](#publishing)
  - [Image notes](#image-notes)
- [Development](#development)
- [CI](#ci)
- [Further reading](#further-reading)
- [License](#license)

<!-- /toc -->

## Running the Server


Run the server using Docker with:

```bash
docker run --rm -p 8000:8000 \
  -v laya-checkpoints:/home/app/.cache/huggingface \
  -e LAYA_SERVER_PRELOAD=english \
  pambrose/laya-server:latest
```

Run the server from the repo with:

```bash
git clone git@github.com:pambrose/laya-server.git
cd laya-server
make install
make dev          # or: make run, for no auto-reload
```

The first start of the server downloads checkpoints from HuggingFace (~843 MB each) and preloads them, which
takes a while. After the initial start, they are cached.

The image is CPU-only (~1.6GB) and runs as a non-root user. Checkpoints are **not** baked in:
they download on first start into the volume mounted at `/home/app/.cache/huggingface`, so the
~843MB-per-checkpoint fetch is paid once and survives restarts. Without that volume, every
`docker run --rm` re-downloads them.

`LAYA_SERVER_PRELOAD=english` loads one checkpoint instead of all three, which makes the first
start much faster. Any other checkpoint is still fetched on demand if routing picks it.

Set `HF_TOKEN` if the initial download stalls; unauthenticated HuggingFace fetches are
throttled.

## Using the Server


Test it from the CLI with:

```bash
curl -sX POST localhost:8000/v1/systemone -H 'Content-Type: application/json' -d '{
  "model": "jev-latest",
  "state": {"body": "We were billed twice for March. Refund it or we cancel."},
  "questions": {"refund": {"type": "noul", "instructions": "Is a refund requested?"}}
}' | jq
```

Point any Jev client at it:

```bash
TYPESAFE_BASE_URL=http://localhost:8000 TYPESAFE_API_KEY=dummy python your_jev_script.py
```

The SDK requires `TYPESAFE_API_KEY` to be set even when the server does not check it.

Point a [jev4k](https://github.com/pambrose/jev4k) client at it:
```bash
git clone git@github.com:pambrose/jev4k.git
cd jev4k
TYPESAFE_BASE_URL=http://localhost:8000 make example
TYPESAFE_BASE_URL=http://localhost:8000 make live-tests
```

## Endpoints


`POST /v1/systemone` — the request and response bodies are Jev's, unchanged.

`model` accepts `jev-latest`, `jev-preview`, and `jev-1.13.0`, all of which let Laya pick a
checkpoint by language detection. As an extension, Laya's own names — `english`,
`multilingual`, `typed-decisions` — force a specific one.

The response echoes the `model` you asked for. The checkpoint that actually served the request
is reported in the `X-Laya-Checkpoint` response header, so the JSON body contains nothing Jev
would not send.

## Models


`GET /v1/models` returns every name the `model` field accepts:

```bash
curl -s localhost:8000/v1/models | jq
{
  "models": [
    {
      "name": "jev-latest",
      "description": "Alias: routes by script and language detection to a Laya checkpoint.",
      "release_date": "2026-09-20"
    },
    ...
  ]
}
```

The three Jev aliases let Laya choose a checkpoint; the three Laya names force one. Descriptions
say what actually answers the request rather than mirroring Jev's wording, and `release_date` is
the PyPI release of the `laya` package that provides the checkpoints — not a Jev date.

Like `/v1/systemone`, this route requires the API key when one is configured.

## Health and readiness


| Route         | Meaning                                                            |
|---------------|--------------------------------------------------------------------|
| `GET /health` | Liveness. `200 {"status": "ok"}` whenever the process is serving.  |
| `GET /ready`  | Readiness. `200` once a checkpoint is resident, `503` otherwise.   |

```bash
curl -s localhost:8000/ready | jq
{
  "status": "ready",
  "device": "cpu",
  "devices": {"english": "cpu"},
  "checkpoints": {
    "loaded": ["english"],
    "available": ["english", "multilingual", "typed-decisions"]
  }
}
```

`devices` reports one entry per resident checkpoint and `device` summarises them, or is
`"mixed"` when they disagree. Both are read from the loaded checkpoints rather than from
configuration, so they show what Laya actually settled on: it downgrades an unavailable `cuda` or
`mps` to `cpu` with only a print to stdout, and it can move a single checkpoint to `cpu`
mid-request after an OOM. The per-checkpoint view is what makes that visible.

Neither route requires an API key, since orchestrators probe without credentials, and neither
takes an inference lock, so a server busy on a forward pass still answers. When
`LAYA_SERVER_API_KEY` is set, `/ready` returns only `{"status": "..."}` — the route stays open so
it can be probed, but the device and checkpoint inventory are withheld rather than published to
unauthenticated callers.

Because checkpoints are preloaded in the lifespan hook, which uvicorn awaits before opening the
listening socket, a container still downloading refuses connections rather than answering `503`.
The `503` path applies once the socket is open, for example if the engine failed to initialise.
Preloading blocks startup deliberately: serving before a checkpoint is resident would push Laya's
unsynchronised cold load onto the request path.

Neither route is part of Jev's API, so a drop-in client never sees them. The container
`HEALTHCHECK` uses `/ready`.

## Configuration


| Variable                | Default   | Meaning                                                                                                                             |
|-------------------------|-----------|-------------------------------------------------------------------------------------------------------------------------------------|
| `LAYA_SERVER_API_KEY`   | unset     | When set, requests need `Authorization: Bearer <key>`; a missing or wrong key gets Jev's 401. When unset, auth is skipped entirely. |
| `LAYA_SERVER_DEVICE`    | auto      | `cuda`, `mps`, or `cpu`. Laya auto-detects when unset.                                                                              |
| `LAYA_SERVER_PRELOAD`   | all three | Comma-separated checkpoints to load at startup.                                                                                     |
| `LAYA_SERVER_MAX_QUEUE` | 32        | In-flight requests **per checkpoint** before that checkpoint answers 529.                                                           |
| `LAYA_SERVER_MAX_BODY_BYTES` | 1000000 | Request bodies larger than this are rejected with a 422 before being parsed.                                                   |
| `LAYA_SERVER_LOG_LEVEL` | `INFO`    | Logger verbosity.                                                                                                                   |

## Known divergences from Jev


These are inherent to Laya, and are documented rather than hidden.

- **Context window.** Jev accepts 32k tokens of state; these checkpoints accept 512 (english) or
  1024 (multilingual), and the room left for state shrinks with the size of the question. Laya
  silently truncates oversized input, which would hand you an answer computed on partial data, so
  this server rejects it with a `422` naming the actual and available token counts instead.
- **`usage.input_tokens`.** Laya sums its attention mask across every question row, so the state
  is counted once per question. Jev counts state plus questions combined. The number is passed
  through unchanged but is not comparable to Jev's.
- **Option text.** Laya hard-truncates each choice option to 48 tokens, and truncates further
  when many options crowd the question header. This happens silently inside the library.
- **Answer quality.** Laya's checkpoints are not Jev. The typed contract guarantees the shape of
  the answer, not that it matches what Jev would have said. Spot-check accuracy on your own data
  before treating this as a substitute.

## Verified against the real SDK


The response contract is checked two ways: `tests/test_sdk_compat.py` parses our output with
TypeSafe's own `SystemOneResponse` model in the fast suite, and the server has been driven
end-to-end by an unmodified `typesafe-sdk` client. Both agree that `legend` is an integer-keyed
map — `api.md` documents it that way and the SDK types it `dict[int, str]`, though the example in
`primitives.md` shows an array.

Errors are parsed too: a `422` from this server surfaces as the SDK's
`TypeSafeUnprocessableEntityError` with the message intact, and a bad key as
`TypeSafeAuthenticationError`.

## Docker


### With the repo (Makefile targets)

| Target               | What it does                                                              |
|----------------------|---------------------------------------------------------------------------|
| `make docker-build`  | Builds a local, single-architecture image tagged `laya-server:dev`.       |
| `make docker-run`    | Runs that image on `PORT` with a named volume for checkpoints.            |
| `make docker-buildx` | Cross-builds every target platform to check they compile. Pushes nothing. |
| `make docker-login`  | `docker login` against `REGISTRY`. Needed once before publishing.         |
| `make docker-push`   | Builds multi-arch and **publishes**. Prints the tags and asks to confirm. |

A normal loop is build, run, then hit it:

```bash
make docker-build
make docker-run PORT=8000 PRELOAD=english   # foreground; Ctrl-C to stop
make smoke PORT=8000                        # in another shell
```

Variables you can override on any of these:

| Variable       | Default                   | Meaning                                          |
|----------------|---------------------------|--------------------------------------------------|
| `IMAGE`        | `laya-server`             | Image name.                                      |
| `TAG`          | `dev`                     | Tag used by `docker-build` / `docker-run`.       |
| `PORT`         | `8000`                    | Host port; the container always listens on 8000. |
| `PRELOAD`      | *(empty — all three)*     | Passed as `LAYA_SERVER_PRELOAD`.                 |
| `CACHE_VOLUME` | `laya-checkpoints`        | Named volume holding downloaded checkpoints.     |
| `REGISTRY`     | `docker.io`               | Registry to log in to and push to.               |
| `NAMESPACE`    | `pambrose`                | Docker Hub user or organization.                 |
| `PLATFORMS`    | `linux/amd64,linux/arm64` | Platforms for `docker-buildx` / `docker-push`.   |

### Publishing

```bash
make docker-login          # once per registry
make docker-buildx         # optional: confirm both platforms build
make docker-push           # builds multi-arch, then publishes
```

`docker-push` publishes two tags — the version read from `pyproject.toml` and `latest` — for
`linux/amd64` and `linux/arm64`. It prints the exact references and waits for confirmation,
because the result is public. To publish elsewhere:

```bash
make docker-push NAMESPACE=myorg
make docker-push REGISTRY=ghcr.io NAMESPACE=myuser
make docker-push PLATFORMS=linux/amd64
```

A multi-arch build cannot be loaded into the local docker daemon, so publishing builds and
pushes in one step. Use `make docker-build` when you want an image you can run locally.

### Image notes

Every `LAYA_SERVER_*` variable from [Configuration](#configuration) works as `-e`. The
`HEALTHCHECK` polls `/ready` and takes no inference lock. While the first checkpoint downloads
the socket is not yet open, so the probe fails with connection-refused rather than `503`;
`start-period` is 5 minutes to cover that.

## Development


```bash
make test        # fast suite, no model download
make test-slow   # loads a real checkpoint
make test-all    # everything
make lint        # ruff check + format check, no changes written
make typecheck   # mypy, strict, over src/
make format      # apply autofixes and reformat
make ci          # exactly what CI runs on a pull request
make smoke       # POST a sample request to a running server
```

The fast suite stubs the Laya Router, so it runs in under a second and never touches the network.

## CI


`.github/workflows/tests.yml` runs the fast suite on every push to `main`/`master` and on every
pull request. The checkpoint-backed suite downloads model weights, so it is not on that path: it
runs weekly on a schedule and on demand via *Run workflow* (`workflow_dispatch`), with the
HuggingFace cache preserved between runs.

CI installs with `uv sync --locked`, so a stale `uv.lock` fails the build. It runs ruff and mypy
in one job, then the test suite against **Python 3.12 and 3.13** — every version
`requires-python` claims to support. `make ci` reproduces all of it locally.

If you enable branch protection, the required status checks are **`lint and types`**,
**`tests (py3.12)`** and **`tests (py3.13)`**. GitHub matches required checks by display name, so
a rule naming a job that no longer exists (such as the old `fast suite`) would never report and
would leave pull requests pending forever.

[Dependabot](.github/dependabot.yml) watches Python dependencies, GitHub Actions and the
Dockerfile base images weekly.

Linting covers `src/`, `tests/` and `scripts/`.

## Further reading


- [Running Laya outside Python](docs/running-laya-outside-python.md) — what the checkpoints
  actually are (a ModernBERT encoder plus a custom decision head), which other languages can
  serve them, and a verified ONNX export that reproduces this server's payload exactly. Also
  documents the per-checkpoint temperature tables and the 48-token option truncation noted
  above.

## License


[Apache 2.0](LICENSE). Laya itself is Apache 2.0 as well.
