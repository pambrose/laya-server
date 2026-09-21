# laya-server

A drop-in replacement for [TypeSafe's Jev API](https://docs.typesafe.ai/api), backed by local
[Laya](https://github.com/NandhaKishorM/laya) checkpoints.

Point an existing TypeSafe client at this server and it keeps working:

```bash
TYPESAFE_BASE_URL=http://localhost:8000 TYPESAFE_API_KEY=dummy python your_jev_script.py
```

## Running the Laya API Server

```bash
make install
make dev          # or: make run, for no auto-reload
```

`make help` lists every target. Override the defaults as `make run PORT=9000 PRELOAD=english`.

The first start downloads checkpoints from HuggingFace (~843 MB each) and preloads them, which
takes a while. Afterwards they are cached.

```bash
curl -sX POST localhost:8000/v1/systemone -H 'Content-Type: application/json' -d '{
  "model": "jev-latest",
  "state": {"body": "We were billed twice for March. Refund it or we cancel."},
  "questions": {"refund": {"type": "noul", "instructions": "Is a refund requested?"}}
}' | jq
```

## Docker

### With the repo (Makefile targets)

| Target | What it does |
| --- | --- |
| `make docker-build` | Builds a local, single-architecture image tagged `laya-server:dev`. |
| `make docker-run` | Runs that image on `PORT` with a named volume for checkpoints. |
| `make docker-buildx` | Cross-builds every target platform to check they compile. Pushes nothing. |
| `make docker-login` | `docker login` against `REGISTRY`. Needed once before publishing. |
| `make docker-push` | Builds multi-arch and **publishes**. Prints the tags and asks to confirm. |

A normal loop is build, run, then hit it:

```bash
make docker-build
make docker-run PORT=8000 PRELOAD=english   # foreground; Ctrl-C to stop
make smoke PORT=8000                        # in another shell
```

Variables you can override on any of these:

| Variable | Default | Meaning |
| --- | --- | --- |
| `IMAGE` | `laya-server` | Image name. |
| `TAG` | `dev` | Tag used by `docker-build` / `docker-run`. |
| `PORT` | `8000` | Host port; the container always listens on 8000. |
| `PRELOAD` | *(empty — all three)* | Passed as `LAYA_SERVER_PRELOAD`. |
| `CACHE_VOLUME` | `laya-checkpoints` | Named volume holding downloaded checkpoints. |
| `REGISTRY` | `docker.io` | Registry to log in to and push to. |
| `NAMESPACE` | `pambrose` | Docker Hub user or organization. |
| `PLATFORMS` | `linux/amd64,linux/arm64` | Platforms for `docker-buildx` / `docker-push`. |

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

### Without the repo

Once the image is published, nothing else is needed — no clone, no Python, no Makefile:

```bash
docker run --rm -p 8000:8000 \
  -v laya-checkpoints:/home/app/.cache/huggingface \
  -e LAYA_SERVER_PRELOAD=english \
  pambrose/laya-server:latest
```

Then, from anywhere:

```bash
curl -sX POST localhost:8000/v1/systemone -H 'Content-Type: application/json' -d '{
  "model": "jev-latest",
  "state": {"body": "We were billed twice for March. Refund it or we cancel."},
  "questions": {"refund": {"type": "noul", "instructions": "Is a refund requested?"}}
}'
```

Or point any TypeSafe client at it:

```bash
TYPESAFE_BASE_URL=http://localhost:8000 TYPESAFE_API_KEY=dummy python your_jev_script.py
```

Run it in the background with a restart policy, an API key and all three checkpoints:

```bash
docker run -d --name laya-server --restart unless-stopped -p 8000:8000 \
  -v laya-checkpoints:/home/app/.cache/huggingface \
  -e LAYA_SERVER_API_KEY=your-secret \
  pambrose/laya-server:latest

docker logs -f laya-server        # watch the first-run checkpoint download
docker stop laya-server && docker rm laya-server
```

As compose:

```yaml
services:
  laya-server:
    image: pambrose/laya-server:latest
    ports: ["8000:8000"]
    environment:
      LAYA_SERVER_PRELOAD: english
    volumes:
      - laya-checkpoints:/home/app/.cache/huggingface
volumes:
  laya-checkpoints:
```

### Notes

The image is CPU-only (~1.6GB) and runs as a non-root user. Checkpoints are **not** baked in:
they download on first start into the volume mounted at `/home/app/.cache/huggingface`, so the
~843MB-per-checkpoint fetch is paid once and survives restarts. Without that volume, every
`docker run --rm` re-downloads them.

`LAYA_SERVER_PRELOAD=english` loads one checkpoint instead of all three, which makes the first
start much faster. Any other checkpoint is still fetched on demand if routing picks it.

Every `LAYA_SERVER_*` variable from [Configuration](#configuration) works as `-e`. The
`HEALTHCHECK` polls `/openapi.json`, which is liveness only — it never takes an inference lock.
Its `start-period` is 5 minutes to cover the first checkpoint download.

Set `HF_TOKEN` if the initial download stalls; unauthenticated HuggingFace fetches are
throttled.

## Endpoint

`POST /v1/systemone` — the request and response bodies are Jev's, unchanged.

`model` accepts `jev-latest`, `jev-preview`, and `jev-1.13.0`, all of which let Laya pick a
checkpoint by language detection. As an extension, Laya's own names — `english`,
`multilingual`, `typed-decisions` — force a specific one.

The response echoes the `model` you asked for. The checkpoint that actually served the request
is reported in the `X-Laya-Checkpoint` response header, so the JSON body contains nothing Jev
would not send.

## Configuration

| Variable                | Default   | Meaning                                                                                                                             |
|-------------------------|-----------|-------------------------------------------------------------------------------------------------------------------------------------|
| `LAYA_SERVER_API_KEY`   | unset     | When set, requests need `Authorization: Bearer <key>`; a missing or wrong key gets Jev's 401. When unset, auth is skipped entirely. |
| `LAYA_SERVER_DEVICE`    | auto      | `cuda`, `mps`, or `cpu`. Laya auto-detects when unset.                                                                              |
| `LAYA_SERVER_PRELOAD`   | all three | Comma-separated checkpoints to load at startup.                                                                                     |
| `LAYA_SERVER_MAX_QUEUE` | 32        | In-flight requests before the server answers 529.                                                                                   |
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

## Further reading

- [Running Laya outside Python](docs/running-laya-outside-python.md) — what the checkpoints
  actually are (a ModernBERT encoder plus a custom decision head), which other languages can
  serve them, and a verified ONNX export that reproduces this server's payload exactly. Also
  documents the per-checkpoint temperature tables and the 48-token option truncation noted
  above.

## Development

```bash
make test        # fast suite, no model download
make test-slow   # loads a real checkpoint
make test-all    # everything
make lint        # ruff check + format check, no changes written
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

CI installs with `uv sync --locked`, so a stale `uv.lock` fails the build, and runs ruff before
the tests. `make ci` reproduces that locally.

Linting covers `src/`, `tests/` and `scripts/`.

## License

[Apache 2.0](LICENSE). Laya itself is Apache 2.0 as well.
