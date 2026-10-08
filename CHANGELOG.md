# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Upgraded laya from 0.3.4 to 0.4.1; the minimum is now `laya>=0.4.1`. Answers on english and
  typed-decisions choice questions with 11 or more options are noticeably less confident, because
  laya now clamps the checkpoints' `choice:11+` temperature from 0.1006 to 0.5. Questions with
  fewer options score exactly as before on `cpu` and `cuda`. On `mps`, laya now runs requests with
  five or more questions in fp16, so their probabilities can differ in the last digits.
- Questions laya 0.4 refuses are now rejected with `422` before inference, with laya's message:
  blank `instructions`, noul `criteria` keys other than `true`/`false` (matched
  case-insensitively), null score levels, and list-form choice labels that are null, non-scalar or
  duplicated.
- Question fields other than `type`, `instructions` and `criteria` are ignored. laya 0.4 would
  otherwise act on its own `labels` and `option_order` fields, which Jev does not have.
- `LAYA_SERVER_PRELOAD` set to only commas or spaces now preloads all three checkpoints, as an
  empty value already did.

### Added

- `LAYA_SERVER_DEFAULT_CHECKPOINT` (default `english`): where a `jev-*` request goes when language
  detection cannot identify the state's language. laya 0.4 changed its own fallback to
  `multilingual`; the server keeps `english`, so the upgrade does not reroute such requests.

### Fixed

- A list-form choice with an unhashable label (`[["a"], "b"]`) returned `500`; it is now a `422`.
- Kept laya 0.4's new response fields off the wire: `answer_confidence` on every answer, and
  `state_tokens`, `truncated` and related diagnostics in `usage`, which now carries only
  `input_tokens` and `output_tokens`.
- Score levels written as objects or arrays are echoed in `legend` as written, not as the JSON
  text laya 0.4 renders them to.
- The budget check now tokenizes under laya's tokenizer lock. laya 0.4 mutates the shared
  tokenizer while encoding options, so an unlocked call could fail concurrent requests.
- Language detection runs off the event loop. laya 0.4 scans every field of a structured state,
  where 0.3.4 stopped after 4000 characters.

## [0.1.0] - 2026-09-20

### Added

- `POST /v1/systemone`, wire-compatible with [TypeSafe's Jev API](https://docs.typesafe.ai/api),
  backed by local [Laya](https://github.com/NandhaKishorM/laya) checkpoints. Pointing
  `TYPESAFE_BASE_URL` at this server is the only change a TypeSafe client needs.
- A validation layer that rejects malformed questions with `422` instead of the raw
  `KeyError`/`AttributeError`/numpy errors Laya raises, and rejects oversized state rather than
  letting Laya silently truncate it.
- Single-owner concurrency around Laya's `Router`, which has no internal locking: every
  checkpoint is preloaded, `max_loaded` covers all of them so eviction never runs mid-request,
  and each checkpoint has its own lock.
- Optional Bearer auth, enforced only when `LAYA_SERVER_API_KEY` is set.
- A CPU-only Docker image (~1.6GB) running as a non-root user, with checkpoints downloaded at
  runtime into a mounted cache volume.
- Compatibility tests that parse responses with TypeSafe's own `SystemOneResponse` model.
- `GET /v1/models`, matching Jev's endpoint of the same name. Lists the three Jev aliases plus
  Laya's own checkpoint names, since all six are accepted by the `model` field.
- `GET /health` (liveness) and `GET /ready` (readiness, reporting resident checkpoints and the
  device actually in use). Both skip auth and take no inference lock; the container
  `HEALTHCHECK` uses `/ready`.

### Fixed

- A question with many options and an empty state returned 500 instead of 422: Laya's
  `head_max_len` limit is tighter than Jev's documented 255-option ceiling, and was not checked.
  Laya errors during inference are now reported as 422 rather than reaching the client as a 500.
- `LAYA_SERVER_LOG_LEVEL` was parsed and documented but applied to no logger.
- The API key was compared with `!=`; it now uses `secrets.compare_digest`.
- Request bodies are now capped by `LAYA_SERVER_MAX_BODY_BYTES`, and budget validation runs off
  the event loop, so a large request no longer stalls the health probes.
- `GET /ready` no longer reorders the router's LRU list as a side effect of being probed.
- Backpressure is tracked per checkpoint, so a queue on one no longer returns 529 for an idle one.
- `GET /ready` reports a device per checkpoint rather than letting one stand in for the server,
  and withholds the inventory when an API key is configured.
- A readiness probe failure returns 503 rather than a 500 traceback.
- `GET /v1/models` derives its catalog and its `release_date` from the installed laya, so neither
  can silently drift.
- A non-numeric `LAYA_SERVER_MAX_QUEUE` or `LAYA_SERVER_MAX_BODY_BYTES` now fails with a message
  naming the variable and the value instead of a bare traceback.

### Known divergences from Jev

See [the README](README.md#known-divergences-from-jev): a much smaller context window,
non-comparable `usage.input_tokens`, silent option-text truncation, and answers that come from
Laya rather than Jev.

[Unreleased]: https://github.com/pambrose/laya-server/compare/0.1.0...HEAD
[0.1.0]: https://github.com/pambrose/laya-server/releases/tag/0.1.0
