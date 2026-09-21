# Code review findings

Open issues found by reviewing `src/laya_server/`. Tick a box when the issue is fixed and update
the **Status** line in its section. Numbers are stable — do not renumber when items are closed.

_Last updated: 2026-09-20. 18 issues: 18 fixed, 0 open._
_Issues 12-18 came from an independent review pass; each was re-verified before being recorded._

## Summary

- [x] **1.** A request with many options and an empty state returns **HTTP 500** instead of 422.
- [x] **2.** Option-count overflow is reported as a misleading "state exceeds model budget" 422.
- [x] **3.** `LAYA_SERVER_LOG_LEVEL` is parsed and documented but never applied to any logger.
- [x] **4.** The API key is compared with `!=`, which is not constant-time.
- [x] **5.** No request body size limit, and validation tokenizes on the event loop (~2s for 10MB).
- [x] **6.** `GET /ready` mutates the router's LRU order, making `checkpoints.loaded` oscillate between probes.
- [x] **7.** The `RequestValidationError` handler is unreachable dead code.
- [x] **8.** State is serialized and tokenized twice per request, once in validation and once in laya.
- [x] **9.** Backpressure uses one global counter despite per-checkpoint locks, so a busy checkpoint can 529 an idle one.
- [x] **10.** `CHECKPOINTS_RELEASED` in `models.py` is hardcoded and silently drifts when laya is bumped.
- [x] **11.** A malformed `LAYA_SERVER_MAX_QUEUE` crashes startup with a bare `ValueError` traceback.
- [x] **12.** `/ready` reports one arbitrary agent's device as if it were the server's, and it flaps.
- [x] **13.** `/ready`'s 503 branch is unreachable in the container, so the `HEALTHCHECK` claim overstates what it verifies.
- [x] **14.** `engine.status()` is unguarded, so a probe failure returns 500 rather than 503.
- [x] **15.** The router test double does not mirror real LRU semantics, hiding issue 6 from the suite.
- [x] **16.** `/ready` discloses device and checkpoint inventory unauthenticated while `/v1/models` needs a key.
- [x] **17.** The CI job display names changed, which can strand branch-protection required checks.
- [x] **18.** Model names are duplicated as literals in `models.py` rather than derived from `engine.py`.

---

## 1. Many options plus an empty state returns 500

**Severity:** high · **Status:** fixed — option budget validated, plus an engine backstop · `validation.py`, `engine.py`

Laya raises `ValueError("question %r options exceed head_max_len=%d")` at `agent.py:263` when a
question's rendered options do not fit the 192-token head. `validate_questions` only enforces
Jev's documented limit of 255 options, so a question with ~200 options passes validation and the
`ValueError` propagates out of `Engine.predict` as an unhandled 500.

Long or short states are caught *incidentally* by `validate_budget` (see issue 2), but an empty
state has zero tokens, so the budget check passes and nothing stops it.

```
state="word "*200 -> HTTP 422   (budget check fires, wrong reason)
state="hi"        -> HTTP 422   (budget check fires, wrong reason)
state=""          -> HTTP 500   Internal Server Error
```

**Fix:** validate the option budget against the checkpoint's `head_max_len` in `validation.py`,
mirroring the `build_sequence`/`render_options` marker check, and raise `ValidationFailed`. As a
backstop, catch `ValueError` from laya in `Engine.predict` and convert it to a 422 rather than
letting any future laya error reach the client as a 500.

## 2. Option overflow reported as a state-size error

**Severity:** medium · **Status:** fixed — options are checked before state, and named in the message · `validation.py`

When the options blow the head budget, the room left for state collapses, so `validate_budget`
rejects with a message about the *state* even when the state is one token:

```
state exceeds model budget: 1 tokens, but question 'pick' leaves room for 0 on checkpoint 'english'
```

The caller is told to shrink the state when the actual problem is the question's option count.
Fixing issue 1 should also fix this, by checking the option budget first and reporting it as such.

## 3. `LAYA_SERVER_LOG_LEVEL` has no effect

**Severity:** medium · **Status:** fixed — applied in `create_app` · `config.py`, `app.py`

`Settings.log_level` is read from the environment and documented in the README's configuration
table, but nothing ever calls `logging.basicConfig`, `setLevel`, or configures uvicorn's log
config with it. Setting `LAYA_SERVER_LOG_LEVEL=DEBUG` changes nothing.

Note also that laya reports device downgrades and OOM fallbacks with bare `print()` to stdout
(`agent.py:159,163,219-227,280`), so those never reach the logger regardless.

**Fix:** apply the level in `create_app` (or the lifespan), and decide whether to capture laya's
stdout into the logger. Alternatively drop the setting and its README row.

## 4. API key comparison is not constant-time

**Severity:** medium · **Status:** fixed — `secrets.compare_digest` · `app.py:44`

```python
if authorization != expected:
    raise Unauthorized("missing or invalid API key")
```

`!=` on `str` short-circuits at the first differing byte, which leaks key content through
response timing. The practical risk is bounded — the key is a shared secret over a network, and
the noise floor is high — but the fix is one line.

**Fix:** `secrets.compare_digest(authorization or "", expected)`.

## 5. No request size limit, and validation blocks the event loop

**Severity:** medium · **Status:** fixed — `LAYA_SERVER_MAX_BODY_BYTES` + validation offloaded · `app.py`, `validation.py`

`await request.json()` reads an unbounded body, and `validate_budget` then tokenizes the
serialized state with the real HuggingFace tokenizer **on the event loop**, before any lock is
taken. Measured with the english checkpoint:

| Body | Questions | Event loop blocked |
| --- | --- | --- |
| 10KB | 10 | ~0.00s |
| 1MB | 10 | ~0.18s |
| 10MB | 10 | ~1.95s |

While blocked, nothing else is served — including `/health` and `/ready`, which were specifically
designed not to wait on inference. A handful of large unauthenticated requests is a cheap denial
of service.

**Fix:** reject bodies over a configurable limit by `Content-Length` before reading them, and
consider offloading `validate_budget` to the threadpool as `_run_inference` already is.

## 6. `GET /ready` mutates router state, so reported order oscillates

**Severity:** medium · **Status:** fixed — device read without `load()`, `loaded` sorted · `engine.py`

`Engine.status()` calls `self._router.load(loaded[0])` purely to read `.device`. `Router.load`
calls `_touch()` (`router.py:169-185`), which does `self._order.remove(key)` then
`self._order.append(key)`. Since `Router.loaded` is `list(self._order)`, every probe reorders
what the next probe reports. Verified against the real `laya.Router`:

```
initial _order: ['english', 'multilingual']
probe 1: reported=['english', 'multilingual']  -> order after=['multilingual', 'english']
probe 2: reported=['multilingual', 'english']  -> order after=['english', 'multilingual']
probe 3: reported=['english', 'multilingual']  -> order after=['multilingual', 'english']
```

The Docker `HEALTHCHECK` runs every 30s, so `checkpoints.loaded` flips forever as a side effect of
observing it. This also contradicts the comment in `app.py` and the `Dockerfile` claiming the
probes "touch no checkpoint" — they enter `Router.load`, the lazy build path.

**Fix:** capture the device at preload time and report from that, without calling `load()`. Sort
`loaded` before reporting so the output is stable regardless. See also issues 12 and 15.

## 7. Dead `RequestValidationError` handler

**Severity:** low · **Status:** fixed — handler deleted · `app.py:69`

The handler cannot fire. The request body is validated manually inside the route (deliberately,
so auth precedes body validation), and the only declared parameter is
`authorization: str | None = Header(default=None)`, which accepts anything. No path or query
parameters are declared. FastAPI therefore never raises `RequestValidationError` for these routes,
and no test exercises the handler.

**Fix:** delete it, or keep it with a comment explaining what future route would need it.

## 8. State is serialized and tokenized twice

**Severity:** low · **Status:** fixed — state serialized once, before inference · `validation.py`

`validate_budget` calls `serialize_state` and tokenizes the result to count tokens. Laya then
calls `serialize_state` again inside `build_sequence` — once **per question** — and re-tokenizes.
For a large state with several questions this is measurable duplicated work on every request.

**Fix:** if this shows up in profiling, serialize once and thread the result through; it is not
worth restructuring on speculation.

## 9. Backpressure counter is global, locks are per-checkpoint

**Severity:** low · **Status:** fixed — counters are per checkpoint · `engine.py`

`Engine._pending` counts in-flight requests across *all* checkpoints, but each checkpoint has its
own lock. A queue building up on `english` therefore makes requests for an idle `multilingual`
return 529, even though that checkpoint could serve immediately.

**Fix:** track pending counts per checkpoint, or document the global limit as intentional
whole-process backpressure.

## 10. Checkpoint release date is hardcoded

**Severity:** low · **Status:** fixed — date keyed by laya version, with a test that fails on an unrecorded one · `models.py`

`CHECKPOINTS_RELEASED = "2026-09-20"` is laya 0.3.4's PyPI upload date, used as `release_date` for
every entry in `GET /v1/models`. `pyproject.toml` pins `laya>=0.3.4`, so `uv lock --upgrade` can
install a newer laya and leave this reporting a stale date. `test_models.py` only checks the
string's shape, so nothing catches the drift.

**Fix:** a test asserting the constant matches the installed laya version's release, or at minimum
derive the value from `importlib.metadata.version("laya")` so the coupling is explicit.

## 11. Malformed `LAYA_SERVER_MAX_QUEUE` crashes startup

**Severity:** low · **Status:** fixed — `_int_env` names the variable and the value · `config.py`

`int(env.get("LAYA_SERVER_MAX_QUEUE", ...))` raises a bare `ValueError` for a non-numeric value,
producing a traceback rather than a message naming the offending variable. Compare
`LAYA_SERVER_PRELOAD`, where laya fails fast with a clear message listing the valid names.

**Fix:** catch the conversion error and re-raise with the variable name and the value received.

## 12. `/ready` reports one arbitrary agent's device as the server's

**Severity:** medium · **Status:** fixed — a `devices` map per checkpoint; `device` summarises or reports `mixed` · `engine.py`

`status()` reads `.device` from `loaded[0]` only. `Agent.device` is per-agent, and laya reassigns
it mid-request on an OOM fallback (`agent.py:278-290`), so `english` can be on `cpu` while
`multilingual` is still on `cuda`. The response presents one agent's device as the whole server's.

The flapping described here is gone: issue 6's fix sorts `loaded` and reads without `load()`, so
the sampled agent is now deterministic. What remains is that one agent's device still stands in
for the whole server. The docstring says the point is to surface laya's
silent downgrades; sampling one agent hides them for every other checkpoint.

**Fix:** report a device per checkpoint, or the distinct set of devices in use.

## 13. `/ready`'s 503 branch is unreachable in the container

**Severity:** medium · **Status:** fixed — claims corrected in the Dockerfile and README; see the note below · `app.py`, `Dockerfile`, `README.md`

`Engine.create()` runs in the lifespan hook, and uvicorn awaits lifespan startup
(`server.py:116`) *before* creating the listening socket (`server.py:154`). While checkpoints
download, there is no socket, so the in-container probe gets connection-refused — never a 503.
Confirmed earlier by reading the container's health log: three `Connection refused` probes, then
success.

`LAYA_SERVER_PRELOAD=""` also falls back to all three checkpoints (`config.py`), so `loaded` is
never empty after startup either. The `Dockerfile` comment and the README's "turns healthy only
once a checkpoint is resident" therefore describe behavior the old `/openapi.json` probe already
had; `/ready` adds diagnostics, not a new guarantee.

**Fix applied:** the claims in the `Dockerfile` and README now describe what actually happens —
connection-refused during the first download, with `503` applying once the socket is open.

Moving preload off the startup path was considered and rejected. It would let requests arrive
before a checkpoint is resident, pushing laya's unsynchronised cold load (`Router.load`, 10-20s)
onto the request path on the event loop — the exact hazard the preload design exists to avoid.
Making `/ready` meaningful during startup would require rejecting inference while not ready, which
is a larger change than the reporting inaccuracy warranted.

## 14. `engine.status()` is unguarded, so probe failures are 500

**Severity:** low · **Status:** fixed — `status()` wrapped; a probe failure is 503 · `app.py`

`Engine.__init__` accepts `router: Any`. A router without `.loaded` makes `/ready` raise
`AttributeError`, which returns a 500 traceback rather than "not ready". Verified: `/ready` with
a plain `FakeRouter` returns **HTTP 500** where 503 is correct.

**Fix:** wrap the `status()` call in `try/except Exception` and return 503 on failure — a probe
should report unhealthy, never crash.

## 15. The router test double hides issue 6

**Severity:** medium · **Status:** fixed — fixed alongside 6; the fake now mirrors `_order`/`_touch` · `tests/conftest.py`

`LoadedFakeRouter.loaded` returns a fixed list and `FakeRouter.load()` never reorders it, whereas
the real `Router.loaded` is LRU-ordered and `load()` reorders it. `test_health.py` asserts
`loaded == ["english", "multilingual"]` and passes, while the real server returns a different
order on the second probe. The double is more forgiving than the thing it stands in for, which is
why issue 6 reached this review instead of being caught by the suite.

**Fix:** mirror `_order`/`_touch` semantics in the fake, then add a test asserting two consecutive
`/ready` calls report the same order.

## 16. `/ready` discloses configuration without auth

**Severity:** low · **Status:** fixed — detail withheld when a key is configured · `app.py`

With `LAYA_SERVER_API_KEY` set, `/v1/models` — a static catalog — sits behind the key, while
`/ready` returns the resolved device and the resident/available checkpoint inventory to any
unauthenticated caller. The asymmetry is worth a deliberate decision rather than an accident.

**Fix:** if the detail is wanted for orchestrators, keep it and note the choice; otherwise return
a bare `{"status": "ready"}` when auth is enabled.

## 17. CI job display names changed

**Severity:** low · **Status:** fixed — required check names documented in the README · `.github/workflows/tests.yml`

The job keeps the id `fast`, but its display name went from `fast suite` to `tests (py3.12)` /
`tests (py3.13)`, and lint moved to a new `lint` job. GitHub branch protection matches required
checks by *name*. If a rule requires "fast suite", it will never report again and PRs will sit
pending forever.

No protection rule exists yet — the repo was pushed recently — so this is a note for whenever one
is added, not a live breakage.

**Fix:** set branch protection to require `lint and types`, `tests (py3.12)` and `tests (py3.13)`.

## 18. Model names duplicated as literals

**Severity:** low · **Status:** fixed — `_DESCRIPTIONS` built from the engine constants · `models.py`

`_DESCRIPTIONS` restates the names in `engine.JEV_MODELS` / `engine.LAYA_MODELS`. Adding a model
to `engine.py` alone would make `/v1/models` under-report a name the API accepts.
`test_models.py` catches that in CI, so this is a maintainability trap rather than a live bug.

**Fix:** build the mapping keyed off the engine constants so the two cannot diverge.
