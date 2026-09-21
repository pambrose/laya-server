"""Owns the Laya Router and serializes access to it.

Laya 0.3.4 contains no locking of any kind. `Router.load` is check-then-act
(router.py:169-181), `_touch` can raise from a concurrent evict (router.py:183),
and the OOM fallback reassigns `self.device` mid-flight (agent.py:278-290).
So this module is the single owner of the Router:

  * every checkpoint is preloaded at startup, and max_loaded is raised to cover
    them, so eviction never runs while a request is in flight;
  * each checkpoint has its own lock, so one busy checkpoint cannot block
    another;
  * the blocking forward pass is offloaded, keeping the event loop free.
"""

import asyncio
import logging
from collections.abc import Mapping
from typing import Any, NamedTuple

from laya.common import serialize_state
from starlette.concurrency import run_in_threadpool

from .config import ALL_CHECKPOINTS, Settings
from .errors import Overloaded, ValidationFailed
from .translate import to_jev_response
from .validation import validate_budget, validate_questions

logger = logging.getLogger(__name__)

# Jev model ids. All resolve to the same service, so all mean "let Laya route".
JEV_MODELS = ("jev-latest", "jev-preview", "jev-1.13.0")

# Laya's own checkpoint names, accepted as an explicit override.
LAYA_MODELS = ("english", "multilingual", "typed-decisions")


class Prediction(NamedTuple):
    body: dict[str, Any]
    checkpoint: str
    reason: str


def resolve_model(requested: str) -> str | None:
    """Map a request's `model` onto Laya's `route(model=...)` argument.

    A jev-* id carries no checkpoint information, so it becomes None and Laya's
    script/language detection chooses. Laya's own names pass through.
    """
    if requested in JEV_MODELS:
        return None
    if requested in LAYA_MODELS:
        return requested
    raise ValidationFailed(
        f"unknown model {requested!r}; expected one of "
        f"{', '.join(JEV_MODELS + LAYA_MODELS)}"
    )


class Engine:
    def __init__(self, router: Any, settings: Settings):
        self._router = router
        self._settings = settings
        self._locks: dict[str, asyncio.Lock] = {}
        # Per checkpoint, not global: the locks are per checkpoint, so a queue
        # on one must not reject requests for an idle other.
        self._pending: dict[str, int] = {}

    @classmethod
    def create(cls, settings: Settings) -> "Engine":
        """Build a Router with every configured checkpoint already resident."""
        import laya

        # Sized for every checkpoint, not just the preloaded ones: routing can
        # reach any of them, and a tight max_loaded would evict a resident
        # checkpoint to make room (router.py:188-195), rebuilding it from
        # scratch on its next request.
        router = laya.Router(
            device=settings.device,
            max_loaded=len(ALL_CHECKPOINTS),
        )
        router.preload(list(settings.preload))
        return cls(router, settings)

    def status(self) -> dict[str, Any]:
        """What is resident and on which device, for the readiness probe.

        The device is read from a loaded agent rather than from settings,
        because Laya downgrades an unavailable cuda/mps to cpu with only a
        print to stdout (agent.py:159,163). Reporting the configured value
        would hide that.
        """
        # Sorted, and read without Router.load(): load() calls _touch(), which
        # reorders the very list `loaded` reports (router.py:169-185), so a
        # probe would change what the next probe sees. Reading the agent store
        # directly is observation without mutation.
        loaded = sorted(self._router.loaded)
        agents = getattr(self._router, "_agents", {})

        # Per checkpoint: Agent.device is per agent, and laya reassigns it
        # mid-request on an OOM fallback (agent.py:278-290), so one checkpoint
        # can be on cpu while another is still on cuda. Reporting a single
        # agent's device would hide exactly the downgrade this is here to show.
        devices = {
            name: str(getattr(agents[name], "device", self._settings.device))
            for name in loaded
            if name in agents
        }
        distinct = set(devices.values())
        summary: str | None
        if len(distinct) == 1:
            summary = distinct.pop()
        elif not distinct:
            summary = self._settings.device
        else:
            summary = "mixed"

        return {
            "device": summary,
            "devices": devices,
            "checkpoints": {"loaded": loaded, "available": list(ALL_CHECKPOINTS)},
        }

    def _lock_for(self, checkpoint: str) -> asyncio.Lock:
        return self._locks.setdefault(checkpoint, asyncio.Lock())

    async def _run_inference(
        self, agent: Any, state: Any, questions: Mapping[str, Any]
    ) -> dict[str, Any]:
        return await run_in_threadpool(agent.system_one, state, questions)

    async def predict(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: Mapping[str, Any],
        requested_model: str,
    ) -> Prediction:
        validate_questions(questions)
        model = resolve_model(requested_model)

        # Routing is pure and loads nothing (router.py:241-290), so it happens
        # outside the lock: that is what lets different checkpoints run at once.
        decision = self._router.route(state, questions, model=model)
        checkpoint = decision["model"]
        agent = self._router.load(checkpoint)

        # Offloaded: tokenizing a large state is CPU-bound and would otherwise
        # block the event loop, stalling every other request and both probes.
        await run_in_threadpool(validate_budget, agent, state, questions, checkpoint)

        if self._pending.get(checkpoint, 0) >= self._settings.max_queue_depth:
            raise Overloaded(
                f"checkpoint {checkpoint!r} is at capacity; retry with backoff"
            )

        # Serialized once here rather than once per question inside laya.
        # Routing above still saw the original structure, which matters because
        # language detection walks dict/list leaves and ignores keys.
        serialized = serialize_state(state)

        self._pending[checkpoint] = self._pending.get(checkpoint, 0) + 1
        try:
            async with self._lock_for(checkpoint):
                try:
                    raw = await self._run_inference(agent, serialized, questions)
                except (ValueError, KeyError, AttributeError, TypeError) as exc:
                    # Laya has no exception types and no input validation, so a
                    # request it cannot encode surfaces as one of these. They are
                    # caused by the request, not the server, so report 422 rather
                    # than letting a caller trigger a 500. RuntimeError is left
                    # alone: that is where genuine faults such as OOM arrive.
                    raise ValidationFailed(
                        f"model rejected the request: {type(exc).__name__}: {exc}"
                    ) from exc
        finally:
            self._pending[checkpoint] -= 1

        return Prediction(
            body=to_jev_response(raw, requested_model=requested_model),
            checkpoint=checkpoint,
            reason=decision.get("reason", ""),
        )
