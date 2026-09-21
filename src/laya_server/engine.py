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

from starlette.concurrency import run_in_threadpool

from .config import Settings
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
        self._pending = 0

    @classmethod
    def create(cls, settings: Settings) -> "Engine":
        """Build a Router with every configured checkpoint already resident."""
        import laya

        from .config import ALL_CHECKPOINTS

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

    def _lock_for(self, checkpoint: str) -> asyncio.Lock:
        return self._locks.setdefault(checkpoint, asyncio.Lock())

    async def _run_inference(self, agent: Any, state: Any, questions: Mapping) -> dict:
        return await run_in_threadpool(agent.system_one, state, questions)

    async def predict(
        self,
        state: str | dict | list,
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

        validate_budget(agent, state, questions, checkpoint)

        if self._pending >= self._settings.max_queue_depth:
            raise Overloaded("server is at capacity; retry with backoff")

        self._pending += 1
        try:
            async with self._lock_for(checkpoint):
                raw = await self._run_inference(agent, state, questions)
        finally:
            self._pending -= 1

        return Prediction(
            body=to_jev_response(raw, requested_model=requested_model),
            checkpoint=checkpoint,
            reason=decision.get("reason", ""),
        )
