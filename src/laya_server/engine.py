"""Owns the Laya Router and decides how requests share it.

Laya 0.3.4 had no locking at all, and this module existed to make it safe.
Laya 0.4 locks the Router's lifecycle itself (router.py:713-727, 851-970) and
makes inference on one Agent re-entrant (agent.py:166-211; common.py:41-91),
so what remains here is policy rather than protection:

  * every checkpoint is preloaded at startup, and max_loaded is raised to cover
    them, so no request pays for a build and nothing is evicted and rebuilt;
  * each checkpoint has its own lock, so requests for one checkpoint run one
    at a time -- bounding CPU threads and GPU memory, and giving the queue
    depth limit something to count -- while different checkpoints run at once;
  * routing, budget checks and the forward pass are offloaded, keeping the
    event loop free for other requests and the probes.

This is still the only module that touches the Router.
"""

import asyncio
import logging
from collections.abc import Mapping
from typing import Any, NamedTuple

from starlette.concurrency import run_in_threadpool

from .config import ALL_CHECKPOINTS, Settings
from .errors import Overloaded, ValidationFailed
from .translate import to_jev_response
from .validation import jev_questions, validate_budget, validate_questions

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
        # checkpoint to make room (router.py:944-970), rebuilding it from
        # scratch on its next request.
        router = laya.Router(
            device=settings.device,
            max_loaded=len(ALL_CHECKPOINTS),
            # Explicit, never laya's own: its default moved from english to
            # multilingual in 0.4.0, which would silently reroute requests
            # whose language detection abstains.
            default=settings.default_checkpoint,
        )
        router.preload(list(settings.preload))
        return cls(router, settings)

    def status(self) -> dict[str, Any]:
        """What is resident and on which device, for the readiness probe.

        The device is read from a loaded agent rather than from settings,
        because Laya downgrades an unavailable cuda/mps to cpu with only a
        RuntimeWarning on stderr (agent.py:633-659). Reporting the configured
        value would hide that.
        """
        # Sorted, and read without Router.load(): load() calls _touch(), which
        # reorders the very list `loaded` reports (router.py:938-942,
        # 1087-1090), so a probe would change what the next probe sees. Reading
        # the agent store directly is observation without mutation; Laya
        # offers no public accessor for it.
        loaded = sorted(self._router.loaded)
        agents = getattr(self._router, "_agents", {})

        # Per checkpoint: Agent.device is per agent, so one checkpoint can be
        # on cpu while another is still on cuda. An OOM fallback during a
        # request is now scoped to it and restored afterwards
        # (agent.py:1255-1303); only a failed restore sticks (agent.py:957-962).
        # Reporting a single agent's device would hide exactly the downgrade
        # this is here to show.
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
        # Only Jev's fields go further: Laya's own extensions would change
        # what the model is shown (see validation.JEV_QUESTION_FIELDS).
        questions = jev_questions(questions)

        # Routing is pure and loads nothing (router.py:1112-1236), so it happens
        # outside the lock: that is what lets different checkpoints run at once.
        # Offloaded because language detection now reads every leaf of a
        # structured state (lang.py:808-860) where 0.3.4 stopped at 4000 chars.
        decision = await run_in_threadpool(
            self._router.route, state, questions, model=model
        )
        checkpoint = decision["model"]
        agent = self._router.load(checkpoint)

        # Offloaded: tokenizing a large state is CPU-bound and would otherwise
        # block the event loop, stalling every other request and both probes.
        await run_in_threadpool(validate_budget, agent, state, questions, checkpoint)

        if self._pending.get(checkpoint, 0) >= self._settings.max_queue_depth:
            raise Overloaded(
                f"checkpoint {checkpoint!r} is at capacity; retry with backoff"
            )

        self._pending[checkpoint] = self._pending.get(checkpoint, 0) + 1
        try:
            async with self._lock_for(checkpoint):
                try:
                    raw = await self._run_inference(agent, state, questions)
                except (ValueError, KeyError, AttributeError, TypeError) as exc:
                    # Laya has no exception types of its own: input it refuses
                    # arrives as ValueError or TypeError (agent.py:967-1106,
                    # 1467-1479), and validation should have caught it first.
                    # This is the backstop for whatever it has not learned
                    # about yet. The request caused it, so report 422 rather
                    # than letting a caller trigger a 500. RuntimeError is left
                    # alone: that is where genuine faults arrive.
                    raise ValidationFailed(
                        f"model rejected the request: {type(exc).__name__}: {exc}"
                    ) from exc
        finally:
            self._pending[checkpoint] -= 1

        return Prediction(
            body=to_jev_response(
                raw, requested_model=requested_model, questions=questions
            ),
            checkpoint=checkpoint,
            reason=decision.get("reason", ""),
        )
