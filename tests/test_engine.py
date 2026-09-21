import asyncio
import time

import pytest

from laya_server.engine import Engine
from laya_server.errors import Overloaded, ValidationFailed
from tests.conftest import FakeInferenceAgent, FakeRouter

NOUL = {"type": "noul", "instructions": "Is a refund requested?"}
QUESTIONS = {"refund": NOUL}


def engine(router=None, **kw):
    from laya_server.config import Settings

    return Engine(router or FakeRouter(), Settings.from_env({}), **kw)


def predict(eng, model="jev-latest", state="billed twice", questions=QUESTIONS):
    return asyncio.run(eng.predict(state, questions, requested_model=model))


class TestModelMapping:
    @pytest.mark.parametrize("alias", ["jev-latest", "jev-preview", "jev-1.13.0"])
    def test_jev_aliases_delegate_routing_to_laya(self, alias):
        """A jev-* id means 'pick for me', so Laya's detection must decide."""
        router = FakeRouter()
        predict(engine(router), model=alias)
        assert router.route_calls[0]["model"] is None

    def test_native_checkpoint_name_is_an_explicit_override(self):
        router = FakeRouter()
        predict(engine(router), model="multilingual")
        assert router.route_calls[0]["model"] == "multilingual"

    def test_unknown_model_rejected(self):
        with pytest.raises(ValidationFailed) as exc:
            predict(engine(), model="gpt-4")
        assert "gpt-4" in str(exc.value)


class TestPrediction:
    def test_returns_jev_shaped_body(self):
        result = predict(engine())
        assert result.body == {
            "model": "jev-latest",
            "answers": {"refund": {"type": "noul", "noul": 0.91}},
            "usage": {"input_tokens": 42, "output_tokens": 0},
        }

    def test_reports_checkpoint_that_served_the_request(self):
        result = predict(engine(FakeRouter(checkpoint="multilingual")))
        assert result.checkpoint == "multilingual"

    def test_rejects_oversized_state_before_running_inference(self):
        router = FakeRouter()
        with pytest.raises(ValidationFailed):
            predict(engine(router), state=" ".join(["word"] * 600))
        assert router.agents["english"].calls == [], "inference must not have run"


class TestConcurrency:
    def test_same_checkpoint_requests_are_serialized(self):
        """Laya has no internal locking; two concurrent forwards on one Agent
        must not overlap."""
        overlap = {"current": 0, "peak": 0}

        def track():
            overlap["current"] += 1
            overlap["peak"] = max(overlap["peak"], overlap["current"])
            # Inference runs in a real threadpool thread; hold the window open
            # long enough that unsynchronized calls would genuinely overlap.
            time.sleep(0.02)
            overlap["current"] -= 1

        agent = FakeInferenceAgent(on_call=track)
        router = FakeRouter(agents={"english": agent})
        eng = engine(router)

        async def race():
            await asyncio.gather(
                *(eng.predict("x", QUESTIONS, "jev-latest") for _ in range(6))
            )

        asyncio.run(race())
        assert overlap["peak"] == 1
        assert len(agent.calls) == 6

    def test_different_checkpoints_are_not_blocked_by_each_other(self):
        eng = engine()
        order = []

        async def run(name):
            await eng.predict("x", QUESTIONS, requested_model=name)
            order.append(name)

        async def race():
            await asyncio.gather(run("english"), run("multilingual"))

        asyncio.run(race())
        assert set(order) == {"english", "multilingual"}


async def _saturate(eng, count, checkpoint):
    """Start `count` requests and wait until the engine counts them as pending.

    `predict` awaits the threadpool for budget validation before it registers
    as pending, so a single `await asyncio.sleep(0)` does not mean the requests
    have arrived; the assertion that follows would then be racy. Waiting on the
    counter is also the right condition rather than waiting for inference to
    start: the per-checkpoint lock means only the first request reaches the
    forward pass, while the rest queue behind it -- all of them pending.
    """
    tasks = [
        asyncio.create_task(eng.predict("x", QUESTIONS, checkpoint))
        for _ in range(count)
    ]
    async with asyncio.timeout(10):
        while sum(eng._pending.values()) < count:
            await asyncio.sleep(0.01)
    return tasks


async def _drain(tasks, release):
    """Let the blocked requests finish, then await them. Cancelling a task that
    is inside the threadpool can hang; releasing it cannot."""
    release.set()
    async with asyncio.timeout(10):
        await asyncio.gather(*tasks, return_exceptions=True)


class TestBackpressure:
    def test_saturated_queue_is_rejected_as_overloaded(self):
        from laya_server.config import Settings

        eng = Engine(FakeRouter(), Settings.from_env({"LAYA_SERVER_MAX_QUEUE": "2"}))

        async def scenario():
            release = asyncio.Event()

            async def slow(*_):
                await release.wait()
                return None

            eng._run_inference = slow  # block every in-flight request
            inflight = await _saturate(eng, 2, "jev-latest")

            with pytest.raises(Overloaded):
                await eng.predict("x", QUESTIONS, "jev-latest")

            await _drain(inflight, release)

        asyncio.run(scenario())


class TestRouterConstruction:
    def test_max_loaded_covers_all_checkpoints_with_subset_preload(self, monkeypatch):
        """Laya evicts LRU when max_loaded is tight (router.py:188-195), and an
        evicted checkpoint is rebuilt from scratch on its next request. Routing
        can reach any checkpoint regardless of what was preloaded, so the cache
        must be sized for all of them, not just the preloaded subset."""
        import laya

        from laya_server.config import ALL_CHECKPOINTS, Settings

        built = {}

        class RecordingRouter(FakeRouter):
            def __init__(self, device=None, max_loaded=None):
                super().__init__()
                built["max_loaded"] = max_loaded
                built["device"] = device

        monkeypatch.setattr(laya, "Router", RecordingRouter)

        Engine.create(Settings.from_env({"LAYA_SERVER_PRELOAD": "english"}))

        assert built["max_loaded"] >= len(ALL_CHECKPOINTS)

    def test_preloads_only_what_was_configured(self, monkeypatch):
        import laya

        from laya_server.config import Settings

        routers = []

        class RecordingRouter(FakeRouter):
            def __init__(self, device=None, max_loaded=None):
                super().__init__()
                routers.append(self)

        monkeypatch.setattr(laya, "Router", RecordingRouter)

        Engine.create(Settings.from_env({"LAYA_SERVER_PRELOAD": "english"}))

        assert routers[0].preloaded == ["english"]


class TestLayaErrorsBecome422:
    """Issue 1 backstop: Laya raises bare ValueError/KeyError for inputs the
    validation layer has not learned about yet. None of them should reach the
    client as a 500."""

    def _engine_raising(self, exc):
        agent = FakeInferenceAgent()

        def boom(*_a, **_kw):
            raise exc

        agent.system_one = boom
        return engine(FakeRouter(agents={"english": agent}))

    @pytest.mark.parametrize(
        "exc",
        [
            ValueError("question 'pick' options exceed head_max_len=192"),
            KeyError("type"),
            AttributeError("'NoneType' object has no attribute 'items'"),
        ],
    )
    def test_laya_failure_is_reported_as_validation_failure(self, exc):
        with pytest.raises(ValidationFailed):
            predict(self._engine_raising(exc))

    def test_the_message_mentions_the_underlying_problem(self):
        exc = ValueError("question 'pick' options exceed head_max_len=192")
        with pytest.raises(ValidationFailed) as got:
            predict(self._engine_raising(exc))
        assert "head_max_len" in str(got.value)

    def test_unrelated_runtime_errors_still_propagate(self):
        """A genuine server fault must stay a 500, not be mislabelled a client
        error."""
        with pytest.raises(RuntimeError):
            predict(self._engine_raising(RuntimeError("CUDA out of memory")))


class TestStateSerializedOnce:
    """Issue 8: the state was JSON-serialized in validation and again by laya,
    once per question."""

    def test_inference_receives_the_already_serialized_state(self):
        agent = FakeInferenceAgent()
        router = FakeRouter(agents={"english": agent})
        predict(engine(router), state={"body": "billed twice"})
        sent_state, _ = agent.calls[0]
        assert isinstance(sent_state, str), (
            "laya would re-serialize a dict per question"
        )
        assert "billed twice" in sent_state

    def test_routing_still_sees_the_original_structure(self):
        """Language detection walks dict/list leaves and ignores keys
        (lang.py:67-88), so it must not be handed the JSON text."""
        router = FakeRouter()
        predict(engine(router), state={"body": "billed twice"})
        assert router.route_calls[0]["state"] == {"body": "billed twice"}


class TestPerCheckpointBackpressure:
    """Issue 9: one global counter meant a queue on one checkpoint returned 529
    for a different, idle one."""

    def test_a_busy_checkpoint_does_not_reject_an_idle_one(self):
        from laya_server.config import Settings

        eng = Engine(FakeRouter(), Settings.from_env({"LAYA_SERVER_MAX_QUEUE": "2"}))

        async def scenario():
            release = asyncio.Event()

            async def slow(agent, state, questions):
                await release.wait()
                return None

            eng._run_inference = slow
            busy = await _saturate(eng, 2, "english")

            # english is saturated; multilingual is untouched and must serve.
            async def quick(agent, state, questions):
                return agent.system_one(state, questions)

            eng._run_inference = quick
            result = await eng.predict("x", QUESTIONS, "multilingual")
            assert result.checkpoint == "multilingual"

            await _drain(busy, release)

        asyncio.run(scenario())

    def test_a_saturated_checkpoint_still_rejects_its_own_overflow(self):
        from laya_server.config import Settings

        eng = Engine(FakeRouter(), Settings.from_env({"LAYA_SERVER_MAX_QUEUE": "2"}))

        async def scenario():
            release = asyncio.Event()

            async def slow(agent, state, questions):
                await release.wait()
                return None

            eng._run_inference = slow
            busy = await _saturate(eng, 2, "english")

            with pytest.raises(Overloaded):
                await eng.predict("x", QUESTIONS, "english")

            await _drain(busy, release)

        asyncio.run(scenario())
