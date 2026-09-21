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


class TestBackpressure:
    def test_saturated_queue_is_rejected_as_overloaded(self):
        from laya_server.config import Settings

        release = asyncio.Event()

        eng = Engine(FakeRouter(), Settings.from_env({"LAYA_SERVER_MAX_QUEUE": "2"}))

        async def scenario():
            async def slow(*_):
                await release.wait()
                return None

            eng._run_inference = slow  # block every in-flight request
            inflight = [
                asyncio.create_task(eng.predict("x", QUESTIONS, "jev-latest"))
                for _ in range(2)
            ]
            await asyncio.sleep(0)
            with pytest.raises(Overloaded):
                await eng.predict("x", QUESTIONS, "jev-latest")
            release.set()
            for t in inflight:
                t.cancel()

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
