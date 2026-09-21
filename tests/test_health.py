import threading
import time

from fastapi.testclient import TestClient

from laya_server.app import create_app
from laya_server.config import Settings
from laya_server.engine import Engine
from tests.conftest import FakeInferenceAgent, LoadedFakeRouter

AUTH_ENV = {"LAYA_SERVER_API_KEY": "secret"}


def client(env=None, router=None, engine="build"):
    settings = Settings.from_env(env or {})
    if engine == "build":
        engine = Engine(router or LoadedFakeRouter(), settings)
    return TestClient(create_app(settings, engine=engine))


class TestHealth:
    def test_returns_ok(self):
        r = client().get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}

    def test_needs_no_api_key(self):
        """Orchestrators probe liveness without credentials."""
        assert client(AUTH_ENV).get("/health").status_code == 200

    def test_is_ok_even_before_the_engine_exists(self):
        """Liveness answers whether the process is up, nothing more."""
        assert client(engine=None).get("/health").status_code == 200


class TestReady:
    def test_reports_loaded_checkpoints_and_device(self):
        router = LoadedFakeRouter(loaded=("english", "multilingual"))
        r = client(router=router).get("/ready")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ready"
        assert body["checkpoints"]["loaded"] == ["english", "multilingual"]
        assert body["checkpoints"]["available"] == [
            "english",
            "multilingual",
            "typed-decisions",
        ]

    def test_reports_the_device_actually_in_use(self):
        """Laya silently downgrades an unavailable cuda to cpu, so report what
        the loaded agent really has rather than what was configured."""
        router = LoadedFakeRouter(loaded=("english",))
        router.agents["english"] = FakeInferenceAgent(device="cpu")
        c = client(env={"LAYA_SERVER_DEVICE": "cuda"}, router=router)
        body = c.get("/ready").json()
        assert body["device"] == "cpu"

    def test_is_503_before_the_engine_exists(self):
        r = client(engine=None).get("/ready")
        assert r.status_code == 503
        assert r.json()["status"] == "not ready"

    def test_is_503_when_no_checkpoint_is_resident(self):
        r = client(router=LoadedFakeRouter(loaded=())).get("/ready")
        assert r.status_code == 503

    def test_needs_no_api_key(self):
        assert client(AUTH_ENV).get("/ready").status_code == 200


class TestProbesAreNotBlockedByInference:
    def test_health_answers_while_an_inference_holds_the_lock(self):
        """The whole point of probing without touching the engine: a server
        busy on a long forward pass must still report liveness."""
        entered = threading.Event()
        release = threading.Event()

        def block():
            entered.set()
            release.wait(timeout=10)

        router = LoadedFakeRouter(agents={"english": FakeInferenceAgent(on_call=block)})
        settings = Settings.from_env({})
        engine = Engine(router, settings)
        app = create_app(settings, engine=engine)

        with TestClient(app) as c:
            worker = threading.Thread(
                target=lambda: c.post(
                    "/v1/systemone",
                    json={
                        "model": "jev-latest",
                        "state": "x",
                        "questions": {"q": {"type": "noul", "instructions": "Urgent?"}},
                    },
                ),
                daemon=True,
            )
            worker.start()
            assert entered.wait(timeout=10), "inference never started"

            # The forward pass holds the lock for up to 10s. A probe that
            # waited on it would take about that long, so timing is the
            # assertion: it must answer immediately.
            started = time.monotonic()
            assert c.get("/health").status_code == 200
            assert c.get("/ready").status_code == 200
            elapsed = time.monotonic() - started

            release.set()
            worker.join(timeout=10)

            assert elapsed < 2.0, (
                f"probes took {elapsed:.1f}s while inference held the lock; "
                "they must not wait on it"
            )


class TestReadyIsSideEffectFree:
    """Issue 6: status() went through Router.load() purely to read .device,
    and load() reorders the LRU list that `loaded` reports."""

    def test_repeated_probes_report_a_stable_order(self):
        c = client(router=LoadedFakeRouter(loaded=("english", "multilingual")))
        orders = [c.get("/ready").json()["checkpoints"]["loaded"] for _ in range(4)]
        assert len({tuple(o) for o in orders}) == 1, f"order oscillates: {orders}"

    def test_probing_does_not_reorder_the_router(self):
        router = LoadedFakeRouter(loaded=("english", "multilingual"))
        before = router.loaded
        client(router=router).get("/ready")
        assert router.loaded == before, "a read-only probe must not mutate LRU order"

    def test_loaded_is_reported_in_a_stable_order(self):
        c = client(router=LoadedFakeRouter(loaded=("multilingual", "english")))
        assert c.get("/ready").json()["checkpoints"]["loaded"] == [
            "english",
            "multilingual",
        ]


class TestPerCheckpointDevices:
    """Issue 12: one agent's device stood in for the whole server, hiding a
    downgrade on any other checkpoint."""

    def _router_with_devices(self, **devices):
        router = LoadedFakeRouter(loaded=tuple(devices))
        for name, dev in devices.items():
            router.agents[name] = FakeInferenceAgent(device=dev)
        return router

    def test_reports_a_device_per_checkpoint(self):
        r = client(router=self._router_with_devices(english="cpu", multilingual="cuda"))
        assert r.get("/ready").json()["devices"] == {
            "english": "cpu",
            "multilingual": "cuda",
        }

    def test_summary_device_when_all_agree(self):
        r = client(router=self._router_with_devices(english="cpu", multilingual="cpu"))
        assert r.get("/ready").json()["device"] == "cpu"

    def test_summary_is_mixed_when_they_disagree(self):
        """A downgrade on one checkpoint must not be masked by another."""
        r = client(router=self._router_with_devices(english="cpu", multilingual="cuda"))
        assert r.get("/ready").json()["device"] == "mixed"


class TestProbeFailuresAre503:
    """Issue 14: an unguarded status() turned a probe failure into a 500."""

    def test_a_broken_router_yields_503_not_500(self):
        class BrokenRouter:
            pass  # no .loaded

        settings = Settings.from_env({})
        app = create_app(settings, engine=Engine(BrokenRouter(), settings))
        r = TestClient(app, raise_server_exceptions=False).get("/ready")
        assert r.status_code == 503
        assert r.json()["status"] == "not ready"


class TestReadyDisclosure:
    """Issue 16: /ready is auth-exempt, so with a key configured it leaked the
    device and checkpoint inventory that /v1/models sits behind."""

    def test_details_are_withheld_when_auth_is_enabled(self):
        body = client(AUTH_ENV).get("/ready").json()
        assert body == {"status": "ready"}

    def test_details_are_present_when_auth_is_disabled(self):
        body = client().get("/ready").json()
        assert "checkpoints" in body and "device" in body

    def test_still_returns_200_without_a_key(self):
        assert client(AUTH_ENV).get("/ready").status_code == 200

    def test_not_ready_is_still_reported_when_auth_is_enabled(self):
        r = client(AUTH_ENV, router=LoadedFakeRouter(loaded=())).get("/ready")
        assert r.status_code == 503
        assert r.json() == {"status": "not ready"}
