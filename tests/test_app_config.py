import logging
import secrets

from fastapi.testclient import TestClient

from laya_server import app as app_module
from laya_server.app import create_app
from laya_server.config import Settings
from laya_server.engine import Engine
from tests.conftest import LoadedFakeRouter


def build(env):
    settings = Settings.from_env(env)
    return settings, create_app(settings, engine=Engine(LoadedFakeRouter(), settings))


class TestLogLevel:
    """Issue 3: LAYA_SERVER_LOG_LEVEL was parsed and documented but applied
    nowhere, so setting it did nothing."""

    def test_configured_level_is_applied_to_the_server_logger(self):
        build({"LAYA_SERVER_LOG_LEVEL": "DEBUG"})
        assert logging.getLogger("laya_server").getEffectiveLevel() == logging.DEBUG

    def test_default_level_is_info(self):
        build({})
        assert logging.getLogger("laya_server").getEffectiveLevel() == logging.INFO

    def test_lowercase_is_accepted(self):
        build({"LAYA_SERVER_LOG_LEVEL": "warning"})
        assert logging.getLogger("laya_server").getEffectiveLevel() == logging.WARNING

    def test_an_unknown_level_falls_back_to_info_without_crashing(self):
        build({"LAYA_SERVER_LOG_LEVEL": "chatty"})
        assert logging.getLogger("laya_server").getEffectiveLevel() == logging.INFO


class TestConstantTimeAuth:
    """Issue 4: `!=` on str short-circuits at the first differing byte."""

    def test_uses_a_constant_time_comparison(self, monkeypatch):
        calls = []
        real = secrets.compare_digest

        def spy(a, b):
            calls.append((a, b))
            return real(a, b)

        monkeypatch.setattr(app_module.secrets, "compare_digest", spy)
        _, app = build({"LAYA_SERVER_API_KEY": "secret"})
        TestClient(app).get("/v1/models", headers={"Authorization": "Bearer secret"})
        assert calls, "auth must not compare the key with =="

    def test_correct_key_still_accepted(self):
        _, app = build({"LAYA_SERVER_API_KEY": "secret"})
        r = TestClient(app).get(
            "/v1/models", headers={"Authorization": "Bearer secret"}
        )
        assert r.status_code == 200

    def test_wrong_key_still_rejected(self):
        _, app = build({"LAYA_SERVER_API_KEY": "secret"})
        r = TestClient(app).get("/v1/models", headers={"Authorization": "Bearer nope"})
        assert r.status_code == 401

    def test_missing_header_still_rejected(self):
        _, app = build({"LAYA_SERVER_API_KEY": "secret"})
        assert TestClient(app).get("/v1/models").status_code == 401
