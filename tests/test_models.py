import json

import pytest
from fastapi.testclient import TestClient
from typesafe_sdk import ListModelsResponse

from laya_server.app import create_app
from laya_server.config import Settings
from laya_server.engine import JEV_MODELS, LAYA_MODELS, Engine
from tests.conftest import LoadedFakeRouter

AUTH_ENV = {"LAYA_SERVER_API_KEY": "secret"}

MODELS_PATH = "/v1/models"


def client(env=None):
    settings = Settings.from_env(env or {})
    engine = Engine(LoadedFakeRouter(), settings)
    return TestClient(create_app(settings, engine=engine))


class TestContract:
    def test_returns_a_models_array(self):
        r = client().get(MODELS_PATH)
        assert r.status_code == 200
        assert set(r.json()) == {"models"}

    def test_every_entry_has_the_documented_fields(self):
        for m in client().get(MODELS_PATH).json()["models"]:
            assert set(m) == {"name", "description", "release_date"}

    def test_the_real_sdk_parses_the_response(self):
        """Parsed exactly as the SDK does, from raw JSON bytes."""
        body = client().get(MODELS_PATH).json()
        parsed = ListModelsResponse.model_validate_json(json.dumps(body))
        assert {m.name for m in parsed.models}


class TestListedNames:
    def test_lists_every_name_the_server_accepts(self):
        """`name` is documented as a value the `model` field accepts, so the
        list and the accepted set must not drift apart."""
        listed = {m["name"] for m in client().get(MODELS_PATH).json()["models"]}
        assert listed == set(JEV_MODELS) | set(LAYA_MODELS)

    @pytest.mark.parametrize("name", JEV_MODELS + LAYA_MODELS)
    def test_each_listed_name_is_actually_usable(self, name):
        c = client()
        listed = {m["name"] for m in c.get(MODELS_PATH).json()["models"]}
        assert name in listed

        r = c.post(
            "/v1/systemone",
            json={
                "model": name,
                "state": "billed twice",
                "questions": {"q": {"type": "noul", "instructions": "Refund?"}},
            },
        )
        assert r.status_code == 200, f"{name} is listed but rejected: {r.text}"

    def test_release_date_is_an_iso_date(self):
        for m in client().get(MODELS_PATH).json()["models"]:
            assert len(m["release_date"]) == 10
            assert m["release_date"][4] == m["release_date"][7] == "-"


class TestAuth:
    def test_requires_a_key_when_one_is_configured(self):
        """Jev sends Authorization on this route, so it is not a public probe."""
        r = client(AUTH_ENV).get(MODELS_PATH)
        assert r.status_code == 401
        assert r.json()["error"]["type"] == "authentication_error"

    def test_accepts_the_correct_key(self):
        r = client(AUTH_ENV).get(
            MODELS_PATH, headers={"Authorization": "Bearer secret"}
        )
        assert r.status_code == 200

    def test_open_when_no_key_is_configured(self):
        assert client().get(MODELS_PATH).status_code == 200


class TestCatalogDerivedFromEngine:
    """Issue 18: the names were restated as literals in models.py."""

    def test_descriptions_are_keyed_off_the_engine_constants(self):
        from laya_server.models import _DESCRIPTIONS

        assert set(_DESCRIPTIONS) == set(JEV_MODELS) | set(LAYA_MODELS)


class TestReleaseDateTracksLaya:
    """Issue 10: a hand-maintained date under a `>=` pin drifts silently."""

    def test_the_installed_laya_version_has_a_recorded_release_date(self):
        from importlib.metadata import version

        from laya_server.models import LAYA_RELEASE_DATES

        installed = version("laya")
        assert installed in LAYA_RELEASE_DATES, (
            f"laya {installed} has no recorded release date; add it to "
            "LAYA_RELEASE_DATES so /v1/models stops reporting a stale one"
        )

    def test_reported_date_matches_the_installed_version(self):
        from importlib.metadata import version

        from laya_server.models import LAYA_RELEASE_DATES, list_models

        expected = LAYA_RELEASE_DATES[version("laya")]
        assert {m["release_date"] for m in list_models()["models"]} == {expected}
