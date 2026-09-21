import pytest
from fastapi.testclient import TestClient

from laya_server.app import create_app
from laya_server.config import Settings
from laya_server.engine import Engine
from tests.conftest import FakeRouter

NOUL = {"type": "noul", "instructions": "Is a refund requested?"}

VALID_REQUEST = {
    "model": "jev-latest",
    "state": {"body": "We were billed twice for March. Refund it or we cancel."},
    "questions": {"refund": NOUL},
}


def client(env=None, router=None):
    settings = Settings.from_env(env or {})
    engine = Engine(router or FakeRouter(), settings)
    return TestClient(create_app(settings, engine=engine))


class TestSuccessfulRequest:
    def test_returns_jev_response_body(self):
        r = client().post("/v1/systemone", json=VALID_REQUEST)
        assert r.status_code == 200
        assert r.json() == {
            "model": "jev-latest",
            "answers": {"refund": {"type": "noul", "noul": 0.91}},
            "usage": {"input_tokens": 42, "output_tokens": 0},
        }

    def test_body_carries_no_keys_jev_would_not_send(self):
        body = client().post("/v1/systemone", json=VALID_REQUEST).json()
        assert set(body) == {"model", "answers", "usage"}

    def test_reports_serving_checkpoint_in_header(self):
        r = client(router=FakeRouter(checkpoint="multilingual")).post(
            "/v1/systemone", json=VALID_REQUEST
        )
        assert r.headers["X-Laya-Checkpoint"] == "multilingual"

    def test_accepts_plain_string_state(self):
        payload = {**VALID_REQUEST, "state": "billed twice"}
        r = client().post("/v1/systemone", json=payload)
        assert r.status_code == 200

    def test_accepts_array_state(self):
        r = client().post("/v1/systemone", json={**VALID_REQUEST, "state": ["a", "b"]})
        assert r.status_code == 200


class TestRequestValidation:
    @pytest.mark.parametrize("missing", ["model", "state", "questions"])
    def test_missing_required_field_is_422(self, missing):
        payload = {k: v for k, v in VALID_REQUEST.items() if k != missing}
        assert client().post("/v1/systemone", json=payload).status_code == 422

    def test_malformed_question_is_422_not_500(self):
        """The whole point of the validation layer: Laya would raise KeyError."""
        payload = {**VALID_REQUEST, "questions": {"q": {"instructions": "no type"}}}
        r = client().post("/v1/systemone", json=payload)
        assert r.status_code == 422

    def test_error_body_has_type_and_message(self):
        payload = {**VALID_REQUEST, "questions": {"q": {"instructions": "no type"}}}
        body = client().post("/v1/systemone", json=payload).json()
        assert body["error"]["type"] == "validation_error"
        assert "q" in body["error"]["message"]

    def test_oversized_state_is_422(self):
        payload = {**VALID_REQUEST, "state": " ".join(["word"] * 600)}
        r = client().post("/v1/systemone", json=payload)
        assert r.status_code == 422
        assert "exceeds" in r.json()["error"]["message"]


class TestAuthDisabled:
    def test_no_key_configured_means_no_auth_required(self):
        assert client().post("/v1/systemone", json=VALID_REQUEST).status_code == 200


AUTH_ENV = {"LAYA_SERVER_API_KEY": "secret"}


class TestAuthEnabled:
    def test_missing_header_is_401(self):
        r = client(AUTH_ENV).post("/v1/systemone", json=VALID_REQUEST)
        assert r.status_code == 401
        assert r.json()["error"]["type"] == "authentication_error"

    def test_wrong_key_is_401(self):
        r = client(AUTH_ENV).post(
            "/v1/systemone",
            json=VALID_REQUEST,
            headers={"Authorization": "Bearer nope"},
        )
        assert r.status_code == 401

    def test_correct_bearer_token_is_accepted(self):
        r = client(AUTH_ENV).post(
            "/v1/systemone",
            json=VALID_REQUEST,
            headers={"Authorization": "Bearer secret"},
        )
        assert r.status_code == 200

    def test_auth_is_checked_before_request_validation(self):
        """An unauthenticated caller must not learn whether its body was valid."""
        r = client(AUTH_ENV).post("/v1/systemone", json={"garbage": True})
        assert r.status_code == 401


class TestOverload:
    def test_saturated_engine_returns_529(self):
        from laya_server.errors import Overloaded

        class FullEngine(Engine):
            async def predict(self, *a, **kw):
                raise Overloaded("server is at capacity; retry with backoff")

        settings = Settings.from_env({})
        app = create_app(settings, engine=FullEngine(FakeRouter(), settings))
        r = TestClient(app).post("/v1/systemone", json=VALID_REQUEST)
        assert r.status_code == 529
        assert r.json()["error"]["type"] == "overloaded_error"


class TestBodySizeLimit:
    """Issue 5: an unbounded body was read and then tokenized on the event
    loop, blocking every other request including the health probes."""

    def test_oversized_body_is_rejected(self):
        c = client({"LAYA_SERVER_MAX_BODY_BYTES": "1000"})
        payload = {**VALID_REQUEST, "state": "x" * 5000}
        r = c.post("/v1/systemone", json=payload)
        assert r.status_code == 422
        assert "too large" in r.json()["error"]["message"].lower()

    def test_the_message_reports_both_sizes(self):
        c = client({"LAYA_SERVER_MAX_BODY_BYTES": "1000"})
        msg = c.post(
            "/v1/systemone", json={**VALID_REQUEST, "state": "x" * 5000}
        ).json()["error"]["message"]
        assert "1000" in msg

    def test_a_normal_body_is_unaffected(self):
        assert (
            client({"LAYA_SERVER_MAX_BODY_BYTES": "1000000"})
            .post("/v1/systemone", json=VALID_REQUEST)
            .status_code
            == 200
        )

    def test_rejected_before_the_body_is_parsed(self):
        """Oversized bodies must not reach json parsing or tokenization."""
        c = client({"LAYA_SERVER_MAX_BODY_BYTES": "500"})
        r = c.post(
            "/v1/systemone",
            content=b"{" + b"x" * 5000,
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code == 422, "malformed oversized body must be size-rejected"
