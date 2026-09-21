"""End-to-end tests against a real Laya checkpoint.

Marked slow: these download and load model weights. Run with `uv run pytest -m slow`.
"""

import pytest
from fastapi.testclient import TestClient

from laya_server.app import create_app
from laya_server.config import Settings
from laya_server.engine import Engine

pytestmark = pytest.mark.slow

REQUEST = {
    "model": "jev-latest",
    "state": {
        "from": "user@acme.com",
        "subject": "Duplicate charge on invoice #4411",
        "body": "We were billed twice for March. Refund the duplicate or we cancel.",
    },
    "questions": {
        "department": {
            "type": "choice",
            "instructions": "Which department should handle this request?",
            "criteria": {
                "billing": "invoices, payments, refunds",
                "technical": "bugs, outages, system errors",
                "sales": "pricing, new contracts",
            },
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this request?",
            "criteria": ["not urgent", "soon", "critical deadline"],
        },
        "refund_requested": {
            "type": "noul",
            "instructions": "Does the user explicitly request a refund?",
        },
    },
}


@pytest.fixture(scope="module")
def live_client():
    settings = Settings.from_env({"LAYA_SERVER_PRELOAD": "english"})
    return TestClient(create_app(settings, engine=Engine.create(settings)))


def test_answers_all_three_primitives(live_client):
    r = live_client.post("/v1/systemone", json=REQUEST)
    assert r.status_code == 200, r.text
    body = r.json()

    assert set(body) == {"model", "answers", "usage"}
    assert body["model"] == "jev-latest"
    assert set(body["answers"]) == {"department", "urgency", "refund_requested"}


def test_answer_shapes_match_jev_contract(live_client):
    answers = live_client.post("/v1/systemone", json=REQUEST).json()["answers"]

    assert set(answers["department"]) == {
        "type",
        "choice",
        "probabilities",
        "confidence",
    }
    assert set(answers["urgency"]) == {
        "type",
        "score",
        "legend",
        "probabilities",
        "confidence",
    }
    assert set(answers["refund_requested"]) == {"type", "noul"}


def test_answers_are_substantively_correct(live_client):
    answers = live_client.post("/v1/systemone", json=REQUEST).json()["answers"]

    assert answers["department"]["choice"] == "billing"
    assert answers["refund_requested"]["noul"] > 0.5


def test_reports_the_serving_checkpoint(live_client):
    r = live_client.post("/v1/systemone", json=REQUEST)
    assert r.headers["X-Laya-Checkpoint"] == "english"
