"""Pin wire compatibility against TypeSafe's own SDK response models.

These parse our output with the models the real client uses, so a drift in
types or missing fields fails here rather than in a user's migrated script.

They prove the SDK *can* parse what we send, not that we send exactly Jev's
fields: the SDK's models ignore unknown keys, so a stray `action` block would
slip through here. `test_translate.py` and `test_api.py` cover exactness.
"""

import json

from typesafe_sdk import SystemOneResponse

from laya_server.translate import to_jev_response

# A realistic `Router.predict` result covering all three primitives, including
# the `action` blocks and the noul `confidence` that Jev does not have.
LAYA_RESULT = {
    "model": "laya-rl-agent",
    "answers": {
        "dept": {
            "type": "choice",
            "choice": "billing",
            "probabilities": {"billing": 0.9842, "technical": 0.0158},
            "confidence": 0.8827,
            "action": {"act_probability": 0.11},
        },
        "urgency": {
            "type": "score",
            "score": 1.4297,
            "legend": {"0": "not urgent", "1": "soon", "2": "critical"},
            "probabilities": {"0": 0.1109, "1": 0.3486, "2": 0.5406},
            "confidence": 0.141,
            "action": {"act_probability": 0.42},
        },
        "refund": {
            "type": "noul",
            "noul": 0.9492,
            "confidence": 0.9492,
            "action": {"act_probability": 0.37},
        },
    },
    "usage": {"input_tokens": 150, "output_tokens": 0},
    "routing": {"model": "english", "reason": "detection"},
}


def parsed():
    """Parse exactly as the SDK does: model_validate_json over the raw bytes
    (response_types.py:163). JSON object keys are strings on the wire, and that
    path coerces them to the int keys the SDK's models declare."""
    body = to_jev_response(LAYA_RESULT, requested_model="jev-latest")
    return SystemOneResponse.model_validate_json(json.dumps(body))


class TestSdkParsesOurResponse:
    def test_choice_answer_round_trips(self):
        answer = parsed().choices["dept"]
        assert answer.choice == "billing"
        assert answer.probabilities["billing"] == 0.9842

    def test_score_legend_parses_as_an_integer_keyed_map(self):
        """The SDK types legend as dict[int, str]; api.md agrees, while the
        example in primitives.md shows an array. The SDK is authoritative."""
        assert parsed().scores["urgency"].legend == {
            0: "not urgent",
            1: "soon",
            2: "critical",
        }

    def test_noul_answer_round_trips(self):
        assert parsed().nouls["refund"].noul == 0.9492

    def test_usage_round_trips(self):
        assert parsed().usage.input_tokens == 150
