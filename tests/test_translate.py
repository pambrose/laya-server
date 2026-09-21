from laya_server.translate import to_jev_answer, to_jev_response


class TestChoiceAnswer:
    def test_strips_action_key(self):
        laya = {
            "type": "choice",
            "choice": "billing",
            "probabilities": {"billing": 0.94, "sales": 0.06},
            "confidence": 0.88,
            "action": {"act_probability": 0.12},
        }
        assert to_jev_answer(laya) == {
            "type": "choice",
            "choice": "billing",
            "probabilities": {"billing": 0.94, "sales": 0.06},
            "confidence": 0.88,
        }


class TestScoreAnswer:
    def test_strips_action_and_keeps_legend(self):
        laya = {
            "type": "score",
            "score": 1.5,
            "legend": {"0": "low", "1": "mid", "2": "high"},
            "probabilities": {"0": 0.2, "1": 0.6, "2": 0.2},
            "confidence": 0.71,
            "action": {"act_probability": 0.4},
        }
        assert to_jev_answer(laya) == {
            "type": "score",
            "score": 1.5,
            "legend": {"0": "low", "1": "mid", "2": "high"},
            "probabilities": {"0": 0.2, "1": 0.6, "2": 0.2},
            "confidence": 0.71,
        }


class TestNoulAnswer:
    def test_strips_action_and_confidence(self):
        """Jev documents that noul has no separate confidence."""
        laya = {
            "type": "noul",
            "noul": 0.85,
            "confidence": 0.85,
            "action": {"act_probability": 0.3},
        }
        assert to_jev_answer(laya) == {"type": "noul", "noul": 0.85}


class TestResponse:
    def test_echoes_requested_model_not_laya_internal_name(self):
        laya = {
            "model": "laya-rl-agent",
            "answers": {"refund": {"type": "noul", "noul": 0.9, "confidence": 0.9}},
            "usage": {"input_tokens": 120, "output_tokens": 0},
            "routing": {"model": "english", "reason": "detection"},
        }
        assert to_jev_response(laya, requested_model="jev-latest") == {
            "model": "jev-latest",
            "answers": {"refund": {"type": "noul", "noul": 0.9}},
            "usage": {"input_tokens": 120, "output_tokens": 0},
        }

    def test_drops_routing_key_added_by_router(self):
        laya = {
            "model": "laya-rl-agent",
            "answers": {},
            "usage": {"input_tokens": 1, "output_tokens": 0},
            "routing": {"model": "multilingual"},
        }
        assert "routing" not in to_jev_response(laya, requested_model="jev-latest")
