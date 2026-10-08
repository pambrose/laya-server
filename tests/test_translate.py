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


class TestLaya04Additions:
    """Keys laya 0.4 added to its output (agent.py:1357-1405, 1541-1566)."""

    def test_answer_confidence_is_stripped(self):
        laya = {
            "type": "choice",
            "choice": "billing",
            "probabilities": {"billing": 0.94, "sales": 0.06},
            "confidence": 0.88,
            "answer_confidence": 0.94,
            "action": {"act_probability": 0.12},
        }
        assert "answer_confidence" not in to_jev_answer(laya)

    def test_usage_keeps_only_jevs_fields(self):
        laya = {
            "model": "laya-rl-agent",
            "answers": {},
            "usage": {
                "input_tokens": 120,
                "output_tokens": 0,
                "state_tokens": 40,
                "state_tokens_dropped": 0,
                "truncated": False,
                "truncated_questions": [],
                "options": {"q": {"collided": 1}},
            },
        }
        assert to_jev_response(laya, requested_model="jev-latest")["usage"] == {
            "input_tokens": 120,
            "output_tokens": 0,
        }


class TestScoreLegend:
    """laya 0.4 renders every score level to text (agent.py:1384-1390)."""

    @staticmethod
    def _translate(criteria, laya_legend):
        laya = {
            "model": "laya-rl-agent",
            "answers": {
                "u": {
                    "type": "score",
                    "score": 0.5,
                    "legend": laya_legend,
                    "probabilities": {k: 0.5 for k in laya_legend},
                    "confidence": 0.1,
                }
            },
            "usage": {"input_tokens": 1, "output_tokens": 0},
        }
        questions = {"u": {"type": "score", "instructions": "?", "criteria": criteria}}
        body = to_jev_response(laya, requested_model="jev-latest", questions=questions)
        return body["answers"]["u"]["legend"]

    def test_object_and_array_levels_are_echoed_as_written(self):
        """Jev's legend admits objects and arrays; laya would send JSON text."""
        criteria = [{"desc": "low"}, ["mid", "ish"]]
        legend = self._translate(
            criteria, {"0": '{"desc": "low"}', "1": '["mid", "ish"]'}
        )
        assert legend == {"0": {"desc": "low"}, "1": ["mid", "ish"]}

    def test_text_levels_are_unchanged(self):
        assert self._translate(["low", "high"], {"0": "low", "1": "high"}) == {
            "0": "low",
            "1": "high",
        }

    def test_numeric_levels_keep_layas_text(self):
        """The SDK is strict and a legend value must be str, dict or list."""
        assert self._translate([1, 2], {"0": "1", "1": "2"}) == {"0": "1", "1": "2"}
