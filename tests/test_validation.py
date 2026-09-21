import pytest

from laya_server.errors import ValidationFailed
from laya_server.validation import validate_questions


def expect_rejected(questions) -> str:
    with pytest.raises(ValidationFailed) as exc:
        validate_questions(questions)
    return str(exc.value)


class TestLayaCrashModes:
    """Each case crashes raw Laya with an untyped error; all must become 422s.

    References are to the installed laya 0.3.4 source.
    """

    def test_empty_questions_rejected(self):
        # collate_items returns None -> TypeError at agent.py:271
        assert "at least one question" in expect_rejected({})

    def test_missing_type_rejected(self):
        # KeyError: 'type' at agent.py:231
        assert "type" in expect_rejected({"q": {"instructions": "Is it urgent?"}})

    def test_unknown_type_rejected(self):
        # falls through render_options, dies at KeyError: 'bool' agent.py:264
        msg = expect_rejected({"q": {"type": "bool", "instructions": "Urgent?"}})
        assert "bool" in msg

    def test_missing_instructions_rejected(self):
        # KeyError: 'instructions' at agent.py:235
        assert "instructions" in expect_rejected({"q": {"type": "noul"}})

    def test_choice_with_null_criteria_rejected(self):
        # AttributeError: 'NoneType' has no attribute 'items' at common.py:38
        msg = expect_rejected(
            {"q": {"type": "choice", "instructions": "Which?", "criteria": None}}
        )
        assert "criteria" in msg

    def test_choice_with_empty_criteria_rejected(self):
        # k=0 reaches numpy -> ValueError: zero-size array at agent.py:307
        msg = expect_rejected(
            {"q": {"type": "choice", "instructions": "Which?", "criteria": {}}}
        )
        assert "criteria" in msg

    def test_score_with_empty_criteria_rejected(self):
        msg = expect_rejected(
            {"q": {"type": "score", "instructions": "How bad?", "criteria": []}}
        )
        assert "criteria" in msg

    def test_noul_with_list_criteria_rejected(self):
        # AttributeError at common.py:42 (crit.get on a list)
        msg = expect_rejected(
            {
                "q": {
                    "type": "noul",
                    "instructions": "Urgent?",
                    "criteria": ["yes", "no"],
                }
            }
        )
        assert "criteria" in msg


class TestJevDocumentedLimits:
    def test_choice_over_255_options_rejected(self):
        criteria = {f"opt{i}": f"option {i}" for i in range(256)}
        msg = expect_rejected(
            {"q": {"type": "choice", "instructions": "Which?", "criteria": criteria}}
        )
        assert "255" in msg

    def test_score_with_one_level_rejected(self):
        msg = expect_rejected(
            {"q": {"type": "score", "instructions": "How bad?", "criteria": ["only"]}}
        )
        assert "2" in msg and "10" in msg

    def test_score_with_eleven_levels_rejected(self):
        criteria = [f"level {i}" for i in range(11)]
        msg = expect_rejected(
            {"q": {"type": "score", "instructions": "How bad?", "criteria": criteria}}
        )
        assert "2" in msg and "10" in msg

    def test_error_names_the_offending_question_id(self):
        msg = expect_rejected({"churn_risk": {"type": "noul"}})
        assert "churn_risk" in msg


class TestValidRequests:
    def test_noul_without_criteria_is_valid(self):
        validate_questions({"q": {"type": "noul", "instructions": "Urgent?"}})

    def test_noul_with_true_false_criteria_is_valid(self):
        validate_questions(
            {
                "q": {
                    "type": "noul",
                    "instructions": "Phishing?",
                    "criteria": {"true": "it is", "false": "it is not"},
                }
            }
        )

    def test_choice_with_null_valued_options_is_valid(self):
        """Laya supports bare labels: {"code": None} (common.py:38)."""
        validate_questions(
            {
                "q": {
                    "type": "choice",
                    "instructions": "Domain?",
                    "criteria": {"code": None, "math": None},
                }
            }
        )

    def test_choice_with_list_criteria_is_valid(self):
        """Coerced to {c: None} at agent.py:233-234."""
        validate_questions(
            {"q": {"type": "choice", "instructions": "Which?", "criteria": ["a", "b"]}}
        )

    def test_structured_instructions_are_valid(self):
        validate_questions(
            {
                "q": {
                    "type": "noul",
                    "instructions": {"ask": "Urgent?", "note": "consider deadlines"},
                }
            }
        )
