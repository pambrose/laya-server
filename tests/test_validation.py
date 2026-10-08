import pytest

from laya_server.errors import ValidationFailed
from laya_server.validation import jev_question, validate_questions


def expect_rejected(questions) -> str:
    with pytest.raises(ValidationFailed) as exc:
        validate_questions(questions)
    return str(exc.value)


class TestLayaCrashModes:
    """Each case crashed laya 0.3.4 with an untyped error (a client-triggerable
    500); all must be 422s raised before inference. laya 0.4 now refuses each
    one itself with a ValueError, at the agent.py lines cited."""

    def test_missing_type_rejected(self):
        # agent.py:984-986
        assert "type" in expect_rejected({"q": {"instructions": "Is it urgent?"}})

    def test_unknown_type_rejected(self):
        # agent.py:984-986
        msg = expect_rejected({"q": {"type": "bool", "instructions": "Urgent?"}})
        assert "bool" in msg

    def test_missing_instructions_rejected(self):
        # agent.py:987-988
        assert "instructions" in expect_rejected({"q": {"type": "noul"}})

    def test_choice_with_null_criteria_rejected(self):
        # agent.py:1001-1003
        msg = expect_rejected(
            {"q": {"type": "choice", "instructions": "Which?", "criteria": None}}
        )
        assert "criteria" in msg

    def test_choice_with_empty_criteria_rejected(self):
        # agent.py:1004-1005
        msg = expect_rejected(
            {"q": {"type": "choice", "instructions": "Which?", "criteria": {}}}
        )
        assert "criteria" in msg

    def test_score_with_empty_criteria_rejected(self):
        # agent.py:1065-1066
        msg = expect_rejected(
            {"q": {"type": "score", "instructions": "How bad?", "criteria": []}}
        )
        assert "criteria" in msg

    def test_noul_with_list_criteria_rejected(self):
        # agent.py:1070-1072
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


class TestLayasOwnRules:
    """Rules laya 0.4 added in `Agent._check_question`. The server runs that
    check during validation, so these are 422s before any tokenizing."""

    def test_blank_instructions_rejected(self):
        msg = expect_rejected({"q": {"type": "noul", "instructions": "   "}})
        assert "'q'" in msg

    def test_noul_criteria_keys_other_than_true_false_rejected(self):
        msg = expect_rejected(
            {
                "q": {
                    "type": "noul",
                    "instructions": "Urgent?",
                    "criteria": {"true": "yes", "maybe": "unsure"},
                }
            }
        )
        assert "maybe" in msg

    def test_duplicate_list_labels_rejected(self):
        expect_rejected(
            {"q": {"type": "choice", "instructions": "Which?", "criteria": ["a", "a"]}}
        )

    def test_unhashable_list_label_rejected_not_500(self):
        """Issue 20: this escaped as a TypeError from the budget check's
        dict comprehension, i.e. a 500."""
        expect_rejected(
            {
                "q": {
                    "type": "choice",
                    "instructions": "Which?",
                    "criteria": [["a"], "b"],
                }
            }
        )


class TestJevDocumentedLimits:
    def test_empty_questions_rejected(self):
        """Jev requires a question; laya 0.4 would answer with nothing."""
        assert "at least one question" in expect_rejected({})

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
        """Laya supports bare labels: {"code": None} (common.py:141)."""
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
        """Coerced to {c: None} at agent.py:1112-1113."""
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

    def test_noul_criteria_keys_are_case_insensitive(self):
        """laya lowercases them (agent.py:1114-1116)."""
        validate_questions(
            {
                "q": {
                    "type": "noul",
                    "instructions": "Phishing?",
                    "criteria": {"TRUE": "it is", "False": "it is not"},
                }
            }
        )


class TestOnlyJevFieldsReachLaya:
    """laya 0.4 honors `labels` (noul) and `option_order`, which change what
    the model is shown. Jev has neither, so they are ignored like any other
    unknown field rather than exposing behavior Jev lacks."""

    def test_laya_only_fields_are_dropped(self):
        question = {
            "type": "noul",
            "instructions": "Urgent?",
            "labels": {"false": "calm", "true": "urgent"},
            "option_order": [1, 0],
        }
        assert jev_question(question) == {"type": "noul", "instructions": "Urgent?"}

    def test_malformed_laya_only_fields_are_not_validated(self):
        """Ignored means ignored: laya would reject these labels, but they
        never reach it."""
        validate_questions(
            {
                "q": {
                    "type": "choice",
                    "instructions": "Which?",
                    "criteria": ["a", "b"],
                    "labels": "nonsense",
                    "option_order": [7],
                }
            }
        )
