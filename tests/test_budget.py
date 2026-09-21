import pytest

from laya_server.errors import ValidationFailed
from laya_server.validation import validate_budget
from tests.conftest import FakeAgent

NOUL = {"type": "noul", "instructions": "Is a refund requested?"}


class TestFittingState:
    def test_short_state_accepted(self, agent):
        validate_budget(agent, "we were billed twice", {"q": NOUL}, "english")

    def test_state_just_under_budget_accepted(self, agent):
        # 512 max_len, minus a short head; 400 words is comfortably inside.
        validate_budget(agent, " ".join(["word"] * 400), {"q": NOUL}, "english")


class TestOversizedState:
    def test_state_over_budget_rejected(self, agent):
        with pytest.raises(ValidationFailed) as exc:
            validate_budget(agent, " ".join(["word"] * 600), {"q": NOUL}, "english")
        assert "exceeds" in str(exc.value)

    def test_error_reports_counts_and_checkpoint(self, agent):
        with pytest.raises(ValidationFailed) as exc:
            validate_budget(agent, " ".join(["word"] * 600), {"q": NOUL}, "english")
        msg = str(exc.value)
        assert "600" in msg, "should report the actual token count"
        assert "english" in msg, "should name the checkpoint whose budget applies"

    def test_larger_checkpoint_accepts_what_smaller_one_rejects(self):
        state = " ".join(["word"] * 600)
        validate_budget(
            FakeAgent(max_len=1024, head_max_len=256),
            state,
            {"q": NOUL},
            "multilingual",
        )
        with pytest.raises(ValidationFailed):
            validate_budget(FakeAgent(), state, {"q": NOUL}, "english")


class TestPerQuestionBudget:
    def test_long_question_shrinks_room_for_state(self, agent):
        """Room for state is max_len minus that question's head, so a long
        question can push an otherwise-fitting state over the limit.

        The head is itself capped at head_max_len (192), which bounds how far
        room can shrink: 488 tokens for a short noul, 316 for the bulky choice
        below. A 400-token state therefore fits one and not the other.
        """
        state = " ".join(["word"] * 400)
        validate_budget(agent, state, {"q": NOUL}, "english")

        bulky = {
            "type": "choice",
            "instructions": " ".join(["instruction"] * 150),
            "criteria": {f"opt{i}": " ".join(["desc"] * 20) for i in range(8)},
        }
        with pytest.raises(ValidationFailed):
            validate_budget(agent, state, {"q": bulky}, "english")


class TestOptionBudget:
    """Issue 1/2: Laya raises ValueError when a question's options do not fit
    head_max_len (agent.py:263). Jev's 255-option limit is looser, so this must
    be caught here or it reaches the client as a 500."""

    @staticmethod
    def _many_options(n):
        return {
            "pick": {
                "type": "choice",
                "instructions": "Which one?",
                "criteria": {
                    f"option_number_{i}": f"the {i}th choice" for i in range(n)
                },
            }
        }

    def test_options_that_overflow_the_head_are_rejected(self, agent):
        with pytest.raises(ValidationFailed) as exc:
            validate_budget(agent, "", self._many_options(200), "english")
        assert "option" in str(exc.value).lower()

    def test_rejected_even_when_the_state_is_empty(self, agent):
        """The empty state is the case that previously reached Laya and 500'd,
        because the state-size check trivially passes."""
        with pytest.raises(ValidationFailed):
            validate_budget(agent, "", self._many_options(200), "english")

    def test_the_message_blames_the_options_not_the_state(self, agent):
        """Issue 2: reporting 'state exceeds model budget: 1 tokens' sends the
        caller to shrink the wrong thing."""
        with pytest.raises(ValidationFailed) as exc:
            validate_budget(agent, "hi", self._many_options(200), "english")
        msg = str(exc.value).lower()
        assert "option" in msg
        assert "state exceeds" not in msg

    def test_names_the_offending_question(self, agent):
        with pytest.raises(ValidationFailed) as exc:
            validate_budget(agent, "", self._many_options(200), "english")
        assert "pick" in str(exc.value)

    def test_a_reasonable_question_still_passes(self, agent):
        validate_budget(agent, "billed twice", self._many_options(4), "english")
