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
