"""Tests for the Quality Gate — the verification checkpoint system."""

import pytest

from src.core.quality_gate import (
    QualityGateError,
    approve_step,
    can_unlock_step,
    record_user_edit,
    submit_for_review,
    unlock_step,
)
from src.models.venture import StepStatus, VentureState


def _make_state() -> VentureState:
    return VentureState(venture_id="test-001", founder_name="Test Founder")


class TestCanUnlockStep:
    def test_step_1_always_unlockable(self) -> None:
        state = _make_state()
        assert can_unlock_step(state, 1) is True

    def test_step_2_requires_step_1_verified(self) -> None:
        state = _make_state()
        assert can_unlock_step(state, 2) is False

        # Simulate Step 1 verified
        state = unlock_step(state, 1, "Problem Definition")
        state = submit_for_review(state, 1, {"test": True})
        state = approve_step(state, 1)
        assert can_unlock_step(state, 2) is True

    def test_step_3_requires_step_2(self) -> None:
        state = _make_state()
        assert can_unlock_step(state, 3) is False


class TestUnlockStep:
    def test_unlock_step_1(self) -> None:
        state = _make_state()
        state = unlock_step(state, 1, "Problem Definition")
        assert state.steps[1].status == StepStatus.PROCESSING

    def test_unlock_step_2_without_step_1_raises(self) -> None:
        state = _make_state()
        with pytest.raises(QualityGateError):
            unlock_step(state, 2, "Market Fact-Checking")


class TestSubmitForReview:
    def test_submit_changes_status(self) -> None:
        state = _make_state()
        state = unlock_step(state, 1, "Problem Definition")
        state = submit_for_review(state, 1, {"charter": "test"})
        assert state.steps[1].status == StepStatus.REVIEW_NEEDED

    def test_submit_without_unlock_raises(self) -> None:
        state = _make_state()
        with pytest.raises(QualityGateError):
            submit_for_review(state, 1, {"data": "test"})


class TestApproveStep:
    def test_approve_locks_step(self) -> None:
        state = _make_state()
        state = unlock_step(state, 1, "Problem Definition")
        state = submit_for_review(state, 1, {"charter": "test"})
        state = approve_step(state, 1)
        assert state.steps[1].status == StepStatus.VERIFIED
        assert state.steps[1].approved is True
        assert state.steps[1].locked_at is not None

    def test_approve_non_review_step_raises(self) -> None:
        state = _make_state()
        state = unlock_step(state, 1, "Problem Definition")
        with pytest.raises(QualityGateError):
            approve_step(state, 1)


class TestRecordUserEdit:
    def test_edit_increments_counters(self) -> None:
        state = _make_state()
        state = unlock_step(state, 1, "Problem Definition")
        state = submit_for_review(state, 1, {"original": True})

        state = record_user_edit(state, 1, {"edited": True})
        assert state.steps[1].output == {"edited": True}
        assert state.steps[1].user_edits == 1
        assert state.total_edits == 1

    def test_multiple_edits(self) -> None:
        state = _make_state()
        state = unlock_step(state, 1, "Problem Definition")
        state = submit_for_review(state, 1, {"v1": True})
        state = record_user_edit(state, 1, {"v2": True})
        state = record_user_edit(state, 1, {"v3": True})
        assert state.steps[1].user_edits == 2
        assert state.total_edits == 2
