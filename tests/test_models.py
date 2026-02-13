"""Tests for the Pydantic data models."""

from src.models.venture import (
    BusinessModelType,
    StepRecord,
    StepStatus,
    VentureCharter,
    VentureState,
    ViabilityDashboard,
    ViabilityVerdict,
)


class TestVentureState:
    def test_default_state(self) -> None:
        state = VentureState()
        assert state.venture_id == ""
        assert state.steps == {}
        assert state.total_edits == 0

    def test_state_with_charter(self) -> None:
        charter = VentureCharter(
            problem_statement="Founders waste time on bad ideas",
            target_audience="First-time entrepreneurs",
            proposed_solution="AI-powered validation tool",
            value_proposition="Verify before building",
        )
        state = VentureState(venture_id="v-001", charter=charter)
        assert state.charter is not None
        assert state.charter.problem_statement == "Founders waste time on bad ideas"


class TestStepRecord:
    def test_default_locked(self) -> None:
        record = StepRecord(step_number=1, name="Test")
        assert record.status == StepStatus.LOCKED
        assert record.approved is False
        assert record.critic_score is None

    def test_serialization(self) -> None:
        record = StepRecord(
            step_number=3,
            name="Competitive Landscape",
            status=StepStatus.VERIFIED,
            critic_score=0.92,
            approved=True,
        )
        data = record.model_dump()
        assert data["step_number"] == 3
        assert data["status"] == "verified"


class TestViabilityDashboard:
    def test_go_verdict(self) -> None:
        dash = ViabilityDashboard(
            market_volume_score=0.8,
            competitor_density_score=0.3,
            weighted_score=0.65,
            verdict=ViabilityVerdict.GO,
        )
        assert dash.verdict == ViabilityVerdict.GO

    def test_no_go_with_override(self) -> None:
        dash = ViabilityDashboard(
            verdict=ViabilityVerdict.NO_GO,
            user_override=True,
        )
        assert dash.verdict == ViabilityVerdict.NO_GO
        assert dash.user_override is True


class TestBusinessModelType:
    def test_enum_values(self) -> None:
        assert BusinessModelType.SAAS_TIERED.value == "saas_tiered"
        assert BusinessModelType.MARKETPLACE_TRANSACTION_FEE.value == "marketplace_transaction_fee"
