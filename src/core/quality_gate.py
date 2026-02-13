"""Quality Gate — the verification checkpoint between every pipeline step.

No step advances without passing this gate.  The user is the Pilot;
the AI is the Engine.
"""

from __future__ import annotations

from src.models.venture import StepRecord, StepStatus, VentureState


class QualityGateError(Exception):
    """Raised when a step fails the quality gate and cannot proceed."""


def is_step_approved(state: VentureState, step_number: int) -> bool:
    """Return True if the given step has been approved by the user."""
    record = state.steps.get(step_number)
    return record is not None and record.approved


def can_unlock_step(state: VentureState, step_number: int) -> bool:
    """Determine whether *step_number* is eligible to run.

    Rules:
    - Step 1 is always unlockable.
    - Every subsequent step requires the previous step to be ``VERIFIED``.
    - Step 5 additionally requires Step 4's verdict to be GO or OVERRIDE.
    """
    if step_number == 1:
        return True

    prev = state.steps.get(step_number - 1)
    if prev is None or prev.status != StepStatus.VERIFIED:
        return False

    # Special gate: Step 5 requires viability GO/OVERRIDE
    if step_number == 5 and state.viability is not None:
        from src.models.venture import ViabilityVerdict

        if state.viability.verdict == ViabilityVerdict.NO_GO and not state.viability.user_override:
            return False

    return True


def unlock_step(state: VentureState, step_number: int, name: str) -> VentureState:
    """Transition a step from LOCKED to PROCESSING.

    Raises ``QualityGateError`` if prerequisites are not met.
    """
    if not can_unlock_step(state, step_number):
        raise QualityGateError(
            f"Step {step_number} cannot be unlocked — previous step not verified."
        )

    state.steps[step_number] = StepRecord(
        step_number=step_number,
        name=name,
        status=StepStatus.PROCESSING,
    )
    return state


def submit_for_review(state: VentureState, step_number: int, output: object) -> VentureState:
    """Mark a step's output as ready for founder review."""
    record = state.steps.get(step_number)
    if record is None:
        raise QualityGateError(f"Step {step_number} has no record — was it unlocked first?")

    record.output = output
    record.status = StepStatus.REVIEW_NEEDED
    return state


def approve_step(state: VentureState, step_number: int) -> VentureState:
    """Founder approves the step, locking it into the Truth Source."""
    from datetime import datetime

    record = state.steps.get(step_number)
    if record is None or record.status != StepStatus.REVIEW_NEEDED:
        raise QualityGateError(
            f"Step {step_number} is not in REVIEW_NEEDED state — cannot approve."
        )

    record.approved = True
    record.status = StepStatus.VERIFIED
    record.locked_at = datetime.utcnow()
    return state


def record_user_edit(state: VentureState, step_number: int, new_output: object) -> VentureState:
    """User overwrites the AI output — the system accepts the edit as the new Truth."""
    record = state.steps.get(step_number)
    if record is None:
        raise QualityGateError(f"Step {step_number} has no record.")

    record.output = new_output
    record.user_edits += 1
    state.total_edits += 1
    return state
