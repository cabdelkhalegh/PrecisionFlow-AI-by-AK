"""FastAPI Control Room — the "Flight Check" interface.

Exposes the 16-step pipeline via REST endpoints with full HITL support.
The UI renders a vertical timeline with state indicators:
  - LOCKED:         Cannot access yet.
  - PROCESSING:     AI/Code is working.
  - REVIEW_NEEDED:  Output generated, waiting for Founder approval.
  - VERIFIED:       Approved and locked into the Truth Source.
"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.core.config import settings
from src.core.orchestrator import STEP_REGISTRY, VenturePipeline
from src.core.quality_gate import record_user_edit
from src.models.venture import VentureState

app = FastAPI(
    title="PrecisionFlow — The Entrepreneur's Toolbox",
    description="A Verify-then-Proceed Venture Operating System. The user is the Pilot; the AI is the Engine.",
    version="5.0.0",
)

# In-memory venture store (swap for Supabase in production)
_ventures: dict[str, VentureState] = {}
_pipeline = VenturePipeline()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class StartVentureRequest(BaseModel):
    founder_name: str
    user_input: str = Field(description="The founder's idea in their own words.")
    founder_skills: list[str] = Field(default_factory=list)
    jurisdiction: str = "Delaware, USA"
    region: str = "United States"


class ApproveStepRequest(BaseModel):
    step_number: int = Field(ge=1, le=16)


class EditStepRequest(BaseModel):
    step_number: int = Field(ge=1, le=16)
    new_output: dict[str, Any]


class StarAssetsRequest(BaseModel):
    asset_indices: list[int] = Field(
        description="Indices of the marketing assets to star (Step 13)."
    )


class SelectModelRequest(BaseModel):
    model_type: str = Field(description="The chosen BusinessModelType value.")


class OverrideViabilityRequest(BaseModel):
    """User explicitly overrides a NO-GO verdict."""

    confirm: bool = True


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok", "version": "5.0.0"}


@app.post("/ventures", status_code=201)
async def start_venture(req: StartVentureRequest) -> dict[str, Any]:
    """Create a new venture and run Step 1 (Problem Definition)."""
    venture_id = str(uuid.uuid4())
    state = await _pipeline.start_venture(
        venture_id=venture_id,
        founder_name=req.founder_name,
        user_input=req.user_input,
        founder_skills=req.founder_skills,
        jurisdiction=req.jurisdiction,
        region=req.region,
    )
    _ventures[venture_id] = state
    return {
        "venture_id": venture_id,
        "status": _pipeline.get_status(state),
        "step_1_output": state.charter.model_dump() if state.charter else None,
    }


@app.get("/ventures/{venture_id}")
async def get_venture(venture_id: str) -> dict[str, Any]:
    """Get the full status of a venture."""
    state = _ventures.get(venture_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Venture not found.")
    return _pipeline.get_status(state)


@app.get("/ventures/{venture_id}/steps/{step_number}")
async def get_step_output(venture_id: str, step_number: int) -> dict[str, Any]:
    """Get the output of a specific step."""
    state = _ventures.get(venture_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Venture not found.")

    record = state.steps.get(step_number)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Step {step_number} not started.")

    output = record.output
    if hasattr(output, "model_dump"):
        output = output.model_dump()

    return {
        "step_number": step_number,
        "name": record.name,
        "status": record.status.value,
        "output": output,
        "critic_score": record.critic_score,
        "user_edits": record.user_edits,
        "approved": record.approved,
    }


@app.post("/ventures/{venture_id}/approve")
async def approve_step(venture_id: str, req: ApproveStepRequest) -> dict[str, Any]:
    """Founder approves a step, which locks it and triggers the next step."""
    state = _ventures.get(venture_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Venture not found.")

    state = await _pipeline.approve_and_advance(state, req.step_number)
    _ventures[venture_id] = state

    next_step = req.step_number + 1
    next_output = None
    if next_step <= 16:
        record = state.steps.get(next_step)
        if record and hasattr(record.output, "model_dump"):
            next_output = record.output.model_dump()
        elif record:
            next_output = record.output

    return {
        "approved_step": req.step_number,
        "status": _pipeline.get_status(state),
        "next_step_output": next_output,
    }


@app.post("/ventures/{venture_id}/edit")
async def edit_step(venture_id: str, req: EditStepRequest) -> dict[str, Any]:
    """User overwrites an AI output. The system accepts the edit as the new Truth."""
    state = _ventures.get(venture_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Venture not found.")

    state = record_user_edit(state, req.step_number, req.new_output)
    _ventures[venture_id] = state

    return {
        "step_number": req.step_number,
        "user_edits": state.steps[req.step_number].user_edits,
        "total_edits": state.total_edits,
    }


@app.post("/ventures/{venture_id}/override-viability")
async def override_viability(
    venture_id: str, req: OverrideViabilityRequest
) -> dict[str, Any]:
    """User explicitly overrides a NO-GO verdict from Step 4."""
    state = _ventures.get(venture_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Venture not found.")

    if state.viability is None:
        raise HTTPException(status_code=400, detail="Step 4 has not been completed.")

    if req.confirm:
        state.viability.user_override = True
    _ventures[venture_id] = state

    return {"viability_override": state.viability.user_override}


@app.post("/ventures/{venture_id}/star-assets")
async def star_assets(venture_id: str, req: StarAssetsRequest) -> dict[str, Any]:
    """User stars their preferred marketing assets (Step 13).

    The user must star the best 3 to train the model on preference.
    """
    state = _ventures.get(venture_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Venture not found.")

    if state.campaign_assets is None:
        raise HTTPException(status_code=400, detail="Step 13 has not been completed.")

    for idx in req.asset_indices:
        if 0 <= idx < len(state.campaign_assets.assets):
            state.campaign_assets.assets[idx].starred = True

    state.campaign_assets.starred_count = sum(
        1 for a in state.campaign_assets.assets if a.starred
    )
    _ventures[venture_id] = state

    return {"starred_count": state.campaign_assets.starred_count}


@app.post("/ventures/{venture_id}/select-model")
async def select_business_model(
    venture_id: str, req: SelectModelRequest
) -> dict[str, Any]:
    """User selects their preferred monetization model (Step 7)."""
    from src.models.venture import BusinessModelType

    state = _ventures.get(venture_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Venture not found.")

    if state.monetization is None:
        raise HTTPException(status_code=400, detail="Step 7 has not been completed.")

    try:
        model_type = BusinessModelType(req.model_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model type. Valid: {[t.value for t in BusinessModelType]}",
        )

    state.monetization.selected = model_type
    _ventures[venture_id] = state

    return {"selected_model": model_type.value}


@app.get("/ventures/{venture_id}/metrics")
async def get_metrics(venture_id: str) -> dict[str, Any]:
    """Success metrics dashboard.

    - Edit Rate: >20 % means the Golden Database needs improvement.
    - Critic Pass Rate: How often the Critic approves the first draft.
    """
    state = _ventures.get(venture_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Venture not found.")

    total_steps_completed = sum(
        1 for r in state.steps.values() if r.approved
    )
    edit_rate = state.total_edits / max(total_steps_completed, 1)

    critic_scores = [r.critic_score for r in state.steps.values() if r.critic_score is not None]
    avg_critic = sum(critic_scores) / len(critic_scores) if critic_scores else 0.0

    return {
        "venture_id": state.venture_id,
        "steps_completed": total_steps_completed,
        "total_edits": state.total_edits,
        "edit_rate": round(edit_rate, 3),
        "edit_rate_warning": edit_rate > 0.20,
        "average_critic_score": round(avg_critic, 3),
        "critic_scores": {
            step_num: r.critic_score
            for step_num, r in state.steps.items()
            if r.critic_score is not None
        },
    }
