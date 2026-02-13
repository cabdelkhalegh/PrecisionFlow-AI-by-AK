"""LangGraph Orchestrator — the backbone of the Verify-then-Proceed pipeline.

Uses LangGraph to define a cyclic graph with human-in-the-loop checkpoints
at every step.  No step advances without passing a Quality Gate.

Architecture:
- Each of the 16 steps is a node.
- Edges enforce the linear "Input -> Process -> Verification -> Output" flow.
- Human interrupt points block execution until the founder approves.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, StateGraph

from src.core.quality_gate import approve_step, can_unlock_step
from src.models.venture import StepStatus, VentureState
from src.steps.phase1_validation import (
    step1_problem_definition,
    step2_market_fact_checking,
    step3_competitive_landscape,
    step4_viability_gate,
)
from src.steps.phase2_strategy import (
    step5_customer_persona,
    step6_objection_simulation,
    step7_business_model,
    step8_supply_chain,
)
from src.steps.phase3_execution import (
    step9_tech_stack,
    step10_legal_compliance,
    step11_team_gap_analysis,
    step12_prototype_spec,
)
from src.steps.phase4_scale import (
    step13_marketing_assets,
    step14_roadmap,
    step15_financial_model,
    step16_deal_room,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step metadata (name, description, phase)
# ---------------------------------------------------------------------------

STEP_REGISTRY: dict[int, dict[str, str]] = {
    1:  {"name": "Problem Definition",          "phase": "Validation"},
    2:  {"name": "Market Fact-Checking",         "phase": "Validation"},
    3:  {"name": "Competitive Landscape",        "phase": "Validation"},
    4:  {"name": "Viability Gate",               "phase": "Validation"},
    5:  {"name": "Customer Persona",             "phase": "Strategy"},
    6:  {"name": "Objection Simulation",         "phase": "Strategy"},
    7:  {"name": "Business Model Architect",     "phase": "Strategy"},
    8:  {"name": "Supply Chain",                 "phase": "Strategy"},
    9:  {"name": "Tech Stack Recommendation",    "phase": "Execution"},
    10: {"name": "Legal Compliance",             "phase": "Execution"},
    11: {"name": "Team Gap Analysis",            "phase": "Execution"},
    12: {"name": "Prototype Spec",               "phase": "Execution"},
    13: {"name": "Marketing Asset Generation",   "phase": "Scale"},
    14: {"name": "Roadmap (Gantt)",              "phase": "Scale"},
    15: {"name": "Financial Modeling",           "phase": "Scale"},
    16: {"name": "Deal Room",                    "phase": "Scale"},
}


def get_step_status_map(state: VentureState) -> dict[int, str]:
    """Return the current status of every step for the Control Room UI."""
    result: dict[int, str] = {}
    for step_num in range(1, 17):
        record = state.steps.get(step_num)
        if record:
            result[step_num] = record.status.value
        elif can_unlock_step(state, step_num):
            result[step_num] = "ready"
        else:
            result[step_num] = StepStatus.LOCKED.value
    return result


# ---------------------------------------------------------------------------
# LangGraph pipeline builder
# ---------------------------------------------------------------------------

def _make_step_node(step_num: int, step_fn: Any) -> Any:
    """Wrap a step function into a LangGraph-compatible node.

    Each node:
    1. Executes the step function.
    2. Returns the updated state (the state object IS the graph state).
    """

    async def node(state: dict[str, Any]) -> dict[str, Any]:
        venture = VentureState(**state)
        venture = await step_fn(venture)
        return venture.model_dump()

    node.__name__ = f"step_{step_num}"
    return node


def build_pipeline() -> StateGraph:
    """Construct the 16-step LangGraph pipeline.

    The graph is linear with human interrupt points between every step.
    The founder must approve each step before the next node executes.

    Returns a *compiled* ``StateGraph`` ready to be invoked.
    """
    # Map step numbers to their async handler functions
    step_functions = {
        # Phase 1
        2:  step2_market_fact_checking,
        3:  step3_competitive_landscape,
        4:  step4_viability_gate,
        # Phase 2
        5:  step5_customer_persona,
        6:  step6_objection_simulation,
        7:  step7_business_model,
        8:  step8_supply_chain,
        # Phase 3
        9:  step9_tech_stack,
        10: step10_legal_compliance,
        11: step11_team_gap_analysis,
        12: step12_prototype_spec,
        # Phase 4
        13: step13_marketing_assets,
        14: step14_roadmap,
        15: step15_financial_model,
        16: step16_deal_room,
    }

    graph = StateGraph(dict)

    # Add nodes for steps 2-16 (Step 1 is handled via the API since it
    # requires user_input that isn't part of the graph state).
    for step_num, fn in step_functions.items():
        graph.add_node(f"step_{step_num}", _make_step_node(step_num, fn))

    # Linear edges: step_2 -> step_3 -> ... -> step_16
    step_nums = sorted(step_functions.keys())
    graph.set_entry_point(f"step_{step_nums[0]}")

    for i in range(len(step_nums) - 1):
        graph.add_edge(f"step_{step_nums[i]}", f"step_{step_nums[i + 1]}")

    graph.add_edge(f"step_{step_nums[-1]}", END)

    return graph


class VenturePipeline:
    """High-level interface for running ventures through the pipeline.

    This class manages:
    - Initializing a new venture state.
    - Running individual steps on demand (HITL mode).
    - Tracking approval state.
    """

    def __init__(self) -> None:
        self._graph = build_pipeline()

    async def start_venture(
        self,
        venture_id: str,
        founder_name: str,
        user_input: str,
        founder_skills: list[str] | None = None,
        jurisdiction: str = "Delaware, USA",
        region: str = "United States",
    ) -> VentureState:
        """Initialize a new venture and run Step 1."""
        state = VentureState(
            venture_id=venture_id,
            founder_name=founder_name,
            founder_skills=founder_skills or [],
            jurisdiction=jurisdiction,
            region=region,
        )
        # Step 1 requires user_input, so we call it directly
        state = await step1_problem_definition(state, user_input)
        return state

    async def approve_and_advance(
        self, state: VentureState, step_number: int
    ) -> VentureState:
        """Approve a step and run the next one.

        This is the core HITL interaction:
        1. Founder reviews the output of *step_number*.
        2. Founder clicks "Approve".
        3. The system locks the step and runs the next one.
        """
        state = approve_step(state, step_number)
        next_step = step_number + 1

        if next_step > 16:
            logger.info("Venture %s — all 16 steps complete.", state.venture_id)
            return state

        # Run the next step
        step_functions = {
            2:  step2_market_fact_checking,
            3:  step3_competitive_landscape,
            4:  step4_viability_gate,
            5:  step5_customer_persona,
            6:  step6_objection_simulation,
            7:  step7_business_model,
            8:  step8_supply_chain,
            9:  step9_tech_stack,
            10: step10_legal_compliance,
            11: step11_team_gap_analysis,
            12: step12_prototype_spec,
            13: step13_marketing_assets,
            14: step14_roadmap,
            15: step15_financial_model,
            16: step16_deal_room,
        }

        fn = step_functions.get(next_step)
        if fn:
            state = await fn(state)

        return state

    def get_status(self, state: VentureState) -> dict[str, Any]:
        """Return the full Control Room status for the UI."""
        return {
            "venture_id": state.venture_id,
            "founder": state.founder_name,
            "steps": get_step_status_map(state),
            "registry": STEP_REGISTRY,
            "total_edits": state.total_edits,
            "critic_pass_rate": state.critic_pass_rate,
        }
