"""Phase 4: Scale (The Math) — Steps 13-16.

Step 13: Marketing Asset Generation (objection-based)
Step 14: Roadmap / Gantt (historical data)
Step 15: Financial Modeling (100 % code)
Step 16: Deal Room (compilation)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI

from src.core.config import settings
from src.core.critic import critic_loop
from src.core.quality_gate import submit_for_review, unlock_step
from src.financial.engine import build_financial_model
from src.models.venture import (
    CampaignAssetFolder,
    DealRoom,
    GanttTask,
    MarketingAsset,
    RoadmapGantt,
    VentureState,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Historical development data for timeline estimation (Step 14)
# ---------------------------------------------------------------------------

HISTORICAL_DURATIONS: dict[str, float] = {
    "Authentication System": 2.0,
    "User Dashboard": 1.5,
    "Payment Integration": 2.0,
    "Landing Page": 1.0,
    "API Development": 3.0,
    "Database Schema": 1.0,
    "Admin Panel": 2.0,
    "Email System": 1.0,
    "Search Functionality": 1.5,
    "Analytics Integration": 1.0,
    "CI/CD Pipeline": 1.0,
    "Testing Suite": 2.0,
    "Mobile Responsive": 1.0,
    "SEO Optimization": 0.5,
    "Documentation": 1.0,
}


# ---------------------------------------------------------------------------
# Step 13 — Marketing Asset Generation
# ---------------------------------------------------------------------------

async def _generate_marketing(
    risk_report: dict[str, Any],
    persona: dict[str, Any],
    charter: dict[str, Any],
    critic_feedback: list[str] | None = None,
) -> CampaignAssetFolder:
    """Generator Bot for Step 13: create ad copy that addresses known objections."""
    feedback_text = ""
    if critic_feedback:
        feedback_text = (
            "\n\nPREVIOUS DRAFT FAILED. Fix:\n"
            + "\n".join(f"- {f}" for f in critic_feedback)
        )

    prompt = f"""Create marketing assets that DIRECTLY address the objections found
in the risk report.  Each asset must target a specific marketing channel.

RISK REPORT (objections to address):
{json.dumps(risk_report, indent=2)}

CUSTOMER PERSONA:
{json.dumps(persona, indent=2)}

VENTURE:
{json.dumps(charter, indent=2)}
{feedback_text}

Return JSON:
{{
  "assets": [
    {{
      "asset_type": "<ad/email/social/landing_page>",
      "headline": "<headline>",
      "body": "<body copy>",
      "channel": "<specific channel: Google Ads, Facebook, LinkedIn, Email, etc.>"
    }},
    ...
  ]
}}
Generate at least 6 variations.  Only output valid JSON."""

    llm = ChatGoogleGenerativeAI(
        model=settings.llm_model,
        temperature=0.3,  # Slightly creative for marketing
        google_api_key=settings.gemini_api_key,
    )
    response = await llm.ainvoke(prompt)
    content = response.content if hasattr(response, "content") else str(response)
    data = json.loads(content)

    assets = [MarketingAsset(**a) for a in data.get("assets", [])]
    return CampaignAssetFolder(assets=assets, starred_count=0)


async def step13_marketing_assets(state: VentureState) -> VentureState:
    """Step 13: Generate campaign assets based on Step 6 objections.

    User must 'Star' the best 3 to train the model on preference.
    """
    state = unlock_step(state, 13, "Marketing Asset Generation")

    risk_dict = state.risk_report.model_dump() if state.risk_report else {}
    persona_dict = state.persona.model_dump() if state.persona else {}
    charter_dict = state.charter.model_dump() if state.charter else {}

    assets, critic_result = await critic_loop(
        step_number=13,
        generator_fn=_generate_marketing,
        risk_report=risk_dict,
        persona=persona_dict,
        charter=charter_dict,
    )

    state.campaign_assets = assets
    state.steps[13].critic_score = critic_result.score
    state = submit_for_review(state, 13, assets)
    return state


# ---------------------------------------------------------------------------
# Step 14 — Roadmap / Gantt (historical data, not AI)
# ---------------------------------------------------------------------------

async def step14_roadmap(state: VentureState) -> VentureState:
    """Step 14: Build a realistic timeline from historical development data."""
    state = unlock_step(state, 14, "Roadmap (Gantt)")

    tasks: list[GanttTask] = []
    week_cursor = 0

    # Core tasks every project needs
    core_tasks = [
        "Database Schema",
        "API Development",
        "Authentication System",
        "User Dashboard",
        "Landing Page",
    ]

    # Add payment if monetization is selected
    if state.monetization and state.monetization.selected:
        core_tasks.append("Payment Integration")

    # Add remaining standard tasks
    core_tasks.extend(["Testing Suite", "CI/CD Pipeline", "Documentation"])

    for task_name in core_tasks:
        duration = HISTORICAL_DURATIONS.get(task_name, 1.5)
        tasks.append(
            GanttTask(
                name=task_name,
                duration_weeks=duration,
                start_week=week_cursor,
                dependencies=[tasks[-1].name] if tasks else [],
            )
        )
        week_cursor += int(duration)

    roadmap = RoadmapGantt(tasks=tasks, total_weeks=week_cursor)
    state.roadmap = roadmap
    state = submit_for_review(state, 14, roadmap)
    return state


# ---------------------------------------------------------------------------
# Step 15 — Financial Modeling (100 % Code, Zero AI)
# ---------------------------------------------------------------------------

async def step15_financial_model(state: VentureState) -> VentureState:
    """Step 15: Generate a mathematically perfect P&L.

    100 % code.  No AI.  The AI only inputs the assumptions.
    """
    state = unlock_step(state, 15, "Financial Modeling")

    # Derive assumptions from previous steps
    price = 49.0  # Default
    cost = 10.0
    cac = 30.0

    if state.monetization and state.monetization.options:
        selected = None
        for opt in state.monetization.options:
            if state.monetization.selected and opt.model_type == state.monetization.selected:
                selected = opt
                break
        if selected is None:
            selected = state.monetization.options[0]
        price = selected.estimated_price or 49.0
        cost = selected.estimated_cost or 10.0

    financials = build_financial_model(
        price_per_unit=price,
        cost_per_unit=cost,
        customer_acquisition_cost=cac,
        initial_customers=10,
        monthly_growth_rate=0.15,
        fixed_monthly_costs=5000.0,
        months=24,
        churn_rate=0.05,
    )

    state.financials = financials
    state.steps[15].critic_score = 1.0  # Code output is always correct
    state = submit_for_review(state, 15, financials)
    return state


# ---------------------------------------------------------------------------
# Step 16 — Deal Room
# ---------------------------------------------------------------------------

async def step16_deal_room(state: VentureState) -> VentureState:
    """Step 16: Compile all verified data into an investor pitch deck."""
    state = unlock_step(state, 16, "Deal Room")

    sections: list[dict[str, str]] = [
        {
            "title": "Problem",
            "content": state.charter.problem_statement if state.charter else "",
        },
        {
            "title": "Solution",
            "content": state.charter.proposed_solution if state.charter else "",
        },
        {
            "title": "Market Opportunity",
            "content": json.dumps(
                state.market_data.search_volumes if state.market_data else {}, indent=2
            ),
        },
        {
            "title": "Competition",
            "content": (
                f"{len(state.competitor_matrix.competitors)} competitors identified"
                if state.competitor_matrix
                else "N/A"
            ),
        },
        {
            "title": "Business Model",
            "content": (
                state.monetization.selected.value
                if state.monetization and state.monetization.selected
                else "TBD"
            ),
        },
        {
            "title": "Target Customer",
            "content": (
                f"{state.persona.name} — {state.persona.occupation}"
                if state.persona
                else "TBD"
            ),
        },
        {
            "title": "Technology",
            "content": (
                f"{state.tech_stack.app_type}: {', '.join(state.tech_stack.backend)}"
                if state.tech_stack
                else "TBD"
            ),
        },
        {
            "title": "Financial Projections",
            "content": (
                f"Year 1 Revenue: ${state.financials.annual_revenue:,.0f} | "
                f"Profit: ${state.financials.annual_profit:,.0f}"
                if state.financials
                else "TBD"
            ),
        },
        {
            "title": "Roadmap",
            "content": (
                f"{state.roadmap.total_weeks} weeks to MVP"
                if state.roadmap
                else "TBD"
            ),
        },
        {
            "title": "Team",
            "content": (
                f"{len(state.team_gaps.requisitions)} hires needed"
                if state.team_gaps
                else "TBD"
            ),
        },
    ]

    deal_room = DealRoom(
        pitch_deck_sections=sections,
        compiled_at=datetime.utcnow(),
    )

    state.deal_room = deal_room
    state = submit_for_review(state, 16, deal_room)
    return state
