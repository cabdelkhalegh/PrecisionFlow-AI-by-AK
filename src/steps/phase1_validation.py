"""Phase 1: Validation (The Truth) — Steps 1-4.

Step 1: Problem Definition
Step 2: Market Fact-Checking (APIs, not AI)
Step 3: Competitive Landscape
Step 4: Viability Gate (code, not AI)
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI

from src.core.config import settings
from src.core.critic import critic_loop
from src.core.quality_gate import submit_for_review, unlock_step
from src.database.vector_store import GoldenDatabase
from src.financial.engine import calculate_viability_score
from src.integrations.competitors import CompetitorAnalyzer
from src.integrations.market_data import MarketDataClient
from src.models.venture import (
    CompetitorMatrix,
    MarketDataReport,
    VentureCharter,
    VentureState,
    ViabilityDashboard,
    ViabilityVerdict,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1 — Problem Definition
# ---------------------------------------------------------------------------

async def _generate_charter(
    user_input: str,
    case_studies: list[dict[str, Any]],
    critic_feedback: list[str] | None = None,
) -> VentureCharter:
    """Generator Bot for Step 1: draft a Venture Charter."""
    studies_text = "\n".join(
        f"- {cs.get('company_name', 'N/A')}: {cs.get('summary', '')}"
        for cs in case_studies
    )
    feedback_text = ""
    if critic_feedback:
        feedback_text = (
            "\n\nPREVIOUS DRAFT FAILED QUALITY CHECK. Address this feedback:\n"
            + "\n".join(f"- {f}" for f in critic_feedback)
        )

    prompt = f"""You are a venture strategist.  Based on the founder's input and these
proven case studies from our database, create a structured Venture Charter.

FOUNDER INPUT:
{user_input}

RELEVANT CASE STUDIES:
{studies_text}
{feedback_text}

Return JSON:
{{
  "problem_statement": "<specific pain point>",
  "target_audience": "<who experiences this>",
  "proposed_solution": "<the proposed solution>",
  "value_proposition": "<what makes this different>",
  "matched_case_studies": ["<study 1 name>", ...],
  "database_match_found": true/false
}}
Only output valid JSON."""

    llm = ChatGoogleGenerativeAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        google_api_key=settings.gemini_api_key,
    )
    response = await llm.ainvoke(prompt)
    content = response.content if hasattr(response, "content") else str(response)

    import json

    data = json.loads(content)
    return VentureCharter(**data)


async def step1_problem_definition(state: VentureState, user_input: str) -> VentureState:
    """Step 1: Map user input against the Golden Database and draft a charter."""
    state = unlock_step(state, 1, "Problem Definition")

    # RAG retrieval from the Golden Database
    db = GoldenDatabase()
    case_studies = await db.search_by_text(user_input, top_k=3)
    cs_dicts = [cs.model_dump() for cs in case_studies]

    # Generator -> Critic loop
    charter, critic_result = await critic_loop(
        step_number=1,
        generator_fn=_generate_charter,
        user_input=user_input,
        case_studies=cs_dicts,
    )

    state.charter = charter
    state.steps[1].critic_score = critic_result.score
    state = submit_for_review(state, 1, charter)
    return state


# ---------------------------------------------------------------------------
# Step 2 — Market Fact-Checking (APIs, not AI)
# ---------------------------------------------------------------------------

async def step2_market_fact_checking(state: VentureState) -> VentureState:
    """Step 2: Fetch real search volume data. NO AI estimates.

    If search volume == 0, the system LOCKS and warns the user.
    """
    state = unlock_step(state, 2, "Market Fact-Checking")

    if state.charter is None:
        raise ValueError("Step 2 requires a validated charter from Step 1.")

    # Extract keywords from the charter
    keywords = [
        state.charter.problem_statement.split()[0],  # Primary keyword
        state.charter.target_audience,
        state.charter.proposed_solution.split()[0],
    ]

    client = MarketDataClient()
    report_data = await client.build_report(keywords)

    report = MarketDataReport(**report_data)

    # RELIABILITY CHECK: If no valid data, LOCK the pipeline
    if not report.has_valid_data:
        logger.warning("PIPELINE LOCK: No market data found. Do not proceed.")

    state.market_data = report
    state.steps[2].critic_score = 1.0 if report.has_valid_data else 0.0
    state = submit_for_review(state, 2, report)
    return state


# ---------------------------------------------------------------------------
# Step 3 — Competitive Landscape
# ---------------------------------------------------------------------------

async def step3_competitive_landscape(state: VentureState) -> VentureState:
    """Step 3: Scrape pricing pages. Every claim includes a URL source."""
    state = unlock_step(state, 3, "Competitive Landscape")

    if state.market_data is None:
        raise ValueError("Step 3 requires market data from Step 2.")

    analyzer = CompetitorAnalyzer()
    matrix_data = await analyzer.build_matrix(state.market_data.keywords)

    matrix = CompetitorMatrix(
        competitors=matrix_data["competitors"],
        density_score=matrix_data["density_score"],
    )

    state.competitor_matrix = matrix
    state = submit_for_review(state, 3, matrix)
    return state


# ---------------------------------------------------------------------------
# Step 4 — Viability Gate (Code, not AI)
# ---------------------------------------------------------------------------

async def step4_viability_gate(state: VentureState) -> VentureState:
    """Step 4: Weighted algorithm calculates GO / NO-GO.

    This is 100 % code — no AI involvement.
    """
    state = unlock_step(state, 4, "Viability Gate")

    if state.market_data is None or state.competitor_matrix is None:
        raise ValueError("Step 4 requires outputs from Steps 2 and 3.")

    # Normalize market volume to 0-1 scale
    total_volume = sum(state.market_data.search_volumes.values())
    market_score = min(total_volume / 10_000, 1.0)  # 10K+ searches = max score

    # Competitor density already 0-1
    competitor_score = state.competitor_matrix.density_score

    weighted = calculate_viability_score(market_score, competitor_score)

    verdict = ViabilityVerdict.GO if weighted >= 0.5 else ViabilityVerdict.NO_GO

    dashboard = ViabilityDashboard(
        market_volume_score=market_score,
        competitor_density_score=competitor_score,
        weighted_score=weighted,
        verdict=verdict,
    )

    state.viability = dashboard
    state = submit_for_review(state, 4, dashboard)
    return state
