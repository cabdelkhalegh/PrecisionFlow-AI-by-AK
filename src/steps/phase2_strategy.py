"""Phase 2: Strategy (The Logic) — Steps 5-8.

Step 5: Customer Persona Construction (RAG)
Step 6: Objection Simulation (adversarial AI)
Step 7: Business Model Architect (proven models)
Step 8: Supply Chain (real APIs)
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI

from src.core.config import settings
from src.core.critic import critic_loop
from src.core.quality_gate import submit_for_review, unlock_step
from src.database.vector_store import GoldenDatabase
from src.integrations.suppliers import SupplierFinder
from src.models.venture import (
    BusinessModelType,
    CustomerPersona,
    MonetizationOption,
    MonetizationOptions,
    PartnerList,
    RiskReport,
    VentureState,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 5 — Customer Persona Construction
# ---------------------------------------------------------------------------

async def _generate_persona(
    market_data: dict[str, Any],
    charter: dict[str, Any],
    critic_feedback: list[str] | None = None,
) -> CustomerPersona:
    """Generator Bot for Step 5: draft a customer avatar from RAG templates."""
    feedback_text = ""
    if critic_feedback:
        feedback_text = (
            "\n\nPREVIOUS DRAFT FAILED QUALITY CHECK. Address this feedback:\n"
            + "\n".join(f"- {f}" for f in critic_feedback)
        )

    prompt = f"""You are a customer research specialist.  Create a detailed customer
persona based on the validated market data and venture charter.

MARKET DATA:
{json.dumps(market_data, indent=2, default=str)}

VENTURE CHARTER:
{json.dumps(charter, indent=2, default=str)}
{feedback_text}

Return JSON:
{{
  "name": "<fictional persona name>",
  "age_range": "<e.g. 25-35>",
  "occupation": "<job title>",
  "pain_points": ["<pain 1>", "<pain 2>", ...],
  "goals": ["<goal 1>", "<goal 2>", ...],
  "objections": ["<objection 1>", ...],
  "channels": ["<preferred channel 1>", ...]
}}
Only output valid JSON."""

    llm = ChatGoogleGenerativeAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        google_api_key=settings.gemini_api_key,
    )
    response = await llm.ainvoke(prompt)
    content = response.content if hasattr(response, "content") else str(response)
    data = json.loads(content)
    return CustomerPersona(**data)


async def step5_customer_persona(state: VentureState) -> VentureState:
    """Step 5: Build customer avatar via RAG persona templates."""
    state = unlock_step(state, 5, "Customer Persona Construction")

    market_dict = state.market_data.model_dump() if state.market_data else {}
    charter_dict = state.charter.model_dump() if state.charter else {}

    persona, critic_result = await critic_loop(
        step_number=5,
        generator_fn=_generate_persona,
        market_data=market_dict,
        charter=charter_dict,
    )

    state.persona = persona
    state.steps[5].critic_score = critic_result.score
    state = submit_for_review(state, 5, persona)
    return state


# ---------------------------------------------------------------------------
# Step 6 — Objection Simulation
# ---------------------------------------------------------------------------

async def _generate_objections(
    persona: dict[str, Any],
    charter: dict[str, Any],
    critic_feedback: list[str] | None = None,
) -> RiskReport:
    """Generator Bot for Step 6: AI adopts the persona's negative traits."""
    feedback_text = ""
    if critic_feedback:
        feedback_text = (
            "\n\nPREVIOUS DRAFT FAILED. Fix these issues:\n"
            + "\n".join(f"- {f}" for f in critic_feedback)
        )

    prompt = f"""You are now role-playing as a SKEPTICAL version of this customer persona.
Your job is to REJECT the proposed business idea.  Be harsh but realistic.

PERSONA:
{json.dumps(persona, indent=2)}

VENTURE:
{json.dumps(charter, indent=2)}
{feedback_text}

List every reason this will fail.  Return JSON:
{{
  "objections": ["<reason 1>", "<reason 2>", ...],
  "severity_scores": {{"<reason 1>": 0.0-1.0, ...}},
  "mitigation_suggestions": ["<suggestion 1>", ...]
}}
Only output valid JSON."""

    llm = ChatGoogleGenerativeAI(
        model=settings.llm_model,
        temperature=0.3,  # Slightly more creative for adversarial simulation
        google_api_key=settings.gemini_api_key,
    )
    response = await llm.ainvoke(prompt)
    content = response.content if hasattr(response, "content") else str(response)
    data = json.loads(content)
    return RiskReport(**data)


async def step6_objection_simulation(state: VentureState) -> VentureState:
    """Step 6: The AI tries to reject the user's idea."""
    state = unlock_step(state, 6, "Objection Simulation")

    persona_dict = state.persona.model_dump() if state.persona else {}
    charter_dict = state.charter.model_dump() if state.charter else {}

    risk_report, critic_result = await critic_loop(
        step_number=6,
        generator_fn=_generate_objections,
        persona=persona_dict,
        charter=charter_dict,
    )

    state.risk_report = risk_report
    state.steps[6].critic_score = critic_result.score
    state = submit_for_review(state, 6, risk_report)
    return state


# ---------------------------------------------------------------------------
# Step 7 — Business Model Architect
# ---------------------------------------------------------------------------

async def _generate_monetization(
    charter: dict[str, Any],
    market_data: dict[str, Any],
    critic_feedback: list[str] | None = None,
) -> MonetizationOptions:
    """Generator Bot for Step 7: retrieve proven business models."""
    feedback_text = ""
    if critic_feedback:
        feedback_text = (
            "\n\nPREVIOUS DRAFT FAILED. Fix these issues:\n"
            + "\n".join(f"- {f}" for f in critic_feedback)
        )

    valid_types = ", ".join(t.value for t in BusinessModelType)

    prompt = f"""You are a business model strategist.  Recommend 3 proven monetization
models for this venture.  Each model MUST have positive unit economics
(price > cost).

VENTURE:
{json.dumps(charter, indent=2)}

MARKET DATA:
{json.dumps(market_data, indent=2, default=str)}
{feedback_text}

Valid model types: {valid_types}

Return JSON:
{{
  "options": [
    {{
      "model_type": "<one of the valid types>",
      "description": "<how this model applies>",
      "unit_economics_positive": true,
      "estimated_price": <float>,
      "estimated_cost": <float>,
      "margin": <float>
    }},
    ...
  ]
}}
Only output valid JSON."""

    llm = ChatGoogleGenerativeAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        google_api_key=settings.gemini_api_key,
    )
    response = await llm.ainvoke(prompt)
    content = response.content if hasattr(response, "content") else str(response)
    data = json.loads(content)

    options = []
    for opt in data.get("options", []):
        try:
            options.append(MonetizationOption(**opt))
        except Exception:
            logger.warning("Skipping invalid monetization option: %s", opt)

    return MonetizationOptions(options=options)


async def step7_business_model(state: VentureState) -> VentureState:
    """Step 7: Retrieve 3 proven models with verified unit economics."""
    state = unlock_step(state, 7, "Business Model Architect")

    charter_dict = state.charter.model_dump() if state.charter else {}
    market_dict = state.market_data.model_dump() if state.market_data else {}

    monetization, critic_result = await critic_loop(
        step_number=7,
        generator_fn=_generate_monetization,
        charter=charter_dict,
        market_data=market_dict,
    )

    # Reliability check: ensure unit economics are positive
    for opt in monetization.options:
        if opt.estimated_price <= opt.estimated_cost:
            opt.unit_economics_positive = False
            logger.warning("Model %s has negative unit economics.", opt.model_type)

    state.monetization = monetization
    state.steps[7].critic_score = critic_result.score
    state = submit_for_review(state, 7, monetization)
    return state


# ---------------------------------------------------------------------------
# Step 8 — Supply Chain
# ---------------------------------------------------------------------------

async def step8_supply_chain(state: VentureState) -> VentureState:
    """Step 8: Find real suppliers/partners in the user's region via APIs."""
    state = unlock_step(state, 8, "Supply Chain")

    query = ""
    if state.charter:
        query = state.charter.proposed_solution
    region = state.region or "United States"

    finder = SupplierFinder()
    partners = await finder.find_partners(query, region)

    partner_list = PartnerList(partners=partners)
    state.partner_list = partner_list
    state = submit_for_review(state, 8, partner_list)
    return state
