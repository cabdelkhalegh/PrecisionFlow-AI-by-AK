"""Phase 3: Execution (The Build) — Steps 9-12.

Step 9:  Tech Stack Recommendation (rule-based, not AI)
Step 10: Legal Compliance (template fill, AI forbidden from altering core clauses)
Step 11: Team Gap Analysis (code comparison)
Step 12: Prototype Spec (Gherkin syntax)
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI

from src.core.config import settings
from src.core.critic import critic_loop
from src.core.quality_gate import submit_for_review, unlock_step
from src.models.venture import (
    HiringRequisition,
    LegalCompliance,
    LegalDocument,
    PrototypeSpec,
    TeamGapAnalysis,
    TechStackRecommendation,
    UserStory,
    VentureState,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tech stack rules — deterministic, no AI
# ---------------------------------------------------------------------------

TECH_STACK_RULES: dict[str, TechStackRecommendation] = {
    "mobile_app": TechStackRecommendation(
        app_type="Mobile App",
        frontend=["React Native", "Expo"],
        backend=["Node.js", "Express"],
        database=["PostgreSQL", "Redis"],
        infrastructure=["AWS", "Firebase"],
        rationale="Cross-platform mobile framework with proven scalability.",
    ),
    "web_app": TechStackRecommendation(
        app_type="Web App",
        frontend=["Next.js", "React", "Tailwind CSS"],
        backend=["Node.js", "Express"],
        database=["PostgreSQL", "Redis"],
        infrastructure=["Vercel", "AWS"],
        rationale="Modern full-stack web framework with SSR and edge capabilities.",
    ),
    "ai_app": TechStackRecommendation(
        app_type="AI Application",
        frontend=["React", "Next.js"],
        backend=["Python", "FastAPI", "LangChain"],
        database=["PostgreSQL", "pgvector", "Redis"],
        infrastructure=["AWS", "Modal", "Docker"],
        rationale="Python-first AI stack with vector database for embeddings.",
    ),
    "marketplace": TechStackRecommendation(
        app_type="Marketplace",
        frontend=["Next.js", "React"],
        backend=["Node.js", "Express", "Stripe Connect"],
        database=["PostgreSQL", "Elasticsearch"],
        infrastructure=["AWS", "Cloudflare"],
        rationale="Marketplace stack with payment processing and search.",
    ),
    "saas": TechStackRecommendation(
        app_type="SaaS Platform",
        frontend=["Next.js", "React", "Tailwind CSS"],
        backend=["Node.js", "Express"],
        database=["PostgreSQL", "Redis"],
        infrastructure=["AWS", "Vercel", "Stripe"],
        rationale="Standard SaaS stack with subscription billing.",
    ),
    "ecommerce": TechStackRecommendation(
        app_type="E-Commerce",
        frontend=["Next.js", "React"],
        backend=["Node.js", "Medusa.js"],
        database=["PostgreSQL", "Redis"],
        infrastructure=["AWS", "Cloudflare", "Stripe"],
        rationale="Headless commerce stack with flexible storefront.",
    ),
}

DEFAULT_STACK = TechStackRecommendation(
    app_type="General Web Application",
    frontend=["Next.js", "React", "Tailwind CSS"],
    backend=["Python", "FastAPI"],
    database=["PostgreSQL"],
    infrastructure=["AWS", "Docker"],
    rationale="General-purpose stack suitable for most applications.",
)


def _classify_app_type(charter_text: str) -> str:
    """Simple keyword-based classifier — no AI needed."""
    text = charter_text.lower()
    if any(kw in text for kw in ["mobile", "ios", "android", "phone"]):
        return "mobile_app"
    if any(kw in text for kw in ["ai", "machine learning", "llm", "gpt", "chatbot"]):
        return "ai_app"
    if any(kw in text for kw in ["marketplace", "two-sided", "platform connecting"]):
        return "marketplace"
    if any(kw in text for kw in ["saas", "subscription", "b2b software"]):
        return "saas"
    if any(kw in text for kw in ["ecommerce", "e-commerce", "online store", "shop"]):
        return "ecommerce"
    return "web_app"


# ---------------------------------------------------------------------------
# Step 9 — Tech Stack Recommendation (rule-based)
# ---------------------------------------------------------------------------

async def step9_tech_stack(state: VentureState) -> VentureState:
    """Step 9: Rule-based tech stack recommendation. No AI."""
    state = unlock_step(state, 9, "Tech Stack Recommendation")

    charter_text = ""
    if state.charter:
        charter_text = f"{state.charter.proposed_solution} {state.charter.value_proposition}"

    app_type = _classify_app_type(charter_text)
    tech_stack = TECH_STACK_RULES.get(app_type, DEFAULT_STACK)

    state.tech_stack = tech_stack
    state = submit_for_review(state, 9, tech_stack)
    return state


# ---------------------------------------------------------------------------
# Step 10 — Legal Compliance (template fill — AI forbidden from core clauses)
# ---------------------------------------------------------------------------

LEGAL_TEMPLATES: dict[str, dict[str, str]] = {
    "terms_of_service": {
        "name": "Terms of Service",
        "core_clause": (
            "By accessing or using the Service, you agree to be bound by these Terms. "
            "If you disagree with any part of the terms, you may not access the Service."
        ),
    },
    "privacy_policy": {
        "name": "Privacy Policy",
        "core_clause": (
            "We collect information you provide directly to us, such as when you create "
            "an account, make a purchase, or contact us for support."
        ),
    },
    "nda": {
        "name": "Non-Disclosure Agreement",
        "core_clause": (
            "The Receiving Party agrees to hold and maintain in strict confidence all "
            "Confidential Information disclosed by the Disclosing Party."
        ),
    },
    "founder_agreement": {
        "name": "Founder Agreement",
        "core_clause": (
            "The Founders agree to assign and hereby assign to the Company all right, "
            "title, and interest in and to all Intellectual Property created by the Founders."
        ),
    },
}


async def step10_legal_compliance(state: VentureState) -> VentureState:
    """Step 10: Fill legal templates. AI forbidden from altering core clauses."""
    state = unlock_step(state, 10, "Legal Compliance")

    variables = {
        "company_name": state.founder_name or "TBD",
        "jurisdiction": state.jurisdiction or "Delaware, USA",
        "date": state.created_at.strftime("%B %d, %Y"),
    }

    documents: list[LegalDocument] = []
    for template_key, template in LEGAL_TEMPLATES.items():
        # AI only fills the variables — core clauses are LOCKED
        content = f"# {template['name']}\n\n"
        content += f"Effective Date: {variables['date']}\n"
        content += f"Company: {variables['company_name']}\n"
        content += f"Jurisdiction: {variables['jurisdiction']}\n\n"
        content += f"## Core Terms\n\n{template['core_clause']}\n"

        doc = LegalDocument(
            template_name=template["name"],
            jurisdiction=variables["jurisdiction"],
            variables=variables,
            content=content,
            core_clauses_unmodified=True,  # AI cannot touch this
        )
        documents.append(doc)

    legal = LegalCompliance(documents=documents)
    state.legal = legal
    state = submit_for_review(state, 10, legal)
    return state


# ---------------------------------------------------------------------------
# Step 11 — Team Gap Analysis
# ---------------------------------------------------------------------------

async def step11_team_gap_analysis(state: VentureState) -> VentureState:
    """Step 11: Compare tech stack requirements against founder's skills."""
    state = unlock_step(state, 11, "Team Gap Analysis")

    required_skills: list[str] = []
    if state.tech_stack:
        required_skills.extend(state.tech_stack.frontend)
        required_skills.extend(state.tech_stack.backend)
        required_skills.extend(state.tech_stack.database)
        required_skills.extend(state.tech_stack.infrastructure)

    founder_skills = state.founder_skills or []
    founder_set = {s.lower() for s in founder_skills}
    gaps = [s for s in required_skills if s.lower() not in founder_set]

    requisitions: list[HiringRequisition] = []
    # Group gaps by role
    dev_gaps = [g for g in gaps if g in (state.tech_stack.frontend + state.tech_stack.backend if state.tech_stack else [])]
    infra_gaps = [g for g in gaps if g in (state.tech_stack.infrastructure if state.tech_stack else [])]
    db_gaps = [g for g in gaps if g in (state.tech_stack.database if state.tech_stack else [])]

    if dev_gaps:
        requisitions.append(
            HiringRequisition(role="Full-Stack Developer", skills_required=dev_gaps, priority="high")
        )
    if infra_gaps:
        requisitions.append(
            HiringRequisition(role="DevOps Engineer", skills_required=infra_gaps, priority="medium")
        )
    if db_gaps:
        requisitions.append(
            HiringRequisition(role="Database Administrator", skills_required=db_gaps, priority="medium")
        )

    analysis = TeamGapAnalysis(
        founder_skills=founder_skills,
        required_skills=required_skills,
        gaps=gaps,
        requisitions=requisitions,
    )

    state.team_gaps = analysis
    state = submit_for_review(state, 11, analysis)
    return state


# ---------------------------------------------------------------------------
# Step 12 — Prototype Spec (Gherkin)
# ---------------------------------------------------------------------------

async def _generate_gherkin(
    persona: dict[str, Any],
    charter: dict[str, Any],
    tech_stack: dict[str, Any],
    critic_feedback: list[str] | None = None,
) -> PrototypeSpec:
    """Generator Bot for Step 12: convert user stories to Gherkin syntax."""
    feedback_text = ""
    if critic_feedback:
        feedback_text = (
            "\n\nPREVIOUS DRAFT FAILED. Fix:\n"
            + "\n".join(f"- {f}" for f in critic_feedback)
        )

    prompt = f"""Convert the user's needs into developer-ready user stories in
standard Gherkin syntax (Given/When/Then).

PERSONA:
{json.dumps(persona, indent=2)}

VENTURE:
{json.dumps(charter, indent=2)}

TECH STACK:
{json.dumps(tech_stack, indent=2)}
{feedback_text}

Return JSON:
{{
  "user_stories": [
    {{
      "title": "<story title>",
      "given": "<precondition>",
      "when": "<action>",
      "then": "<expected outcome>"
    }},
    ...
  ],
  "acceptance_criteria": ["<criterion 1>", "<criterion 2>", ...]
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

    stories = [UserStory(**s) for s in data.get("user_stories", [])]
    return PrototypeSpec(
        user_stories=stories,
        acceptance_criteria=data.get("acceptance_criteria", []),
    )


async def step12_prototype_spec(state: VentureState) -> VentureState:
    """Step 12: Produce a dev-ready PRD in Gherkin syntax."""
    state = unlock_step(state, 12, "Prototype Spec")

    persona_dict = state.persona.model_dump() if state.persona else {}
    charter_dict = state.charter.model_dump() if state.charter else {}
    tech_dict = state.tech_stack.model_dump() if state.tech_stack else {}

    spec, critic_result = await critic_loop(
        step_number=12,
        generator_fn=_generate_gherkin,
        persona=persona_dict,
        charter=charter_dict,
        tech_stack=tech_dict,
    )

    state.prototype_spec = spec
    state.steps[12].critic_score = critic_result.score
    state = submit_for_review(state, 12, spec)
    return state
