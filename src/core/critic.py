"""The Critic Loop — Anti-Hallucination engine.

Every AI-generated output goes through a hidden 2-step process:
1. Generator Bot drafts the content.
2. Critic Bot scores the draft against a Quality Rubric.

If the score is < threshold (default 80 %), the Generator must re-write.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.core.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rubrics — each step has specific quality criteria
# ---------------------------------------------------------------------------

QUALITY_RUBRICS: dict[int, list[str]] = {
    1: [
        "Does the problem statement describe a real, specific pain point?",
        "Is the target audience clearly defined?",
        "Is the value proposition differentiated from existing solutions?",
    ],
    5: [
        "Does the persona include specific demographics?",
        "Are pain points grounded in the validated market data?",
        "Are preferred channels listed?",
    ],
    6: [
        "Does the risk report list at least 3 concrete objections?",
        "Are severity scores assigned to each objection?",
        "Are mitigation suggestions provided?",
    ],
    7: [
        "Are unit economics (price - cost) positive for each model?",
        "Are at least 2 model options provided?",
        "Is the rationale tied to industry benchmarks?",
    ],
    12: [
        "Are user stories in valid Gherkin syntax (Given/When/Then)?",
        "Do acceptance criteria cover the core user flows?",
        "Is the spec free of ambiguous requirements?",
    ],
    13: [
        "Does the ad copy address specific objections from the Risk Report?",
        "Are specific marketing channels mentioned?",
        "Is the budget allocation realistic?",
    ],
    16: [
        "Does the pitch deck follow a standard investor format?",
        "Are all data points sourced from verified steps?",
        "Is the financial model referenced correctly?",
    ],
}


class CriticResult(BaseModel):
    """Structured output from the Critic Bot."""

    score: float = Field(ge=0.0, le=1.0, description="Quality score between 0 and 1.")
    passed: bool = False
    feedback: list[str] = Field(default_factory=list)
    rubric_scores: dict[str, float] = Field(default_factory=dict)


def _build_critic_prompt(step_number: int, draft: str) -> str:
    """Build the Critic Bot's evaluation prompt."""
    rubric = QUALITY_RUBRICS.get(step_number, [])
    rubric_text = "\n".join(f"- {r}" for r in rubric) if rubric else "- General quality check."

    return f"""You are a strict Quality Critic.  Evaluate the following draft output
from Step {step_number} of a business-building pipeline.

RUBRIC — score each criterion 0-100 and provide feedback:
{rubric_text}

DRAFT:
{draft}

Respond in JSON with this exact schema:
{{
  "score": <overall score 0.0-1.0>,
  "rubric_scores": {{"<criterion>": <score 0.0-1.0>, ...}},
  "feedback": ["<issue 1>", "<issue 2>", ...]
}}
Only output valid JSON.  No markdown fences."""


async def run_critic(step_number: int, draft_output: Any) -> CriticResult:
    """Run the Critic Bot on a draft and return the evaluation.

    The Critic uses a *separate* LLM call at low temperature to score
    the Generator's output objectively.
    """
    draft_str = (
        draft_output.model_dump_json(indent=2)
        if isinstance(draft_output, BaseModel)
        else json.dumps(draft_output, indent=2, default=str)
    )

    prompt = _build_critic_prompt(step_number, draft_str)

    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0.0,  # Critic is fully deterministic
        api_key=settings.openai_api_key,
    )

    response = await llm.ainvoke(prompt)
    content = response.content if hasattr(response, "content") else str(response)

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        logger.warning("Critic returned non-JSON response; treating as fail.")
        return CriticResult(score=0.0, passed=False, feedback=["Critic output was unparsable."])

    score = float(data.get("score", 0.0))
    return CriticResult(
        score=score,
        passed=score >= settings.critic_pass_threshold,
        feedback=data.get("feedback", []),
        rubric_scores=data.get("rubric_scores", {}),
    )


async def critic_loop(
    step_number: int,
    generator_fn: Any,
    *generator_args: Any,
    **generator_kwargs: Any,
) -> tuple[Any, CriticResult]:
    """Run the Generator -> Critic loop until the output passes or retries are exhausted.

    Returns the final (output, critic_result) pair.
    """
    max_retries = settings.max_critic_retries
    last_result = CriticResult(score=0.0, passed=False, feedback=["Not yet evaluated."])

    for attempt in range(1, max_retries + 1):
        logger.info("Step %d — Generator attempt %d/%d", step_number, attempt, max_retries)

        draft = await generator_fn(*generator_args, **generator_kwargs)
        last_result = await run_critic(step_number, draft)

        logger.info("Step %d — Critic score: %.2f", step_number, last_result.score)

        if last_result.passed:
            return draft, last_result

        # Feed critic feedback back into the next generator call
        generator_kwargs["critic_feedback"] = last_result.feedback

    logger.warning(
        "Step %d — Critic loop exhausted after %d attempts (score=%.2f).",
        step_number,
        max_retries,
        last_result.score,
    )
    return draft, last_result  # type: ignore[possibly-undefined]
