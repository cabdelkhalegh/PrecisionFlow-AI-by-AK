"""Pydantic models for the venture pipeline state and data structures."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class StepStatus(str, Enum):
    """Status indicator for each step in the pipeline."""

    LOCKED = "locked"
    PROCESSING = "processing"
    REVIEW_NEEDED = "review_needed"
    VERIFIED = "verified"


class ViabilityVerdict(str, Enum):
    """Outcome of the Step 4 viability gate."""

    GO = "go"
    NO_GO = "no_go"
    OVERRIDE = "override"


class BusinessModelType(str, Enum):
    """Proven business model archetypes from the Golden Database."""

    SAAS_TIERED = "saas_tiered"
    MARKETPLACE_TRANSACTION_FEE = "marketplace_transaction_fee"
    FREEMIUM = "freemium"
    SUBSCRIPTION = "subscription"
    USAGE_BASED = "usage_based"
    LICENSING = "licensing"
    ADVERTISING = "advertising"
    AFFILIATE = "affiliate"


# ---------------------------------------------------------------------------
# Step output models
# ---------------------------------------------------------------------------


class VentureCharter(BaseModel):
    """Step 1 output: structured problem definition."""

    problem_statement: str
    target_audience: str
    proposed_solution: str
    value_proposition: str
    matched_case_studies: list[str] = Field(default_factory=list)
    database_match_found: bool = False
    approved: bool = False


class MarketDataReport(BaseModel):
    """Step 2 output: hard data from real APIs — no estimates."""

    keywords: list[str] = Field(default_factory=list)
    search_volumes: dict[str, int] = Field(default_factory=dict)
    trend_data: dict[str, Any] = Field(default_factory=dict)
    reddit_mentions: int = 0
    data_sources: list[str] = Field(default_factory=list)
    has_valid_data: bool = False


class CompetitorEntry(BaseModel):
    """A single verified competitor record."""

    name: str
    url: str
    pricing: str
    source_url: str
    features: list[str] = Field(default_factory=list)


class CompetitorMatrix(BaseModel):
    """Step 3 output: verified competitor landscape."""

    competitors: list[CompetitorEntry] = Field(default_factory=list)
    density_score: float = 0.0


class ViabilityDashboard(BaseModel):
    """Step 4 output: GO / NO-GO decision based on weighted scoring."""

    market_volume_score: float = 0.0
    competitor_density_score: float = 0.0
    weighted_score: float = 0.0
    verdict: ViabilityVerdict = ViabilityVerdict.NO_GO
    user_override: bool = False


class CustomerPersona(BaseModel):
    """Step 5 output: customer avatar derived from RAG templates."""

    name: str = ""
    age_range: str = ""
    occupation: str = ""
    pain_points: list[str] = Field(default_factory=list)
    goals: list[str] = Field(default_factory=list)
    objections: list[str] = Field(default_factory=list)
    channels: list[str] = Field(default_factory=list)


class RiskReport(BaseModel):
    """Step 6 output: objection simulation results."""

    objections: list[str] = Field(default_factory=list)
    severity_scores: dict[str, float] = Field(default_factory=dict)
    mitigation_suggestions: list[str] = Field(default_factory=list)


class MonetizationOption(BaseModel):
    """A single monetization model option."""

    model_type: BusinessModelType
    description: str = ""
    unit_economics_positive: bool = False
    estimated_price: float = 0.0
    estimated_cost: float = 0.0
    margin: float = 0.0


class MonetizationOptions(BaseModel):
    """Step 7 output: proven business model options."""

    options: list[MonetizationOption] = Field(default_factory=list)
    selected: BusinessModelType | None = None


class PartnerEntry(BaseModel):
    """A single verified supplier / partner."""

    name: str
    contact_info: str
    region: str
    verified: bool = False


class PartnerList(BaseModel):
    """Step 8 output: verified supply chain partners."""

    partners: list[PartnerEntry] = Field(default_factory=list)


class TechStackRecommendation(BaseModel):
    """Step 9 output: rule-based tech stack."""

    app_type: str = ""
    frontend: list[str] = Field(default_factory=list)
    backend: list[str] = Field(default_factory=list)
    database: list[str] = Field(default_factory=list)
    infrastructure: list[str] = Field(default_factory=list)
    rationale: str = ""


class LegalDocument(BaseModel):
    """A single legal template with filled variables."""

    template_name: str
    jurisdiction: str
    variables: dict[str, str] = Field(default_factory=dict)
    content: str = ""
    core_clauses_unmodified: bool = True


class LegalCompliance(BaseModel):
    """Step 10 output: legal documents with locked core clauses."""

    documents: list[LegalDocument] = Field(default_factory=list)


class HiringRequisition(BaseModel):
    """A single role needed to fill a skill gap."""

    role: str
    skills_required: list[str] = Field(default_factory=list)
    priority: str = "medium"


class TeamGapAnalysis(BaseModel):
    """Step 11 output: hiring requisitions based on tech stack vs founder profile."""

    founder_skills: list[str] = Field(default_factory=list)
    required_skills: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    requisitions: list[HiringRequisition] = Field(default_factory=list)


class UserStory(BaseModel):
    """A single user story in Gherkin syntax."""

    title: str
    given: str
    when: str
    then: str


class PrototypeSpec(BaseModel):
    """Step 12 output: dev-ready PRD with Gherkin user stories."""

    user_stories: list[UserStory] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)


class MarketingAsset(BaseModel):
    """A single marketing asset variation."""

    asset_type: str
    headline: str
    body: str
    channel: str
    starred: bool = False


class CampaignAssetFolder(BaseModel):
    """Step 13 output: marketing assets addressing known objections."""

    assets: list[MarketingAsset] = Field(default_factory=list)
    starred_count: int = 0


class GanttTask(BaseModel):
    """A single task in the development roadmap."""

    name: str
    duration_weeks: float
    dependencies: list[str] = Field(default_factory=list)
    start_week: int = 0


class RoadmapGantt(BaseModel):
    """Step 14 output: realistic project timeline."""

    tasks: list[GanttTask] = Field(default_factory=list)
    total_weeks: int = 0


class FinancialModel(BaseModel):
    """Step 15 output: P&L generated by code — zero AI involvement."""

    assumptions: dict[str, float] = Field(default_factory=dict)
    revenue_monthly: list[float] = Field(default_factory=list)
    costs_monthly: list[float] = Field(default_factory=list)
    profit_monthly: list[float] = Field(default_factory=list)
    break_even_month: int | None = None
    annual_revenue: float = 0.0
    annual_cost: float = 0.0
    annual_profit: float = 0.0


class DealRoom(BaseModel):
    """Step 16 output: compiled investor-ready data room."""

    pitch_deck_sections: list[dict[str, str]] = Field(default_factory=list)
    data_room_link: str = ""
    compiled_at: datetime | None = None


# ---------------------------------------------------------------------------
# Pipeline state — the single source of truth flowing through all 16 steps
# ---------------------------------------------------------------------------


class StepRecord(BaseModel):
    """Tracks the status and output of a single pipeline step."""

    step_number: int
    name: str
    status: StepStatus = StepStatus.LOCKED
    output: Any = None
    critic_score: float | None = None
    user_edits: int = 0
    approved: bool = False
    locked_at: datetime | None = None


class VentureState(BaseModel):
    """The master state object that flows through the entire LangGraph pipeline.

    Every step reads from and writes to this state. Nothing advances
    unless the previous step's ``approved`` flag is True.
    """

    venture_id: str = ""
    founder_name: str = ""
    founder_skills: list[str] = Field(default_factory=list)
    jurisdiction: str = ""
    region: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Step outputs keyed by step number (1-16)
    steps: dict[int, StepRecord] = Field(default_factory=dict)

    # Convenience accessors populated as steps complete
    charter: VentureCharter | None = None
    market_data: MarketDataReport | None = None
    competitor_matrix: CompetitorMatrix | None = None
    viability: ViabilityDashboard | None = None
    persona: CustomerPersona | None = None
    risk_report: RiskReport | None = None
    monetization: MonetizationOptions | None = None
    partner_list: PartnerList | None = None
    tech_stack: TechStackRecommendation | None = None
    legal: LegalCompliance | None = None
    team_gaps: TeamGapAnalysis | None = None
    prototype_spec: PrototypeSpec | None = None
    campaign_assets: CampaignAssetFolder | None = None
    roadmap: RoadmapGantt | None = None
    financials: FinancialModel | None = None
    deal_room: DealRoom | None = None

    # Metrics
    total_edits: int = 0
    critic_pass_rate: float = 0.0
