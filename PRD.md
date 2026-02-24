# 🧰 The Entrepreneur's Toolbox — Product Requirements Document

**Product:** The Entrepreneur's Toolbox v5.0  
**Codename:** PrecisionFlow Venture Operating System  
**Version:** 5.0.0  
**Author:** AK (Abdel Khalegh Cherif Hamadi)  
**Repo:** `cabdelkhalegh/PrecisionFlow-AI-by-AK`  
**Status:** Active Development — PR #1 open  

---

## 1. Vision

> **"The user is the Pilot; the AI is the Engine."**

The Entrepreneur's Toolbox is a Verify-then-Proceed Venture Operating System. It takes a founder's raw idea and runs it through 16 rigorous, human-gated steps — producing a fully verified, hallucination-proof business plan backed by real APIs, real data, and real math.

**What it is NOT:**
- It is NOT a chatbot that generates a 10-page business plan in one shot.
- It is NOT an AI that tells you what you want to hear.
- It IS a structured execution system where every output is checked, challenged, and approved before advancing.

---

## 2. Core Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  FastAPI Control Room (src/api/app.py)                          │
│  HITL endpoints: start, approve, edit, override, metrics, star  │
├─────────────────────────────────────────────────────────────────┤
│  LangGraph Orchestrator (src/core/orchestrator.py)              │
│  16-node linear graph with human interrupt checkpoints          │
├──────────────────┬──────────────────┬───────────────────────────┤
│  Critic Loop     │  Quality Gate    │  Golden Database           │
│  (critic.py)     │  (quality_gate)  │  (vector_store.py)         │
│  Generator +     │  Approve before  │  Supabase + pgvector       │
│  Critic pattern  │  advancing       │  1000+ business models     │
├──────────────────┴──────────────────┴───────────────────────────┤
│  Step Implementations (src/steps/)                               │
│  phase1_validation · phase2_strategy · phase3_execution ·       │
│  phase4_scale                                                    │
├─────────────────────────────────────────────────────────────────┤
│  Integrations (src/integrations/)                                │
│  market_data · competitors · suppliers                           │
├─────────────────────────────────────────────────────────────────┤
│  Financial Engine (src/financial/engine.py)                      │
│  100% NumPy — zero AI for numbers                                │
└─────────────────────────────────────────────────────────────────┘
```

### Anti-Hallucination Guarantee
- **Market data:** Real APIs only (Semrush, Google Trends, Reddit) — not LLM imagination
- **Financial model:** Pure NumPy — no AI for numbers
- **Every AI output:** Runs through Generator → Critic loop before surfacing to founder
- **Every step:** Locked behind HITL Quality Gate — founder must approve before advancing

---

## 3. The 16-Step Verified Pipeline

### Phase 1: Validation (Steps 1–4)

| Step | Name | What It Does | Data Source |
|------|------|--------------|-------------|
| 1 | Problem Definition | Extracts Venture Charter (problem, customer, hypothesis) | LLM + Golden DB RAG |
| 2 | Market Fact-Checking | TAM/SAM/SOM with growth trend + Reddit pain validation | Semrush, Google Trends, Reddit APIs |
| 3 | Competitive Landscape | Named competitors with cited URLs and differentiation gaps | Competitor API + LLM |
| 4 | Viability Gate | Weighted algorithmic score (code) — GO / MARGINAL / NO-GO | NumPy weighted algo |

### Phase 2: Strategy (Steps 5–8)

| Step | Name | What It Does | Data Source |
|------|------|--------------|-------------|
| 5 | Customer Persona | Evidence-based persona via RAG from Golden Database | pgvector similarity search |
| 6 | Adversarial Objection Simulation | Devil's advocate: 10 hardest objections + counter-responses | LLM (Critic loop) |
| 7 | Business Model Architect | Selects + validates business model with unit economics check | LLM + code validation |
| 8 | Supply Chain Mapping | Suppliers + logistics with real location data | Google Places API |

### Phase 3: Execution (Steps 9–12)

| Step | Name | What It Does | Data Source |
|------|------|--------------|-------------|
| 9 | Tech Stack Recommendation | Rule-based stack selection (not AI opinion) | Rule engine |
| 10 | Legal Compliance | Jurisdiction-specific requirements with locked core clauses | LLM + legal templates |
| 11 | Team Gap Analysis | Required roles vs founder skills → gap matrix | LLM + skills analysis |
| 12 | Prototype Spec | Gherkin-syntax BDD feature spec for MVP | LLM |

### Phase 4: Scale (Steps 13–16)

| Step | Name | What It Does | Data Source |
|------|------|--------------|-------------|
| 13 | Marketing Asset Generation | Objection-based copy for each channel | LLM (Critic loop) |
| 14 | Gantt Roadmap | Historical-data milestone timeline | LLM + project patterns |
| 15 | Financial Model | Revenue/cost/breakeven projections | NumPy only |
| 16 | Deal Room | Compiled pitch deck-ready summary | Aggregated from all steps |

---

## 4. Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| API | FastAPI 0.115+ | Async, type-safe, fast |
| Orchestration | LangGraph 0.2+ | HITL graph with interrupt nodes |
| LLM | Google Gemini via LangChain | Cost-efficient, fast |
| Database | Supabase (PostgreSQL 15+) | Auth + pgvector + real-time |
| Vector Search | pgvector | Semantic similarity for RAG |
| Data Science | NumPy 1.26+ | Financial engine |
| Validation | Pydantic v2 | State models, strict types |
| Testing | pytest + pytest-asyncio | 27 tests across core modules |

---

## 5. HITL Endpoints (FastAPI Control Room)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/venture/start` | Initialize venture + run Step 1 |
| `GET`  | `/venture/{id}/status` | Full pipeline status for UI timeline |
| `POST` | `/venture/{id}/approve` | Approve a step → advance to next |
| `PUT`  | `/venture/{id}/edit` | Edit step output before approving |
| `POST` | `/venture/{id}/override-viability` | Override NO-GO verdict (founder's call) |
| `POST` | `/venture/{id}/star-assets` | Star preferred marketing assets |
| `POST` | `/venture/{id}/select-model` | Confirm business model selection |
| `GET`  | `/venture/{id}/metrics` | Critic pass rate, edit count, time per step |
| `GET`  | `/health` | Health check |

---

## 6. Golden Database (pgvector)

**Purpose:** RAG store of 1000+ validated business models used to ground AI outputs in real precedents.

**Schema:**
```sql
CREATE TABLE business_models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_name TEXT NOT NULL,
    industry TEXT NOT NULL,
    model_type TEXT NOT NULL,
    problem_solved TEXT,
    target_customer TEXT,
    revenue_model TEXT,
    key_metrics JSONB,
    lessons_learned TEXT,
    embedding vector(1536),
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

**Usage:** Step 1 (Problem Definition) and Step 5 (Customer Persona) pull semantically similar models via `match_business_models()` RPC.

---

## 7. Quality Gate Logic

Each step transitions through these states:
```
LOCKED → (prerequisite met) → PROCESSING → (AI done) → REVIEW_NEEDED → (founder approves) → VERIFIED
```

**Unlock rules:**
- Step 1: always unlocked (entry point)
- Steps 2–16: previous step must be VERIFIED

**Critic Loop:**
- Generator Bot creates output
- Critic Bot evaluates for hallucinations, unsupported claims, logical gaps
- If critic score < threshold: regenerate (max 3 attempts)
- Critic pass rate tracked in metrics

---

## 8. Running Locally

```bash
# 1. Clone and checkout PR branch
git clone https://github.com/cabdelkhalegh/PrecisionFlow-AI-by-AK.git
cd PrecisionFlow-AI-by-AK
git checkout claude/build-toolbox-verification-S345Q

# 2. Install dependencies
pip install -e ".[dev]"

# 3. Configure environment
cp .env.example .env
# Edit .env: add GOOGLE_API_KEY, SUPABASE_URL, SUPABASE_KEY, etc.

# 4. Set up database
# Run migrations/001_golden_database.sql in Supabase SQL Editor
# Then seed: python scripts/seed_golden_database.py

# 5. Start the API
uvicorn src.api.main:app --reload --port 8000

# 6. Open Control Room
# http://localhost:8000/docs (Swagger UI)

# 7. Run tests
python -m pytest tests/ -v
```

---

## 9. Environment Variables

```env
# LLM
GOOGLE_API_KEY=your_gemini_key

# Supabase
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_anon_key

# External APIs (for market data)
SEMRUSH_API_KEY=optional
REDDIT_CLIENT_ID=optional
REDDIT_CLIENT_SECRET=optional
GOOGLE_PLACES_API_KEY=optional

# App settings
ENVIRONMENT=development
LOG_LEVEL=INFO
```

---

## 10. Roadmap

### v5.0 (Current — PR #1)
- [x] 16-step pipeline implemented
- [x] FastAPI Control Room with HITL endpoints
- [x] LangGraph orchestrator
- [x] Critic Loop (anti-hallucination)
- [x] Quality Gate system
- [x] NumPy financial engine
- [x] pgvector Golden Database
- [x] Market data integrations
- [x] 27 tests

### v5.1 (Next)
- [ ] React frontend for Control Room UI (vertical timeline view)
- [ ] Supabase Auth (founder login/session persistence)
- [ ] Venture state persistence (swap in-memory dict for Supabase rows)
- [ ] Email notifications at step completion
- [ ] Export Deal Room to PDF

### v5.2 (Future)
- [ ] Multi-founder collaboration
- [ ] Regional market data (MENA, Africa, Europe)
- [ ] Industry-specific templates
- [ ] Integrations: Stripe (for SaaS models), LinkedIn (team sourcing)
- [ ] Mobile-responsive UI

---

## 11. File Structure

```
PrecisionFlow-AI-by-AK/
├── src/
│   ├── api/
│   │   ├── app.py              # FastAPI routes (HITL Control Room)
│   │   └── main.py             # Uvicorn entry point
│   ├── core/
│   │   ├── config.py           # Settings (pydantic-settings)
│   │   ├── critic.py           # Critic Loop (Generator/Critic pattern)
│   │   ├── orchestrator.py     # LangGraph pipeline
│   │   └── quality_gate.py     # Step state management
│   ├── database/
│   │   └── vector_store.py     # Supabase pgvector RAG
│   ├── financial/
│   │   └── engine.py           # NumPy financial model
│   ├── integrations/
│   │   ├── competitors.py      # Competitor analysis
│   │   ├── market_data.py      # Semrush, Google Trends, Reddit
│   │   └── suppliers.py        # Google Places supply chain
│   ├── models/
│   │   └── venture.py          # Pydantic v2 state models
│   └── steps/
│       ├── phase1_validation.py   # Steps 1–4
│       ├── phase2_strategy.py     # Steps 5–8
│       ├── phase3_execution.py    # Steps 9–12
│       └── phase4_scale.py        # Steps 13–16
├── tests/
│   ├── test_financial_engine.py
│   ├── test_models.py
│   └── test_quality_gate.py
├── migrations/
│   └── 001_golden_database.sql
├── scripts/
│   └── seed_golden_database.py
├── .env.example
├── pyproject.toml
├── requirements.txt
└── PRD.md                      # This file
```

---

*Built by AK — field engineer turned builder. 14+ years on gas turbines; now building AI systems on the side.*
