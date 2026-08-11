# SalesIQ — Corporate Lending Intelligence

**Explainable, graph-native loan application scoring for the IBM AI Builders Challenge (Wildcard Track: Intelligent Systems for the Future of Work)**

## One-line pitch
Predictive lead scoring already exists — but its explanations are black boxes. SalesIQ keeps every score traceable to specific evidence, works with small datasets via transparent entry, and adds a GraphRAG layer so analysts can ask "why" in plain language and get evidence-grounded answers in seconds.

---

## Quick Start

```bash
# 1. Clone and create venv
git clone https://github.com/durbhavarun/salesiq
cd salesiq
python -m venv venv
venv\Scripts\activate      # Windows PowerShell

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set environment variables
# Copy .env.example to .env and add your GROQ_API_KEY (free at console.groq.com)
# For demo without LLM: set SALESIQ_LLM_PROVIDER=none in .env

# 4. Build the knowledge graph
python graph/populate.py

# 5. Run the API
uvicorn api.main:app --reload --port 8000

# 6. Open the dashboard
# http://localhost:8000
```

---

## Architecture

```
Data (CSV/JSON)
      │
      ▼
LLM Gateway ──► PII scrub ──► cache ──► Groq/Ollama ──► schema validate
      │
      ▼
Knowledge Graph (NetworkX + SQLite)
  Accounts, Deals, Signals, ICP Versions
  Edges: HAS_DEAL, HAS_SIGNAL, REFERRED_BY, SHARES_GUARANTOR
      │                                    │
      ▼                                    ▼
Scoring Engine                      GraphRAG Layer
  ICP Fit (0.70)                     Router → 3 templates
  Intent Score (0.30)                Subgraph retrieval
  Network Proximity (separate)       LLM constrained generation
  Confidence band (display only)     Evidence-cited answers
      │                                    │
      └──────────────┬─────────────────────┘
                     ▼
              FastAPI Backend (17 endpoints)
                     │
                     ▼
              Dashboard (HTML + Chart.js + vis-network)
```

---

## Scoring Formula

```
ICP Fit Score   = Σ(weight[param] × presence[param])   21 parameters
Intent Score    = Σ(signal_confidence × polarity)       5 signal types
RawScore        = 0.70 × ICP_Fit + 0.30 × Intent
Network Prox    = graph hops to nearest won deal        SEPARATE, not blended
```

### Confidence Bands (display modifier — never blended into score)

| Band   | Trigger                              | Display behaviour              |
|--------|--------------------------------------|-------------------------------|
| HIGH   | 4+ signals, avg confidence ≥ 0.80    | Show RawScore as-is           |
| MEDIUM | 2–3 matched signals                  | Show RawScore with ±10 band   |
| LOW    | Fewer than 2 signals                 | Cap at 65, show LOW badge     |

---

## Why 21 Parameters (not 25)

3 fields excluded to prevent Network Proximity double-counting:
- `referred_by_portfolio` — powers REFERRED_BY graph traversal
- `shares_guarantor_with_won` — powers SHARES_GUARANTOR traversal
- `network_proximity_hops` — IS the Network Proximity output

1 field removed (`collateral_coverage`) — duplicate of `collateral_type`, zero independent signal.

---

## Why GraphRAG, Not Vector Embeddings

Vector embeddings (node2vec) find "similar" nodes but cannot tell you **why** two nodes are related. SalesIQ traverses actual graph paths:

```
Account A → SHARES_GUARANTOR → Account B → HAS_DEAL → Deal C (won, 2024-03)
```

This path is evidence. An embedding vector is not. Every GraphRAG answer includes `paths_used` so any claim is independently verifiable.

---

## Responsible AI

- No individual-level personal data — scores companies, not people
- Every AI-extracted signal retains full provenance (signal_id, source, confidence)
- Scoring is deterministic — identical inputs always produce identical outputs
- Human approval required for every ICP weight change (no silent auto-retraining)
- Rep overrides logged as feedback — never used as training labels
- Temporal leakage prevention: `score_eligibility_date ≤ deal_created_date` enforced at query layer

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Dashboard HTML |
| `/score/all` | POST | Score all leads from CSV, ranked |
| `/score/{lead_id}` | GET | Get stored score for one lead |
| `/score/single` | POST | Score manually-entered lead (same pipeline as bulk) |
| `/graph/lead/{lead_id}` | GET | Orbit subgraph for vis-network (READ-ONLY) |
| `/graph/all` | GET | Full knowledge graph (capped 200 nodes) |
| `/explain` | POST | GraphRAG: answer natural-language question |
| `/override` | POST | Record human override decision |
| `/audit` | GET | Full audit log |
| `/backtest/run` | POST | Date-based backtest (Spearman + Precision@10) |
| `/dashboard/summary` | GET | KPI stats for home tab |
| `/dashboard/regions` | GET | Regional breakdown |
| `/upload/deals` | POST | Upload new historical deals CSV |
| `/upload/leads` | POST | Upload new leads CSV |
| `/icp/current` | GET | Current ICP weights |
| `/icp/submit` | POST | Submit weights for approval |
| `/icp/approve` | POST | Approve pending weights |
| `/icp/reject` | POST | Reject pending weights |

Interactive docs at `http://localhost:8000/docs`

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SALESIQ_LLM_PROVIDER` | `groq` | `groq` \| `ollama` \| `none` |
| `GROQ_API_KEY` | — | Free at console.groq.com |
| `GROQ_MODEL` | `llama3-8b-8192` | Groq model ID |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Local Ollama URL |
| `SALESIQ_GRAPH_BACKEND` | `memory` | `memory` \| `neo4j` |

---

## Backtest Results (Synthetic Demo Data)

- **Spearman r ≈ 0.49** — moderate rank correlation (directional signal present)
- **Precision@10 = 0.60** — 6 of top 10 scored leads were actual wins vs 35% baseline

Lead with Precision@10 in the demo: *"6 of the top 10 leads our model ranked highest actually won — against a baseline win rate of 35%."*

---

## Stack

- **Backend:** FastAPI + Uvicorn
- **Graph:** NetworkX (in-memory) / Neo4j (Docker, optional)
- **Scoring:** Pure deterministic Python — LLM never touches scores
- **LLM:** Groq (primary, free) → Ollama (local fallback) → none (structured error)
- **Frontend:** Plain HTML + Chart.js + vis-network (no npm, no React)
- **Hosting:** Render.com free tier

---

## Running Tests

```bash
pytest tests/test_integration.py -v
```

---

## Explicit Exclusions

| Excluded | Reason |
|----------|--------|
| node2vec / embeddings | Documented black-box behavior; contradicts explainability commitment |
| ibm-watsonx-ai package | Conflicts with pandas on Python 3.13; stub exists, env-var activated only |
| Full autonomous agent | Avoids overclaiming agentic orchestration without genuine autonomous action |
| Live CRM integration | Out of scope; CSV/JSON upload only |
