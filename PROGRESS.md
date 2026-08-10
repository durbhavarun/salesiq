# SalesIQ Build Progress Log

> At the start of every new Bob session, say:
> "Read PROGRESS.md, then continue from where we left off."
> Bob has no memory between sessions — this file IS the memory.

## Project
- Product: SalesIQ Version 2 — Corporate Lending
- Challenge: IBM AI Builders Challenge (Wildcard: Intelligent Systems for the Future of Work)
- GitHub: https://github.com/durbhavarun/salesiq
- Live demo (after Day 20): https://salesiq-durbhavarun.onrender.com

## Locked Decisions
- LLM: Groq (primary) → Ollama (fallback) → none
- Hosting: Render.com free tier (no card)
- Graph hosted: NetworkX in-memory
- Graph local: Neo4j via Docker
- ICP params: 22 (NOT 25 — 3 excluded to prevent Network Proximity double-count)
- Scoring: Pure deterministic Python — LLM never touches scores
- Confidence: Display modifier only — never blended into RawScore
- ERROR anomaly: Blocks scoring entirely
- WARNING anomaly: Score proceeds with flag

## Completed
- [x] Day 1: All tools installed (VS Code, Python 3.11, Git, Docker, Ollama)
- [x] Day 2: GitHub repo, folder structure, .gitignore, .env, docker-compose.yml, Dockerfile

## Current Status
Day 2 complete. Ready for Day 3: requirements.txt + virtual environment.

## Next Step
Day 3: Create requirements.txt, create venv, install all packages.
Command to start: pip install -r requirements.txt

## File Map
salesiq/
├── .env                         (local only — gitignored)
├── .gitignore
├── PROGRESS.md
├── README.md                    (Day 19)
├── requirements.txt             (Day 3)
├── docker-compose.yml
├── Dockerfile
├── render.yaml                  (Day 20)
├── data/demo/
│   ├── historical_deals.csv     (Day 4)
│   ├── new_leads.csv            (Day 5)
│   └── signals.json             (Day 5)
├── graph/
│   ├── schema.cypher            (Day 6)
│   ├── populate.py              (Day 6)
│   └── queries/
│       ├── network_proximity.cypher  (Day 6)
│       └── eligible_signals.cypher   (Day 6)
├── engine/
│   ├── __init__.py              (Day 7)
│   ├── icp_weights.py           (Day 7)
│   ├── scoring_engine.py        (Day 8)
│   ├── anomaly_detector.py      (Day 9)
│   ├── data_profiler.py         (Day 10)
│   ├── llm_gateway.py           (Day 11)
│   ├── backtest/
│   │   ├── __init__.py          (Day 10)
│   │   └── split.py             (Day 10)
│   └── graphrag/
│       ├── __init__.py          (Day 12)
│       ├── router.py            (Day 12)
│       └── templates.py         (Day 12)
├── api/
│   ├── __init__.py              (Day 13)
│   └── main.py                  (Day 13)
├── frontend/
│   └── templates/
│       └── dashboard.html       (Day 14)
├── tests/
│   ├── test_scoring.py          (Day 18)
│   ├── test_anomaly.py          (Day 18)
│   ├── test_api.py              (Day 18)
│   └── test_backtest.py         (Day 18)
└── docs/
    ├── demo_script.md           (Day 19)
    ├── architecture_decisions.md (Day 19)
    └── responsible_ai.md        (Day 19)
