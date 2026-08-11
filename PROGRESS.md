# PROGRESS.md — SalesIQ Session Memory
# Read this at the start of every Bob session before doing anything else.

## Current Status
Days 1-20 COMPLETE. All files written and verified. Ready to run run_all.ps1.

## What is built and working
- Days 1-3:  Tools, GitHub repo, venv, requirements.txt
- Day 4:     data/demo/historical_deals.csv (200 deals, 3 planted patterns)
- Day 5:     data/demo/new_leads.csv (25 leads), data/demo/signals.json (fixed dates, only acc_222 stale)
- Day 6:     graph/populate.py (NetworkX + Neo4j backends, graph.pkl)
             NODE attr: "type" | EDGE attr: "rel"  <- never change these
- Day 7:     engine/icp_weights.py (21 params, derive_weights, banding, INTENT_SIGNAL_TYPES)
- Day 8:     engine/scoring_engine.py
             Returns: "score", "raw_score", "icp_fit", "intent_score",
                      "confidence", "network_proximity" (dict with "hops","display")
             NEVER: "total_score", "confidence_band", "network_proximity_hops"
- Day 9:     engine/anomaly_detector.py (7 rules, rule names are strings e.g. "stale_signals")
- Day 10:    engine/backtest/split.py -- run_backtest(train_df, test_df, signals_list)
- Day 11:    engine/llm_gateway.py (Groq->Ollama->none, PII scrub, cache, schema validate)
             _VALID_SIGNAL_TYPES matches INTENT_SIGNAL_TYPES exactly
- Day 12:    engine/graphrag/__init__.py, router.py, templates.py
             All graph traversal uses "type"/"rel" attrs (matches populate.py)
- Day 13:    api/main.py (17 endpoints, derive_weights not compute_weights,
             correct backtest call, correct score key names)
- Day 14:    frontend/templates/dashboard.html (purple theme, 4 KPIs, vis-network)
             JS uses getScore(l)=l.score, getConf(l)=l.confidence, getProxDisplay reads l.network_proximity.display
- Day 15:    Score cards + evidence trail + lead orbit view (all in dashboard.html + api/main.py)
- Day 16:    ICP approval workflow (submit/approve/reject endpoints + UI in Engine Room tab)
- Day 17:    Backtest endpoint + display (in api/main.py + dashboard.html Engine Room)
- Day 18:    tests/test_integration.py (pytest suite, 30 tests)
- Day 19:    README.md, PROGRESS.md, docs/demo_script.py (3-minute timestamps)
- Day 20:    render.yaml, run_all.ps1, ready for final GitHub push and demo recording

## Critical constants (never change without reading this file)
- derive_weights(pd.DataFrame)       -- NOT compute_weights
- score_lead() output key "score"    -- NOT "total_score"
- score_lead() output key "confidence" -- NOT "confidence_band"
- network_proximity is a DICT       -- access .hops, .display
- graph node attr: "type"           -- NOT "node_type"
- graph edge attr: "rel"            -- NOT "relationship"
- run_backtest(train_df,test_df,signals_list) -- 3 args, all correct types
- INTENT_SIGNAL_TYPES = credit_rating_change, expansion_announcement,
                        leadership_change, ma_activity, equipment_purchase
- LLM never touches scores (Critical Rule 1)
- Confidence is display modifier only (Critical Rule 2)

## Repo
https://github.com/durbhavarun/salesiq
Local: C:\Users\durbh\Documents\salesiq

## Next action
Run the master script from C:\Users\durbh\Documents\salesiq with venv active:
  .\venv\Scripts\Activate.ps1
  powershell -ExecutionPolicy Bypass -File "C:\Users\durbh\.bob\playground\salesiq\run_all.ps1"

After all tests pass and GitHub push completes:
  uvicorn api.main:app --reload
  Open http://localhost:8000 and follow docs/demo_script.py for the 3-minute recording.
