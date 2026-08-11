"""
Demo script — 3 minutes, strict timestamps.
For Day 20 recording. Read this top to bottom before hitting record.

SETUP (do this BEFORE recording):
  1. cd C:\Users\durbh\Documents\salesiq
  2. .\venv\Scripts\Activate.ps1
  3. set SALESIQ_LLM_PROVIDER=none   (so demo doesn't wait on Groq API)
  4. python graph/populate.py         (rebuilds graph.pkl)
  5. uvicorn api.main:app --reload    (in a separate terminal)
  6. Open http://localhost:8000 in browser
  7. Run POST /score/all once via the dashboard before recording
  8. Resize browser to full screen

=======================================================
DEMO SCRIPT — 3 MINUTES EXACTLY
=======================================================

[0:00-0:20] PROBLEM STATEMENT (voice only, show homepage KPIs loading)

"B2B lending teams pursue accounts that won't close — because their
Ideal Customer Profile is informal, not connected to actual outcomes.
Existing scoring tools predict, but they're black boxes.
SalesIQ fixes that: every score is traceable to specific evidence."

[0:20-0:40] HOMEPAGE — show KPIs populating

Click: "Score All Leads" button
Point at: Total Leads, Optimal Fit count, Pipeline Value estimate
"25 leads scored in under a second. 35% win rate baseline in the data."
Show: Score Distribution doughnut chart
"Green = Optimal Fit scoring ≥75. The model surfaces these from the noise."

[0:40-1:10] LEADS TABLE — click first Optimal Fit lead to expand

Point at: Score badge, Confidence badge (HIGH), Network Proximity
"Score 78. HIGH confidence — 4+ matched signals, avg confidence 0.80+.
 Not just a number — here's why."
Expand score card → show evidence trail table:
  Factor | Evidence | Source
Point at: "manufacturing industry → seen in 12 won deals → ICP v1"
Point at: "DSCR strong band → seen in 18 won deals → ICP v1"
Point at gaps: "No existing banking relationship — this is a gap."
"Every positive signal and every gap is grounded in a specific deal or signal ID.
 Nothing is a bare number."

[1:10-1:30] NETWORK PROXIMITY — click "View Graph" button

Orbit view loads in vis-network
Point at: Lead node (blue, center)
Point at: Signal nodes (orange) — "credit rating change, expansion signal"
If prox edge exists: point at grey account → green won deal path
"This lead shares a guarantor with a won deal from 2024.
 That 1-hop path is surfaced automatically — a rep would never find this manually."

[1:30-1:55] GRAPHRAG — go to Knowledge Graph tab → Ask the Graph

Type: "Why did acc_201 score 78?"
Hit Ask
Show answer appears with paths_used listed below
"The system retrieved the actual graph paths — not a similarity vector.
 Every claim in this answer has a source you can check."

[1:55-2:15] ENGINE ROOM — show ICP controls

Click Engine Room tab
Show weights table — parameters sorted by abs(weight)
"21 parameters, Laplace-smoothed. Any parameter with fewer than 3
 supporting deals in each outcome class gets weight zero — no spurious patterns."
Click: "Submit for Approval" → "Approve"
Show status chip: DRAFT → PENDING_APPROVAL → ACTIVE
"Human approval required. No silent auto-retraining."

[2:15-2:35] BACKTEST — click Run Backtest

Show: Spearman r, Precision@10, cutoff date
"Date-based split — train on past, test on future. No temporal leakage.
 Precision@10: 6 of our top 10 scored leads actually won.
 Baseline win rate is 35%. The model is surfacing real signal."

[2:35-2:50] AUDIT LOG

Click Audit Log tab
Show entries: score_all, override, icp_submit, icp_approve, backtest_run
"Every action is recorded. Immutable trail. Rep overrides are
 logged as feedback — never used as training labels."

[2:50-3:00] CLOSE

"SalesIQ replaces tribal sales knowledge with a governed, auditable process.
 Every score traces to evidence. Every decision stays with the human.
 IBM AI Builders Challenge — Wildcard Track. Thank you."

=======================================================
POST-RECORDING CHECKLIST
=======================================================
[ ] Video is exactly 3:00 or under
[ ] /docs shown briefly (proves 17 endpoints to judges)
[ ] Score card evidence trail clearly visible
[ ] GraphRAG paths_used shown
[ ] ICP approve lifecycle shown
[ ] Backtest Precision@10 mentioned (not Spearman)
[ ] Audit log shown
[ ] No API errors visible in browser console
"""
