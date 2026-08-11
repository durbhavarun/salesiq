"""
api/main.py
───────────
FastAPI backend — 17 endpoints.

Verified against actual engine files:
  derive_weights(pd.DataFrame)          from engine/icp_weights.py
  score_lead() returns "score","confidence","network_proximity"(dict)
  run_backtest(train_df,test_df,signals) from engine/backtest/split.py
  graph nodes: "type" attr; edges: "rel" attr  from graph/populate.py
  INTENT_SIGNAL_TYPES = credit_rating_change, expansion_announcement,
                        leadership_change, ma_activity, equipment_purchase
"""
from __future__ import annotations
import dataclasses, io, json, logging, os, pickle, uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd
from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from engine.icp_weights import (
    derive_weights, band_dscr, band_debt_to_ebitda, band_years,
    SCORE_OPTIMAL, SCORE_PARTIAL,
)
from engine.scoring_engine import score_lead
from engine.anomaly_detector import detect_anomalies, has_blocking_errors
from engine.backtest.split import date_based_split, run_backtest

logger = logging.getLogger(__name__)

BASE      = Path(__file__).parent.parent
DATA      = BASE / "data" / "demo"
TEMPLATES = BASE / "frontend" / "templates"
STATE     = BASE / "state"
STATE.mkdir(exist_ok=True)

_ICP_FILE    = STATE / "icp_state.json"
_SCORES_FILE = STATE / "scores_state.json"
_AUDIT_FILE  = STATE / "audit_log.json"


def _load_json(path: Path, default: Any) -> Any:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return default


def _save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


app = FastAPI(
    title="SalesIQ Corporate Lending",
    description="Explainable graph-native loan scoring. Every score traces to evidence.",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

_icp:    dict       = {}
_scores: dict       = {}
_audit:  list       = []
_G:      nx.DiGraph = nx.DiGraph()


def _load_graph() -> nx.DiGraph:
    pkl = DATA / "graph.pkl"
    if pkl.exists():
        with open(pkl, "rb") as f:
            return pickle.load(f)
    logger.warning("graph.pkl missing — run graph/populate.py first")
    return nx.DiGraph()


@app.on_event("startup")
def _startup() -> None:
    global _icp, _scores, _audit, _G
    _icp    = _load_json(_ICP_FILE, {"status": "DRAFT", "current_version": None,
                                      "versions": [], "pending": None})
    _scores = _load_json(_SCORES_FILE, {})
    _audit  = _load_json(_AUDIT_FILE, [])
    _G      = _load_graph()
    logger.info("Started. ICP=%s scores=%d", _icp.get("status"), len(_scores))


def _audit_log(action: str, detail: dict) -> None:
    _audit.append({"id": str(uuid.uuid4()),
                   "timestamp": datetime.now(timezone.utc).isoformat(),
                   "action": action, **detail})
    _save_json(_AUDIT_FILE, _audit)


# ── Float banding at API boundary ─────────────────────────────
def _band(lead: dict) -> dict:
    lead = dict(lead)
    for field, fn, bkey in [
        ("dscr",                 band_dscr,           "dscr_band"),
        ("debt_to_ebitda_ratio", band_debt_to_ebitda, "debt_to_ebitda_band"),
        ("years_in_business",    band_years,           "years_band"),
    ]:
        val = lead.get(field)
        if val is not None:
            try:
                lead[bkey] = fn(float(val))
            except (ValueError, TypeError):
                pass
    return lead


# ── Data loaders ──────────────────────────────────────────────
def _deals_df() -> pd.DataFrame:
    f = DATA / "historical_deals.csv"
    if not f.exists():
        raise HTTPException(500, "historical_deals.csv not found")
    return pd.read_csv(f)


def _signals_list() -> list:
    f = DATA / "signals.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else []


def _weights() -> dict:
    return derive_weights(_deals_df())


# ── Pydantic models ───────────────────────────────────────────
class LeadIn(BaseModel):
    account_id:                    str
    company_name:                  str   = ""
    industry:                      str   = ""
    employee_band:                 str   = ""
    business_model:                str   = ""
    geography:                     str   = ""
    company_stage:                 str   = ""
    loan_type:                     str   = ""
    loan_amount_band:              str   = ""
    collateral_type:               str   = ""
    existing_banking_relationship: str   = ""
    dscr:                          float | None = None
    debt_to_ebitda_ratio:          float | None = None
    years_in_business:             float | None = None
    referred_by_portfolio:         str   = "no"
    shares_guarantor_with_won:     str   = "no"


class OverrideIn(BaseModel):
    score_id: str
    decision: str = Field(..., description="approve | flag | deprioritize")
    reason:   str
    rep_id:   str = "anonymous"


class ExplainIn(BaseModel):
    question: str
    lead_id:  str = ""


class ICPSubmitIn(BaseModel):
    reason: str = "Manual weight submission"


# ── Core pipeline ─────────────────────────────────────────────
def _run_pipeline(lead_dict: dict, weights: dict, signals: list) -> dict:
    """
    Banding → anomaly detection → scoring.
    Identical code path for every lead. (Critical Rule 10)
    Uses real keys from scoring_engine.py: "score", "confidence", "network_proximity"
    """
    lead = _band(lead_dict)
    acc  = lead.get("account_id", "unknown")
    anomalies = detect_anomalies(lead, signals)

    if has_blocking_errors(anomalies):
        return {
            "account_id":   acc,
            "company_name": lead.get("company_name", acc),
            "error":        "Scoring blocked — ERROR anomaly detected",
            "anomalies":    [dataclasses.asdict(a) for a in anomalies],
            "score":        None,
            "anomaly_flag": True,
        }

    result = score_lead(lead, signals, weights, _G)

    # Rule 7: post-scoring flip check (requires icp_fit + intent_score)
    icp_fit  = result.get("icp_fit", 0) or 0
    intent   = result.get("intent_score", 0) or 0
    score_v  = result.get("score", 0) or 0
    expected = round(0.70 * icp_fit + 0.30 * intent, 1)
    if abs(score_v - expected) > 5:
        from engine.anomaly_detector import AnomalyResult, SEVERITY_WARNING
        anomalies.append(AnomalyResult(
            rule="score_flip_risk", severity=SEVERITY_WARNING,
            message=f"score={score_v} vs 0.7×{icp_fit}+0.3×{intent}={expected}",
            affected_field="score"))

    result["anomalies"]    = [dataclasses.asdict(a) for a in anomalies] if anomalies else []
    result["anomaly_flag"] = bool(anomalies)
    return result


# ── Helpers for real key names ────────────────────────────────
def _sc(r: dict) -> float | None:
    """Extract score (real key: 'score')."""
    return r.get("score")

def _conf(r: dict) -> str:
    """Extract confidence (real key: 'confidence')."""
    return r.get("confidence", "LOW")

def _prox_hops(r: dict) -> int | None:
    """Extract network proximity hop count from nested dict."""
    p = r.get("network_proximity")
    return p.get("hops") if isinstance(p, dict) else None


# ── ENDPOINT 1: Dashboard HTML ────────────────────────────────
@app.get("/", response_class=HTMLResponse, tags=["Dashboard"])
def serve_dashboard() -> HTMLResponse:
    html = TEMPLATES / "dashboard.html"
    if html.exists():
        return HTMLResponse(html.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>SalesIQ</h1><p>Run Day 14 to build the dashboard.</p>")


# ── SCORING ───────────────────────────────────────────────────
@app.get("/score/{lead_id}", tags=["Scoring"])
def get_score(lead_id: str):
    if lead_id not in _scores:
        raise HTTPException(404, f"No score for {lead_id}. Run POST /score/all first.")
    return _scores[lead_id]


@app.post("/score/all", tags=["Scoring"])
def score_all():
    f = DATA / "new_leads.csv"
    if not f.exists():
        raise HTTPException(500, "new_leads.csv not found")
    w = _weights()
    s = _signals_list()
    results = []
    for _, row in pd.read_csv(f).iterrows():
        r = _run_pipeline(row.to_dict(), w, s)
        _scores[row["account_id"]] = r
        results.append(r)
    _save_json(_SCORES_FILE, _scores)
    _audit_log("score_all", {"lead_count": len(results)})
    results.sort(key=lambda r: _sc(r) or -1, reverse=True)
    return {"leads": results, "count": len(results)}


@app.post("/score/single", tags=["Scoring"])
def score_single(lead: LeadIn):
    """[Decision 2] Identical pipeline to bulk — no simplified code path. (Rule 10)"""
    r = _run_pipeline(lead.model_dump(), _weights(), _signals_list())
    _scores[lead.account_id] = r
    _save_json(_SCORES_FILE, _scores)
    _audit_log("score_single", {"account_id": lead.account_id})
    return r


# ── GRAPH ─────────────────────────────────────────────────────
def _orbit(lead_id: str) -> dict:
    """READ-ONLY subgraph for vis-network. Never writes. (Rule 11)
    Uses real graph attrs: node 'type', edge 'rel'"""
    C = {"lead":"#4A90D9","signal":"#F5A623","won":"#7ED321",
         "lost":"#D0021B","account":"#9B9B9B"}
    vis_nodes: list[dict] = []
    vis_edges: list[dict] = []
    seen: set[str]        = set()

    def add(nid: str, color: str, label: str, title: str = "") -> bool:
        if nid in seen or len(seen) >= 20:
            return False
        seen.add(nid)
        vis_nodes.append({"id": nid, "label": label[:22],
                          "color": color, "title": title or nid})
        return True

    if lead_id not in _G.nodes:
        add(lead_id, C["lead"], lead_id, "Lead (not in graph)")
        return {"nodes": vis_nodes, "edges": vis_edges, "empty_state": True,
                "message": "No network connections detected within 2 hops"}

    la = _G.nodes[lead_id]
    sc = _scores.get(lead_id, {})
    add(lead_id, C["lead"], la.get("company_name", lead_id)[:22],
        f"Lead | score={_sc(sc)}")

    for _, dst, data in _G.edges(lead_id, data=True):
        rel = data.get("rel", "")       # real attr: "rel"
        if rel == "HAS_SIGNAL":
            sa = _G.nodes.get(dst, {})
            if add(dst, C["signal"], sa.get("signal_type", "signal")[:18],
                   f"conf={sa.get('extraction_confidence','')}"):
                vis_edges.append({"from": lead_id, "to": dst, "label": "signal",
                                  "arrows": "to", "color": {"color": C["signal"]}})
        elif rel in ("REFERRED_BY", "SHARES_GUARANTOR"):
            lbl = {"REFERRED_BY": "referred by", "SHARES_GUARANTOR": "guarantor"}[rel]
            ca = _G.nodes.get(dst, {})
            if add(dst, C["account"], ca.get("company_name", dst)[:22], f"Connected | {rel}"):
                vis_edges.append({"from": lead_id, "to": dst, "label": lbl,
                                  "arrows": "to", "dashes": True,
                                  "color": {"color": C["account"]}})
                for _, ddst, dd in _G.edges(dst, data=True):
                    if dd.get("rel") == "HAS_DEAL":
                        da  = _G.nodes.get(ddst, {})
                        won = da.get("outcome") == "won"
                        col = C["won"] if won else C["lost"]
                        dlbl = ("Won" if won else "Lost") + f"\n{str(da.get('deal_closed_date',''))[:7]}"
                        if add(ddst, col, dlbl, str(da)):
                            vis_edges.append({
                                "from": dst, "to": ddst,
                                "label": "won deal" if won else "lost deal",
                                "arrows": "to", "color": {"color": col}})

    empty = len(vis_nodes) <= 1
    return {"nodes": vis_nodes, "edges": vis_edges, "empty_state": empty,
            "message": "No network connections detected within 2 hops" if empty else ""}


@app.get("/graph/lead/{lead_id}", tags=["Graph"])
def graph_lead(lead_id: str):
    """[Decision 3] READ-ONLY orbit subgraph for vis-network. (Rule 11)"""
    return _orbit(lead_id)


@app.get("/graph/all", tags=["Graph"])
def graph_all():
    """Full knowledge graph capped at 200 nodes — Knowledge Brain tab."""
    COLOR = {"Account":"#4A90D9","Deal":"#7ED321","Signal":"#F5A623","ICPVersion":"#9B59B6"}
    nodes = [{"id": nid, "label": nid[:20],
              "color": COLOR.get(a.get("type",""), "#AAA"),   # real attr: "type"
              "title": a.get("type","?")}
             for nid, a in list(_G.nodes(data=True))[:200]]
    seen = {n["id"] for n in nodes}
    edges = [{"from": s, "to": d, "label": data.get("rel",""),   # real attr: "rel"
               "arrows": "to"}
             for s, d, data in _G.edges(data=True) if s in seen and d in seen]
    return {"nodes": nodes, "edges": edges, "total_nodes": _G.number_of_nodes()}


# ── GRAPHRAG ──────────────────────────────────────────────────
@app.post("/explain", tags=["GraphRAG"])
def explain(body: ExplainIn):
    from engine.graphrag.router import classify_question
    from engine.graphrag.templates import run_explain, run_pattern, run_compare
    template, params = classify_question(body.question)
    if body.lead_id and not params.get("lead_id"):
        params["lead_id"] = body.lead_id
    if template == "explain":
        return run_explain(params.get("lead_id",""), _scores, body.question)
    elif template == "pattern":
        return run_pattern(params.get("lead_id",""), _scores, body.question)
    elif template == "compare":
        return run_compare(params.get("lead_id_a",""), params.get("lead_id_b",""),
                           _scores, body.question)
    raise HTTPException(400,
        "Question not recognised. Try: 'Why did acc_201 score 78?' / "
        "'Have we won accounts like acc_201?' / 'Why is acc_201 higher than acc_202?'")


# ── OVERRIDE ──────────────────────────────────────────────────
@app.post("/override", tags=["Override"])
def override(ov: OverrideIn):
    if ov.decision not in ("approve","flag","deprioritize"):
        raise HTTPException(400, f"Invalid decision '{ov.decision}'")
    if ov.score_id not in _scores:
        raise HTTPException(404, f"No score for {ov.score_id}")
    rec = {"override_id": str(uuid.uuid4()), "score_id": ov.score_id,
           "decision": ov.decision, "reason": ov.reason, "rep_id": ov.rep_id,
           "created_at": datetime.now(timezone.utc).isoformat()}
    _scores[ov.score_id].setdefault("overrides", []).append(rec)
    _save_json(_SCORES_FILE, _scores)
    _audit_log("override", {"score_id": ov.score_id, "decision": ov.decision,
                             "rep_id": ov.rep_id})
    return {"status": "recorded", "override": rec}


# ── AUDIT ─────────────────────────────────────────────────────
@app.get("/audit", tags=["Audit"])
def get_audit():
    return {"entries": _audit, "count": len(_audit)}


# ── BACKTEST ──────────────────────────────────────────────────
@app.post("/backtest/run", tags=["Backtest"])
def backtest():
    """
    Correct call sequence:
      date_based_split(deals_df) → train_df, test_df, cutoff
      run_backtest(train_df, test_df, signals_list) → BacktestResult dataclass
      dataclasses.asdict() for JSON serialisation
    """
    df  = _deals_df()
    sig = _signals_list()
    train_df, test_df, cutoff = date_based_split(df)
    bt  = run_backtest(train_df, test_df, sig)
    d   = dataclasses.asdict(bt)
    _audit_log("backtest_run", {"spearman_r": d["spearman_correlation"],
                                 "precision_at_10": d["precision_at_10"]})
    return {
        "metrics": {
            "spearman_r":       d["spearman_correlation"],
            "spearman_pvalue":  d["spearman_pvalue"],
            "precision_at_10":  d["precision_at_10"],
            "train_size":       d["train_size"],
            "test_size":        d["test_size"],
            "won_in_test":      d["won_in_test"],
            "baseline_win_rate": round(d["won_in_test"]/d["test_size"],3)
                                  if d["test_size"] else 0,
        },
        "cutoff_date": cutoff,
        "disclaimer":  d["disclaimer"],
    }


# ── DASHBOARD STATS ───────────────────────────────────────────
@app.get("/dashboard/summary", tags=["Dashboard"])
def summary():
    scored = [v for v in _scores.values() if _sc(v) is not None]
    if not scored:
        return {"total_leads":0,"avg_score":0,"optimal":0,"partial":0,"unfit":0,
                "high_confidence":0,"medium_confidence":0,"low_confidence":0,
                "pipeline_value_estimate":0}
    total = len(scored)
    avg   = round(sum(_sc(s) for s in scored) / total, 1)
    return {
        "total_leads":   total,
        "avg_score":     avg,
        "optimal":  sum(1 for s in scored if _sc(s) >= SCORE_OPTIMAL),
        "partial":  sum(1 for s in scored if SCORE_PARTIAL <= _sc(s) < SCORE_OPTIMAL),
        "unfit":    sum(1 for s in scored if _sc(s) < SCORE_PARTIAL),
        "high_confidence":   sum(1 for s in scored if _conf(s) == "HIGH"),
        "medium_confidence": sum(1 for s in scored if _conf(s) == "MEDIUM"),
        "low_confidence":    sum(1 for s in scored if _conf(s) == "LOW"),
        "pipeline_value_estimate": sum(1 for s in scored
                                       if _sc(s) >= SCORE_OPTIMAL) * 2_500_000,
    }


@app.get("/dashboard/regions", tags=["Dashboard"])
def regions():
    by_r: dict[str, list[float]] = {}
    for v in _scores.values():
        sc = _sc(v)
        if sc is None:
            continue
        by_r.setdefault(v.get("geography","Unknown"), []).append(sc)
    return {r: {"count":len(s),"avg_score":round(sum(s)/len(s),1)}
            for r, s in by_r.items()}


# ── UPLOAD ────────────────────────────────────────────────────
@app.post("/upload/deals", tags=["Upload"])
async def upload_deals(file: UploadFile = File(...)):
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(400, f"Invalid CSV: {e}")
    if len(df) < 10:
        raise HTTPException(400, "CSV must have at least 10 rows")
    (DATA / "historical_deals.csv").write_bytes(content)
    _audit_log("upload_deals", {"rows":len(df),"filename":file.filename})
    return {"status":"uploaded","rows":len(df)}


@app.post("/upload/leads", tags=["Upload"])
async def upload_leads(file: UploadFile = File(...)):
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(400, f"Invalid CSV: {e}")
    (DATA / "new_leads.csv").write_bytes(content)
    _audit_log("upload_leads", {"rows":len(df),"filename":file.filename})
    return {"status":"uploaded","rows":len(df)}


# ── ICP LIFECYCLE ─────────────────────────────────────────────
@app.get("/icp/current", tags=["ICP"])
def icp_current():
    return {"status": _icp.get("status","DRAFT"),
            "current_version": _icp.get("current_version"),
            "weights": _weights(), "pending": _icp.get("pending")}


@app.post("/icp/submit", tags=["ICP"])
def icp_submit(body: ICPSubmitIn):
    if _icp.get("status") == "PENDING_APPROVAL":
        raise HTTPException(409, "ICP already pending — approve or reject first.")
    vid = f"v{len(_icp.get('versions',[]))+1}"
    pending = {"version_id": vid, "weights": _weights(),
               "submitted_at": datetime.now(timezone.utc).isoformat(),
               "reason": body.reason}
    _icp["status"]  = "PENDING_APPROVAL"
    _icp["pending"] = pending
    _save_json(_ICP_FILE, _icp)
    _audit_log("icp_submit", {"version_id": vid, "reason": body.reason})
    return {"status":"PENDING_APPROVAL","pending":pending}


@app.post("/icp/approve", tags=["ICP"])
def icp_approve():
    if _icp.get("status") != "PENDING_APPROVAL":
        raise HTTPException(409, "No ICP pending approval.")
    p = _icp["pending"]
    p["approved_at"] = datetime.now(timezone.utc).isoformat()
    _icp.setdefault("versions",[]).append(p)
    _icp["status"]          = "ACTIVE"
    _icp["current_version"] = p["version_id"]
    _icp["pending"]         = None
    _save_json(_ICP_FILE, _icp)
    _audit_log("icp_approve", {"version_id": p["version_id"]})
    return {"status":"ACTIVE","approved_version":p["version_id"]}


@app.post("/icp/reject", tags=["ICP"])
def icp_reject(reason: str = Body("No reason given", embed=True)):
    if _icp.get("status") != "PENDING_APPROVAL":
        raise HTTPException(409, "No ICP pending approval.")
    rid = (_icp.get("pending") or {}).get("version_id")
    _icp["status"]  = "DRAFT"
    _icp["pending"] = None
    _save_json(_ICP_FILE, _icp)
    _audit_log("icp_reject", {"version_id": rid, "reason": reason})
    return {"status":"DRAFT","rejected_version":rid}
