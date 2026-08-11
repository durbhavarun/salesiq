"""
tests/test_integration.py
──────────────────────────
Full integration test suite — Days 18 coverage.
Runs the entire pipeline: data → weights → score → anomaly → API → override → audit → backtest.
Run with: pytest tests/test_integration.py -v
"""
import sys, os, json, dataclasses
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ["SALESIQ_LLM_PROVIDER"] = "none"
os.environ["SALESIQ_GRAPH_BACKEND"] = "memory"

import pandas as pd
import pytest
from fastapi.testclient import TestClient

# ── Fixtures ──────────────────────────────────────────────────
@pytest.fixture(scope="session")
def client():
    from api.main import app
    return TestClient(app)

@pytest.fixture(scope="session")
def deals_df():
    return pd.read_csv("data/demo/historical_deals.csv")

@pytest.fixture(scope="session")
def leads_df():
    return pd.read_csv("data/demo/new_leads.csv")

@pytest.fixture(scope="session")
def signals():
    return json.loads(Path("data/demo/signals.json").read_text())

@pytest.fixture(scope="session")
def weights(deals_df):
    from engine.icp_weights import derive_weights
    return derive_weights(deals_df)

# ── Day 7: ICP weights ────────────────────────────────────────
class TestICPWeights:
    def test_param_count(self):
        from engine.icp_weights import CANDIDATE_PARAMETERS
        assert len(CANDIDATE_PARAMETERS) == 21

    def test_derive_weights_returns_21_params(self, weights):
        assert isinstance(weights, dict)
        assert len(weights) == 21

    def test_banding_dscr(self):
        from engine.icp_weights import band_dscr
        assert band_dscr(1.65) == "strong"
        assert band_dscr(1.35) == "adequate"
        assert band_dscr(1.10) == "low"
        assert band_dscr(-0.5) == "invalid"

    def test_banding_debt(self):
        from engine.icp_weights import band_debt_to_ebitda
        assert band_debt_to_ebitda(2.0) == "low"
        assert band_debt_to_ebitda(4.0) == "moderate"
        assert band_debt_to_ebitda(6.0) == "high"

    def test_banding_years(self):
        from engine.icp_weights import band_years
        assert band_years(1.0) == "startup"
        assert band_years(5.0) == "established"
        assert band_years(15.0) == "mature"

    def test_intent_signal_types(self):
        from engine.icp_weights import INTENT_SIGNAL_TYPES
        assert set(INTENT_SIGNAL_TYPES) == {
            "credit_rating_change","expansion_announcement",
            "leadership_change","ma_activity","equipment_purchase"}

# ── Day 8: Scoring engine ─────────────────────────────────────
class TestScoringEngine:
    def test_score_lead_returns_real_keys(self, weights, signals):
        from engine.scoring_engine import score_lead
        lead = {"account_id":"acc_201","industry":"Manufacturing",
                "employee_band":"medium","loan_type":"equipment_financing",
                "dscr_band":"strong","company_stage":"established"}
        r = score_lead(lead, signals, weights, None)
        # real keys
        assert "score" in r
        assert "confidence" in r
        assert "network_proximity" in r
        assert isinstance(r["network_proximity"], dict)
        assert "hops" in r["network_proximity"]
        assert "display" in r["network_proximity"]
        # stale invented keys must NOT exist
        assert "total_score" not in r
        assert "confidence_band" not in r

    def test_raw_score_formula(self, weights, signals):
        from engine.scoring_engine import score_lead
        lead = {"account_id":"acc_999_test","industry":"manufacturing"}
        r = score_lead(lead, signals, weights, None)
        expected = round(0.70*r["icp_fit"] + 0.30*r["intent_score"], 2)
        assert abs(r["raw_score"] - expected) < 0.1

    def test_confidence_is_valid(self, weights, signals):
        from engine.scoring_engine import score_lead
        lead = {"account_id":"acc_201"}
        r = score_lead(lead, signals, weights, None)
        assert r["confidence"] in ("HIGH","MEDIUM","LOW")

    def test_recommended_action_mapping(self, weights, signals):
        from engine.scoring_engine import score_lead
        lead = {"account_id":"acc_201"}
        r = score_lead(lead, signals, weights, None)
        sc = r["score"]
        expected = "Prioritize" if sc>=75 else "Nurture" if sc>=50 else "Deprioritize"
        assert r["recommended_action"] == expected

    def test_determinism(self, weights, signals):
        from engine.scoring_engine import score_lead
        lead = {"account_id":"acc_205","industry":"manufacturing","dscr_band":"strong"}
        r1 = score_lead(lead, signals, weights, None)
        r2 = score_lead(lead, signals, weights, None)
        assert r1["score"] == r2["score"]
        assert r1["confidence"] == r2["confidence"]

# ── Day 9: Anomaly detector ───────────────────────────────────
class TestAnomalyDetector:
    def test_error_anomalies_block(self, leads_df, signals):
        from engine.anomaly_detector import detect_anomalies, has_blocking_errors
        sigs_by = {}
        for s in signals:
            sigs_by.setdefault(s["account_id"],[]).append(s)
        errors = 0
        for _,row in leads_df.iterrows():
            anoms = detect_anomalies(row.to_dict(), sigs_by.get(row["account_id"],[]))
            if has_blocking_errors(anoms):
                errors += 1
        assert errors >= 3

    def test_only_acc222_stale(self, leads_df, signals):
        from engine.anomaly_detector import detect_anomalies
        sigs_by = {}
        for s in signals:
            sigs_by.setdefault(s["account_id"],[]).append(s)
        stale = []
        for _,row in leads_df.iterrows():
            anoms = detect_anomalies(row.to_dict(), sigs_by.get(row["account_id"],[]))
            if any(a.rule=="stale_signals" for a in anoms):
                stale.append(row["account_id"])
        assert stale == ["acc_222"]

    def test_dscr_negative_is_error(self, signals):
        from engine.anomaly_detector import detect_anomalies, has_blocking_errors
        lead = {"account_id":"acc_test_neg","dscr":-0.5}
        anoms = detect_anomalies(lead, [])
        assert has_blocking_errors(anoms)

# ── Day 10: Backtest ──────────────────────────────────────────
class TestBacktest:
    def test_backtest_runs(self, deals_df, signals):
        from engine.backtest.split import date_based_split, run_backtest
        train, test, cutoff = date_based_split(deals_df)
        bt = run_backtest(train, test, signals)
        d  = dataclasses.asdict(bt)
        assert d["train_size"] > 0
        assert d["test_size"] > 0
        assert "spearman_correlation" in d
        assert "precision_at_10" in d
        assert isinstance(d["precision_at_10"], float)

# ── Day 11: LLM Gateway ───────────────────────────────────────
class TestLLMGateway:
    def test_provider_none_returns_structured_error(self):
        from engine.llm_gateway import extract_signals
        r = extract_signals("Company reported strong DSCR of 1.65.")
        assert r["provider"] == "none"
        assert r["valid"] == False
        assert r["error"] is not None
        assert isinstance(r["signals"], list)

    def test_pii_scrub(self):
        from engine.llm_gateway import _scrub_pii
        s = _scrub_pii("Email john@acme.com call 555-867-5309")
        assert "[EMAIL]" in s
        assert "[PHONE]" in s
        assert "john@acme.com" not in s

    def test_cache_hit(self):
        from engine.llm_gateway import extract_signals, _CACHE, _cache_key, _EXTRACTION_PREFIX
        import hashlib
        txt = "CACHE_TEST_UNIQUE_ABC_999"
        key = hashlib.sha256((_EXTRACTION_PREFIX+txt).encode()).hexdigest()
        _CACHE[key] = {"signals":[],"valid":True,"error":None}
        r = extract_signals(txt)
        assert r["provider"] == "cache"
        assert r["cache_hit"] == True

    def test_valid_signal_types_match_intent_signal_types(self):
        from engine.llm_gateway import _VALID_SIGNAL_TYPES
        from engine.icp_weights import INTENT_SIGNAL_TYPES
        assert _VALID_SIGNAL_TYPES == set(INTENT_SIGNAL_TYPES)

    def test_schema_rejects_bad_signal_type(self):
        from engine.llm_gateway import _validate_schema
        ok, reason = _validate_schema({"signals":[{"signal_type":"BAD_TYPE",
            "signal_value":"x","extraction_confidence":0.8,"evidence_snippet":"x"}]})
        assert not ok

    def test_graphrag_provider_none(self):
        from engine.llm_gateway import generate_graphrag_answer
        r = generate_graphrag_answer("Why?", "some context")
        assert "answer" in r
        assert "evidence_used" in r
        assert r["provider"] == "none"

# ── Day 12: GraphRAG Router ───────────────────────────────────
class TestGraphRAGRouter:
    def test_explain_template(self):
        from engine.graphrag.router import classify_question
        t, p = classify_question("Why did acc_201 score 78?")
        assert t == "explain"
        assert p["lead_id"] == "acc_201"

    def test_pattern_template(self):
        from engine.graphrag.router import classify_question
        t, p = classify_question("Have we won accounts like acc_205 before?")
        assert t == "pattern"

    def test_compare_template(self):
        from engine.graphrag.router import classify_question
        t, p = classify_question("Why is acc_201 scored higher than acc_202?")
        assert t == "compare"
        assert p["lead_id_a"] == "acc_201"
        assert p["lead_id_b"] == "acc_202"

    def test_fallback_to_explain(self):
        from engine.graphrag.router import classify_question
        t, _ = classify_question("Tell me about acc_210")
        assert t == "explain"

    def test_unknown(self):
        from engine.graphrag.router import classify_question
        t, _ = classify_question("What is the weather today?")
        assert t == "unknown"

# ── Day 12: GraphRAG Templates ────────────────────────────────
class TestGraphRAGTemplates:
    def test_run_explain_no_crash(self):
        from engine.graphrag.templates import run_explain
        scores = {"acc_201":{"score":78,"icp_fit":82,"intent_score":65,
                              "confidence":"HIGH","recommended_action":"Prioritize",
                              "network_proximity":{"hops":None,"display":"None detected within 2 hops"},
                              "positive_signals":[],"gaps":[]}}
        r = run_explain("acc_201", scores)
        assert "answer" in r and "paths_used" in r and "nodes_cited" in r
        assert r["provider"] == "none"

    def test_run_explain_empty_lead_id(self):
        from engine.graphrag.templates import run_explain
        r = run_explain("", {})
        assert r.get("error") is not None

    def test_run_pattern_no_crash(self):
        from engine.graphrag.templates import run_pattern
        r = run_pattern("acc_201", {})
        assert "answer" in r and "paths_used" in r

    def test_run_compare_no_crash(self):
        from engine.graphrag.templates import run_compare
        r = run_compare("acc_201", "acc_202", {})
        assert "answer" in r and "paths_used" in r

# ── Day 13: API endpoints ─────────────────────────────────────
class TestAPI:
    def test_root_returns_html(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]

    def test_score_all(self, client):
        r = client.post("/score/all")
        assert r.status_code == 200
        body = r.json()
        assert "leads" in body and body["count"] > 0
        # sorted descending by "score" key
        scored = [l for l in body["leads"] if l.get("score") is not None]
        scores = [l["score"] for l in scored]
        assert scores == sorted(scores, reverse=True)

    def test_score_keys_correct(self, client):
        r = client.post("/score/all")
        leads = r.json()["leads"]
        scored = [l for l in leads if l.get("score") is not None]
        assert len(scored) > 0
        l = scored[0]
        assert "score" in l
        assert "confidence" in l
        assert "network_proximity" in l
        assert isinstance(l["network_proximity"], dict)
        assert "total_score" not in l      # must not exist
        assert "confidence_band" not in l  # must not exist

    def test_get_score_known_lead(self, client):
        # score all first
        r = client.post("/score/all")
        first_id = r.json()["leads"][0]["account_id"]
        r2 = client.get(f"/score/{first_id}")
        assert r2.status_code == 200
        assert "score" in r2.json()

    def test_get_score_unknown_returns_404(self, client):
        r = client.get("/score/acc_NONEXISTENT_XYZ")
        assert r.status_code == 404

    def test_score_single_valid(self, client):
        payload = {"account_id":"acc_manual_001","company_name":"Test Corp",
                   "industry":"manufacturing","employee_band":"medium",
                   "business_model":"b2b","geography":"north_america",
                   "company_stage":"established","loan_type":"equipment_financing",
                   "loan_amount_band":"medium","collateral_type":"equipment",
                   "existing_banking_relationship":"yes",
                   "dscr":1.65,"debt_to_ebitda_ratio":2.8,"years_in_business":12.0,
                   "referred_by_portfolio":"no","shares_guarantor_with_won":"no"}
        r = client.post("/score/single", json=payload)
        assert r.status_code == 200
        s = r.json()
        assert "score" in s or "error" in s

    def test_score_single_bad_dscr_blocked(self, client):
        payload = {"account_id":"acc_bad_dscr","dscr":-0.5,
                   "company_name":"Bad","industry":"manufacturing",
                   "employee_band":"medium","business_model":"b2b",
                   "geography":"north_america","company_stage":"established",
                   "loan_type":"equipment_financing","loan_amount_band":"medium",
                   "collateral_type":"equipment","existing_banking_relationship":"yes",
                   "referred_by_portfolio":"no","shares_guarantor_with_won":"no"}
        r = client.post("/score/single", json=payload)
        assert r.status_code == 200
        blocked = r.json()
        assert blocked.get("score") is None
        assert blocked.get("error") is not None

    def test_graph_lead_orbit(self, client):
        r = client.post("/score/all")
        first_id = r.json()["leads"][0]["account_id"]
        r2 = client.get(f"/graph/lead/{first_id}")
        assert r2.status_code == 200
        orbit = r2.json()
        assert "nodes" in orbit and "edges" in orbit and "empty_state" in orbit
        assert any(n["id"]==first_id for n in orbit["nodes"])

    def test_graph_all(self, client):
        r = client.get("/graph/all")
        assert r.status_code == 200
        assert "total_nodes" in r.json()

    def test_backtest(self, client):
        r = client.post("/backtest/run")
        assert r.status_code == 200
        bt = r.json()
        assert "metrics" in bt
        assert "spearman_r" in bt["metrics"]
        assert "precision_at_10" in bt["metrics"]

    def test_dashboard_summary(self, client):
        client.post("/score/all")  # ensure scores exist
        r = client.get("/dashboard/summary")
        assert r.status_code == 200
        s = r.json()
        for k in ["total_leads","avg_score","optimal","partial","unfit","high_confidence"]:
            assert k in s
        assert s["total_leads"] > 0

    def test_icp_lifecycle(self, client):
        # submit
        r = client.post("/icp/submit", json={"reason":"test"})
        assert r.status_code == 200
        assert r.json()["status"] == "PENDING_APPROVAL"
        # approve
        r = client.post("/icp/approve")
        assert r.status_code == 200
        assert r.json()["status"] == "ACTIVE"
        # submit again then reject
        r = client.post("/icp/submit", json={"reason":"reject test"})
        assert r.status_code == 200
        r = client.post("/icp/reject", json={"reason":"test"})
        assert r.status_code == 200
        assert r.json()["status"] == "DRAFT"

    def test_override_valid(self, client):
        r = client.post("/score/all")
        first_id = r.json()["leads"][0]["account_id"]
        r2 = client.post("/override", json={"score_id":first_id,
            "decision":"approve","reason":"Strong DSCR","rep_id":"rep_001"})
        assert r2.status_code == 200
        assert r2.json()["status"] == "recorded"

    def test_override_invalid_decision(self, client):
        r = client.post("/score/all")
        first_id = r.json()["leads"][0]["account_id"]
        r2 = client.post("/override", json={"score_id":first_id,
            "decision":"WRONG","reason":"test","rep_id":"rep_001"})
        assert r2.status_code == 400

    def test_audit_log_populated(self, client):
        r = client.get("/audit")
        assert r.status_code == 200
        assert r.json()["count"] > 0

    def test_docs_reachable(self, client):
        r = client.get("/docs")
        assert r.status_code == 200

    def test_acc225_orbit_empty_state(self, client):
        r = client.get("/graph/lead/acc_225")
        assert r.status_code == 200
        assert r.json()["empty_state"] == True
        assert "No network connections" in r.json()["message"]

    def test_nonexistent_lead_orbit_no_crash(self, client):
        r = client.get("/graph/lead/acc_NONEXISTENT_XYZ")
        assert r.status_code == 200
        assert r.json()["empty_state"] == True
