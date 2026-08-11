"""
engine/anomaly_detector.py
==========================
7 validation rules run before score_lead() is called.

ERROR   → has_blocking_errors() returns True → API does not score the lead
WARNING → has_blocking_errors() returns False → scoring proceeds with flag

This file is called identically for bulk uploads AND manual /score/single entries.
There is no separate simplified validation path.
"""

from dataclasses import dataclass
from datetime import date
from typing import List
from engine.icp_weights import INTENT_SIGNAL_TYPES

SEVERITY_ERROR   = "ERROR"
SEVERITY_WARNING = "WARNING"


@dataclass
class AnomalyResult:
    rule:           str
    severity:       str
    message:        str
    affected_field: str


def has_blocking_errors(anomalies: List[AnomalyResult]) -> bool:
    """
    Returns True if any anomaly has severity ERROR.
    The API calls this after detect_anomalies() to decide
    whether to call score_lead() or return the error response.
    """
    return any(a.severity == SEVERITY_ERROR for a in anomalies)


def detect_anomalies(lead: dict, signals: list) -> List[AnomalyResult]:
    """
    Run all 7 rules. Returns list of AnomalyResult (empty = clean).

    Args:
        lead:    dict of lead fields (floats already banded at API boundary)
        signals: full list of all signals (filtered inside to this lead)
    """
    today      = date.today().isoformat()
    account_id = lead.get("account_id", "unknown")
    lead_sigs  = [s for s in signals if s.get("account_id") == account_id]
    anomalies: List[AnomalyResult] = []

    # ── Rule 1: Low extraction confidence (WARNING) ──────────────────
    for sig in lead_sigs:
        try:
            conf = float(sig.get("extraction_confidence", 1.0))
        except (TypeError, ValueError):
            conf = 0.0
        if conf < 0.30:
            anomalies.append(AnomalyResult(
                rule="low_extraction_confidence",
                severity=SEVERITY_WARNING,
                message=(
                    f"Signal {sig.get('signal_id','?')} has extraction_confidence={conf:.2f} "
                    f"(threshold: 0.30). Signal may be unreliable."
                ),
                affected_field="extraction_confidence",
            ))

    # ── Rule 2: Future eligibility date (ERROR) ──────────────────────
    for sig in lead_sigs:
        elig = str(sig.get("score_eligibility_date", "") or "")
        if elig and elig > today:
            anomalies.append(AnomalyResult(
                rule="future_eligibility_date",
                severity=SEVERITY_ERROR,
                message=(
                    f"Signal {sig.get('signal_id','?')} has "
                    f"score_eligibility_date={elig} which is in the future "
                    f"(today={today}). Temporal integrity violation."
                ),
                affected_field="score_eligibility_date",
            ))

    # ── Rule 3: Invalid signal type (ERROR) ──────────────────────────
    for sig in lead_sigs:
        sig_type = sig.get("signal_type", "")
        if sig_type and sig_type not in INTENT_SIGNAL_TYPES:
            anomalies.append(AnomalyResult(
                rule="invalid_signal_type",
                severity=SEVERITY_ERROR,
                message=(
                    f"Signal {sig.get('signal_id','?')} has signal_type='{sig_type}' "
                    f"which is not in the approved list: {INTENT_SIGNAL_TYPES}."
                ),
                affected_field="signal_type",
            ))

    # ── Rule 4: Cold start (WARNING) ─────────────────────────────────
    if len(lead_sigs) == 0:
        anomalies.append(AnomalyResult(
            rule="cold_start",
            severity=SEVERITY_WARNING,
            message=(
                f"Lead {account_id} has no signals. "
                f"Score uses ICP Fit only and will receive LOW confidence band."
            ),
            affected_field="signals",
        ))

    # ── Rule 5: Impossible financial value (ERROR) ───────────────────
    for field_name in ["dscr", "debt_to_ebitda_ratio"]:
        val = lead.get(field_name)
        if val is not None:
            try:
                if float(val) < 0:
                    anomalies.append(AnomalyResult(
                        rule="impossible_financial_value",
                        severity=SEVERITY_ERROR,
                        message=(
                            f"{field_name}={val} is negative (physically impossible). "
                            f"Cannot produce a meaningful score from invalid financial data."
                        ),
                        affected_field=field_name,
                    ))
            except (TypeError, ValueError):
                pass

    # ── Rule 6: Stale signals (WARNING) ──────────────────────────────
    if lead_sigs:
        try:
            newest = max(s.get("event_date", "1900-01-01") for s in lead_sigs)
            from datetime import date as dt
            days_old = (dt.fromisoformat(today) - dt.fromisoformat(newest)).days
            if days_old > 548:   # 18 months
                anomalies.append(AnomalyResult(
                    rule="stale_signals",
                    severity=SEVERITY_WARNING,
                    message=(
                        f"All signals for {account_id} are older than 18 months "
                        f"(newest: {newest}, {days_old} days ago)."
                    ),
                    affected_field="event_date",
                ))
        except (ValueError, TypeError):
            pass

    # ── Rule 7: Score flip risk (WARNING) ────────────────────────────
    # Applied post-scoring in the API layer (requires icp_fit + intent_score).
    # Recorded here as a placeholder so the API knows to check after scoring.
    # The API appends a Rule 7 AnomalyResult if abs(icp_fit - intent) > 40.

    return anomalies
