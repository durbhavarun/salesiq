"""
engine/scoring_engine.py
========================
Deterministic scoring engine.

Formula:
    ICP Fit  = sum of weights for each parameter present on the lead (0-100)
    Intent   = sum of (signal_confidence * polarity) for matched signals (0-100)
    RawScore = 0.70 * ICP_Fit + 0.30 * Intent

Confidence band (display modifier — NEVER blended into RawScore):
    HIGH:   4+ signals AND avg_confidence >= 0.80 → show RawScore as-is
    MEDIUM: 2-3 signals                           → show RawScore +/-10 band
    LOW:    fewer than 2 signals                  → cap displayed score at 65

Network Proximity: computed via graph traversal, returned as separate field.
"""

import math
import networkx as nx
from datetime import date, datetime
from typing import Optional
from engine.icp_weights import (
    CANDIDATE_PARAMETERS, get_weight, ICP_WEIGHT_VERSION,
    SCORE_OPTIMAL, SCORE_PARTIAL, INTENT_SIGNAL_TYPES
)

# Confidence thresholds
CONF_HIGH_MIN_SIGNALS    = 4
CONF_HIGH_MIN_CONFIDENCE = 0.80
CONF_MEDIUM_MIN_SIGNALS  = 2
CONF_LOW_CAP             = 65

# Network Proximity
MAX_PROXIMITY_HOPS = 2

# Recommended actions
ACTION_PRIORITIZE   = "Prioritize"
ACTION_NURTURE      = "Nurture"
ACTION_DEPRIORITIZE = "Deprioritize"

# Signal polarity: how much each signal type contributes (positive or neutral)
SIGNAL_POLARITY = {
    "credit_rating_change":   lambda v: 1.0 if str(v).lower() in ("upgrade","positive") else -0.5,
    "expansion_announcement": lambda v: 1.0,
    "leadership_change":      lambda v: 0.3,
    "ma_activity":            lambda v: 0.5,
    "equipment_purchase":     lambda v: 1.0,
}


def _normalise(raw_sum: float, max_possible: float) -> float:
    """Scale raw weight sum to 0-100. Returns 50 if max_possible is 0."""
    if max_possible <= 0:
        return 50.0
    return max(0.0, min(100.0, round((raw_sum / max_possible) * 100, 2)))


def _compute_icp_fit(lead: dict, weights: dict) -> tuple:
    """
    Compute ICP Fit (0-100) and build evidence trail.
    Returns (icp_fit, positive_signals, gaps)
    """
    positive_signals = []
    gaps = []
    raw_sum = 0.0

    # Max possible = sum of highest positive weight per parameter
    max_possible = sum(
        max((v for v in weights.get(p, {}).values() if v > 0), default=0.0)
        for p in CANDIDATE_PARAMETERS
    )

    for param, (col, transformer) in CANDIDATE_PARAMETERS.items():
        raw_val = lead.get(col) if lead.get(col) is not None else lead.get(param)
        if raw_val is None:
            gaps.append(f"No {param.replace('_', ' ')} data")
            continue

        val = transformer(raw_val) if transformer is not None else str(raw_val).strip()
        w   = get_weight(weights, param, val)

        if w > 0:
            raw_sum += w
            positive_signals.append({
                "factor":   f"{param.replace('_',' ').title()}: {val}",
                "weight":   w,
                "evidence": f"weight={w:+.4f} from ICP {ICP_WEIGHT_VERSION}",
                "source":   f"icp_{ICP_WEIGHT_VERSION}",
                "param":    param,
                "value":    val,
            })
        elif w < 0:
            gaps.append(f"{param.replace('_',' ').title()}: {val} (negative indicator)")

    icp_fit = _normalise(raw_sum, max_possible)
    return icp_fit, positive_signals, gaps


def _compute_intent(lead: dict, signals: list,
                    as_of_date: Optional[str] = None) -> tuple:
    """
    Compute Intent Score (0-100) from matched signals.
    Enforces temporal leakage prevention: only signals with
    score_eligibility_date <= as_of_date are used.
    Returns (intent_score, matched_signals, avg_confidence)
    """
    if as_of_date is None:
        as_of_date = date.today().isoformat()

    account_id = lead.get("account_id", "")
    matched = [
        s for s in signals
        if (s.get("account_id") == account_id
            and s.get("signal_type") in INTENT_SIGNAL_TYPES
            and str(s.get("score_eligibility_date", "")) <= as_of_date)
    ]

    if not matched:
        return 0.0, [], 0.0

    raw_intent = 0.0
    max_intent = float(len(matched))

    for sig in matched:
        conf     = float(sig.get("extraction_confidence", 0.5))
        polarity = SIGNAL_POLARITY.get(
            sig.get("signal_type", ""), lambda v: 0.5
        )(sig.get("signal_value", "positive"))
        raw_intent += conf * polarity

    intent_score = _normalise(max(0.0, raw_intent), max_intent)
    avg_conf     = round(
        sum(float(s.get("extraction_confidence", 0.5)) for s in matched) / len(matched), 3
    )
    return intent_score, matched, avg_conf


def _confidence_band(n_signals: int, avg_conf: float) -> str:
    """
    Determine confidence band.
    This is a DISPLAY MODIFIER — it modifies how the score is shown,
    never changes RawScore itself.
    """
    if n_signals >= CONF_HIGH_MIN_SIGNALS and avg_conf >= CONF_HIGH_MIN_CONFIDENCE:
        return "HIGH"
    if n_signals >= CONF_MEDIUM_MIN_SIGNALS:
        return "MEDIUM"
    return "LOW"


def _network_proximity(account_id: str, graph) -> dict:
    """
    Find shortest path to a won deal within 2 hops via
    REFERRED_BY or SHARES_GUARANTOR edges.
    Returns a dict with display string and path details.
    This result is DISPLAYED SEPARATELY — never added to RawScore.
    """
    result = {
        "hops": None,
        "via_account": None,
        "via_deal": None,
        "relationship_type": None,
        "display": "None detected within 2 hops",
    }

    if graph is None or account_id not in graph:
        return result

    try:
        for neighbor, hops in nx.single_source_shortest_path_length(
            graph, account_id, cutoff=MAX_PROXIMITY_HOPS
        ).items():
            if neighbor == account_id:
                continue
            if graph.nodes.get(neighbor, {}).get("type") != "Account":
                continue
            for _, deal_id in graph.out_edges(neighbor):
                deal = graph.nodes.get(deal_id, {})
                if deal.get("type") == "Deal" and deal.get("outcome") == "won":
                    try:
                        path = nx.shortest_path(graph, account_id, neighbor)
                        rel  = graph.edges.get((path[0], path[1]), {}).get("rel", "CONNECTED")
                        result.update({
                            "hops":              hops,
                            "via_account":       neighbor,
                            "via_deal":          deal_id,
                            "relationship_type": rel,
                            "display": (
                                f"{hops} hop{'s' if hops != 1 else ''} via "
                                f"{rel} to {deal_id} (won)"
                            ),
                        })
                        return result
                    except nx.NetworkXNoPath:
                        continue
    except Exception:
        pass

    return result


def _recommended_action(score: float) -> str:
    if score >= SCORE_OPTIMAL:
        return ACTION_PRIORITIZE
    if score >= SCORE_PARTIAL:
        return ACTION_NURTURE
    return ACTION_DEPRIORITIZE


def _fit_category(score: float) -> str:
    if score >= SCORE_OPTIMAL:
        return "Optimal Fit"
    if score >= SCORE_PARTIAL:
        return "Partial Fit"
    return "Unfit"


def score_lead(lead: dict, signals: list, weights: dict,
               graph=None, as_of_date: Optional[str] = None) -> dict:
    """
    Score one lead. Returns a complete score card dict.

    Called by:
      - POST /score/all   (bulk upload pipeline)
      - POST /score/single (manual entry pipeline)
    Both paths produce identical output for identical input — no shortcuts.

    Args:
        lead:       dict of lead fields (floats already banded at API boundary)
        signals:    list of all signal dicts
        weights:    output of derive_weights()
        graph:      NetworkX DiGraph or None
        as_of_date: ISO date for temporal leakage prevention (default: today)

    Returns:
        Score card dict with guaranteed fields:
        score, raw_score, icp_fit, intent_score, confidence, confidence_note,
        fit_category, recommended_action, network_proximity, positive_signals,
        gaps, icp_version, scored_at, signal_count, avg_signal_confidence
    """
    account_id = lead.get("account_id", "unknown")

    # 1. ICP Fit
    icp_fit, pos_signals, gaps = _compute_icp_fit(lead, weights)

    # 2. Intent
    intent_score, matched_sigs, avg_conf = _compute_intent(lead, signals, as_of_date)

    # Append signal evidence to positive_signals
    for sig in matched_sigs:
        pos_signals.append({
            "factor":   f"{sig['signal_type'].replace('_',' ').title()}: {sig.get('signal_value','')}",
            "weight":   float(sig.get("extraction_confidence", 0.5)),
            "evidence": sig.get("evidence_snippet", "Signal detected"),
            "source":   sig.get("signal_id", "unknown"),
            "param":    "intent_signal",
            "value":    sig.get("signal_value", ""),
        })

    # 3. RawScore — confidence is NOT in this formula
    raw_score = round(0.70 * icp_fit + 0.30 * intent_score, 2)

    # 4. Confidence band (display modifier only)
    confidence = _confidence_band(len(matched_sigs), avg_conf)

    # 5. Displayed score — this is where confidence modifies display, not raw_score
    if confidence == "LOW":
        displayed_score = min(raw_score, CONF_LOW_CAP)
        score_note      = f"Score capped at {CONF_LOW_CAP} — LOW confidence ({len(matched_sigs)} signals)"
    elif confidence == "MEDIUM":
        displayed_score = raw_score
        score_note      = f"MEDIUM confidence — {len(matched_sigs)} signals, show +/-10 uncertainty"
    else:
        displayed_score = raw_score
        score_note      = f"HIGH confidence — {len(matched_sigs)} signals, avg conf {avg_conf}"

    # 6. Network Proximity — separate, never blended
    proximity = _network_proximity(account_id, graph)

    # 7. Assemble and return
    return {
        "account_id":            account_id,
        "company_name":          lead.get("company_name", account_id),
        "score":                 displayed_score,
        "raw_score":             raw_score,
        "icp_fit":               round(icp_fit, 2),
        "intent_score":          round(intent_score, 2),
        "confidence":            confidence,
        "confidence_note":       score_note,
        "fit_category":          _fit_category(displayed_score),
        "recommended_action":    _recommended_action(displayed_score),
        "network_proximity":     proximity,
        "positive_signals":      sorted(pos_signals, key=lambda x: abs(x["weight"]), reverse=True)[:10],
        "gaps":                  gaps[:5],
        "icp_version":           ICP_WEIGHT_VERSION,
        "scored_at":             datetime.utcnow().isoformat() + "Z",
        "signal_count":          len(matched_sigs),
        "avg_signal_confidence": avg_conf,
    }
