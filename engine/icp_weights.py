"""
engine/icp_weights.py
=====================
Derives ICP parameter weights from historical deal outcomes.

Formula (Laplace smoothing, alpha=1):
    p_won  = (count_won  + 1) / (total_won  + 2)
    p_lost = (count_lost + 1) / (total_lost + 2)
    weight = p_won - p_lost

Parameters with fewer than MIN_DEALS_PER_CLASS supporting deals
in either outcome class receive weight=0.0 (minimum-evidence gate).

Why 21 parameters (not 25 or 22):
    - 3 fields excluded to prevent Network Proximity double-counting:
      referred_by_portfolio, shares_guarantor_with_won, network_proximity_hops
    - 1 field removed (collateral_coverage) — it was a duplicate of
      collateral_type reading the same column with no independent signal.
      Double-counting within ICP Fit is the same class of error as
      double-counting with Network Proximity.
"""

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ── Valid signal types for lending vertical ──────────────────────────
# Used by anomaly_detector.py Rule 3 (invalid signal type check)
INTENT_SIGNAL_TYPES = [
    "credit_rating_change",
    "expansion_announcement",
    "leadership_change",
    "ma_activity",
    "equipment_purchase",
]

# ── Scoring display thresholds ───────────────────────────────────────
SCORE_OPTIMAL = 75   # >= 75 → Optimal Fit (green)
SCORE_PARTIAL = 50   # 50-74 → Partial Fit (amber)
                     # <  50 → Unfit (red)

# ── Laplace smoothing ────────────────────────────────────────────────
LAPLACE_ALPHA = 1

# ── Minimum evidence gate ────────────────────────────────────────────
MIN_DEALS_PER_CLASS = 3

# ── ICP version for audit trail ──────────────────────────────────────
ICP_WEIGHT_VERSION = "v1.0"


# ════════════════════════════════════════════════════════════════════
# FLOAT BANDING FUNCTIONS
# Called at the API boundary before any value reaches the scoring engine
# ════════════════════════════════════════════════════════════════════

def band_dscr(value) -> str:
    """
    Debt Service Coverage Ratio.
    > 1.50  = strong   (comfortably covers payments)
    1.20-1.50 = adequate (covers with cushion)
    < 1.20  = low      (tight, higher default risk)
    < 0     = invalid  (physically impossible → triggers ERROR anomaly)
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if v < 0:
        return "invalid"
    if v < 1.20:
        return "low"
    if v <= 1.50:
        return "adequate"
    return "strong"


def band_debt_to_ebitda(value) -> str:
    """
    Debt-to-EBITDA. Lower is better for lenders.
    < 3.0  = low leverage   (healthy)
    3-5    = moderate
    > 5.0  = high leverage  (risky)
    < 0    = invalid
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if v < 0:
        return "invalid"
    if v < 3.0:
        return "low"
    if v <= 5.0:
        return "moderate"
    return "high"


def band_years(value) -> str:
    """
    Years in business.
    < 3    = startup     (limited history)
    3-10   = established
    > 10   = mature      (long track record)
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if v < 3:
        return "startup"
    if v <= 10:
        return "established"
    return "mature"


# ════════════════════════════════════════════════════════════════════
# THE 21 CANDIDATE PARAMETERS
# Maps parameter_name -> (csv_column, transformer_or_None)
#
# EXCLUDED — Network Proximity double-count prevention:
#   referred_by_portfolio      → powers REFERRED_BY graph traversal
#   shares_guarantor_with_won  → powers SHARES_GUARANTOR graph traversal
#   network_proximity_hops     → IS the Network Proximity output value
#
# EXCLUDED — ICP Fit double-count prevention:
#   collateral_coverage        → was identical to collateral_type
#                                 (same column, same transformer, zero
#                                  independent signal)
# ════════════════════════════════════════════════════════════════════

CANDIDATE_PARAMETERS = {
    # ── Firmographic (5) ──────────────────────────────────────────
    "industry":                        ("industry",                     None),
    "employee_band":                   ("employee_band",                None),
    "geography":                       ("geography",                    None),
    "company_stage":                   ("company_stage",                None),
    "business_model":                  ("business_model",               None),

    # ── Financial (5) ─────────────────────────────────────────────
    "dscr_band":                       ("dscr",                         band_dscr),
    "debt_to_ebitda_band":             ("debt_to_ebitda_ratio",         band_debt_to_ebitda),
    "credit_rating":                   ("credit_rating",                None),
    "collateral_type":                 ("collateral_type",              None),
    "years_band":                      ("years_in_business",            band_years),

    # ── Loan-specific (4, not 5 — term_length_band excluded:
    #    column not present in CSV, would silently produce zero weights) ──
    "loan_purpose":                    ("loan_purpose",                 None),
    "loan_amount_band":                ("loan_amount_band",             None),
    "existing_banking_relationship":   ("existing_banking_relationship",None),
    "prior_default_flag":              ("prior_default_flag",           None),

    # ── Growth-signal (5) ─────────────────────────────────────────
    "expansion_signal":                ("loan_purpose",
                                        lambda x: "yes" if str(x) == "expansion"
                                        else "no"),
    "equipment_purchase_signal":       ("loan_purpose",
                                        lambda x: "yes" if str(x) == "equipment_financing"
                                        else "no"),
    "headcount_growth_signal":         ("employee_band",
                                        lambda x: "yes" if str(x) in ["201-500","500+"]
                                        else "no"),
    "revenue_growth_signal":           ("company_stage",
                                        lambda x: "yes" if str(x) == "growth"
                                        else "no"),
    "new_contract_signal":             ("loan_purpose",
                                        lambda x: "yes" if str(x) == "acquisition"
                                        else "no"),

    # ── Relationship-history (2 — trimmed from 5) ─────────────────
    "prior_deal_with_bank":            ("existing_banking_relationship",None),
    "referral_source_quality":         ("referral_source_quality",      None),
}

# Confirm count at import time — fails loudly if someone accidentally adds a duplicate
assert len(CANDIDATE_PARAMETERS) == 21, (
    f"CANDIDATE_PARAMETERS must have exactly 21 entries, found {len(CANDIDATE_PARAMETERS)}"
)


# ════════════════════════════════════════════════════════════════════
# WEIGHT DERIVATION
# ════════════════════════════════════════════════════════════════════

def derive_weights(deals_df: pd.DataFrame) -> dict:
    """
    Compute Laplace-smoothed weights for all 21 parameters.

    Returns:
        dict: {parameter_name: {value_string: float_weight}}
    """
    won_df     = deals_df[deals_df["outcome"] == "won"].copy()
    lost_df    = deals_df[deals_df["outcome"] == "lost"].copy()
    total_won  = len(won_df)
    total_lost = len(lost_df)

    weights = {}

    for param, (col, transformer) in CANDIDATE_PARAMETERS.items():
        if col not in deals_df.columns:
            weights[param] = {}   # Column absent — silently inactive
            continue

        if transformer is not None:
            won_vals  = won_df[col].apply(transformer)
            lost_vals = lost_df[col].apply(transformer)
        else:
            won_vals  = won_df[col].astype(str).str.strip()
            lost_vals = lost_df[col].astype(str).str.strip()

        all_values = (
            set(won_vals.dropna().unique()) |
            set(lost_vals.dropna().unique())
        )
        param_weights = {}

        for val in all_values:
            val_str = str(val)
            if val_str.lower() in ("nan", "none", ""):
                continue

            c_won  = int((won_vals  == val).sum())
            c_lost = int((lost_vals == val).sum())

            # Minimum-evidence gate
            if c_won < MIN_DEALS_PER_CLASS or c_lost < MIN_DEALS_PER_CLASS:
                param_weights[val_str] = 0.0
                continue

            p_won  = (c_won  + LAPLACE_ALPHA) / (total_won  + 2 * LAPLACE_ALPHA)
            p_lost = (c_lost + LAPLACE_ALPHA) / (total_lost + 2 * LAPLACE_ALPHA)
            param_weights[val_str] = round(p_won - p_lost, 4)

        weights[param] = param_weights

    return weights


def get_weight(weights: dict, param: str, value) -> float:
    """
    Safe weight lookup. Returns 0.0 if param or value not found.
    Called by scoring_engine.py for every parameter of every lead.
    """
    return weights.get(param, {}).get(str(value), 0.0)


def weight_summary(weights: dict) -> list:
    """
    Returns [{parameter, value, weight}] sorted by abs(weight) descending.
    Used by the Engine Room dashboard tab.
    """
    rows = []
    for param, vals in weights.items():
        for val, w in vals.items():
            if w != 0.0:
                rows.append({"parameter": param, "value": val, "weight": w})
    return sorted(rows, key=lambda x: abs(x["weight"]), reverse=True)
