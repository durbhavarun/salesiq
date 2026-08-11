"""
engine/backtest/split.py
========================
Date-based backtest harness.

Why date-based (not random):
    Random split allows future deals into the training set relative to test deals.
    The model then sees information it would not have had in production.
    Date-based split mirrors production exactly: train on past, test on future.

Metrics:
    Spearman rank correlation: do higher scores predict wins?
    Precision@10: of the top 10 scored leads, how many won?
"""

import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple
from scipy.stats import spearmanr


@dataclass
class BacktestResult:
    spearman_correlation: float
    spearman_pvalue:      float
    precision_at_10:      float
    train_size:           int
    test_size:            int
    cutoff_date:          str
    won_in_test:          int
    disclaimer:           str


def date_based_split(
    deals_df: pd.DataFrame,
    cutoff_date: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Split deals by deal_created_date.
    Before cutoff → train. On/after cutoff → test.
    Default cutoff = 70th percentile date (~70/30 split).

    Returns: (train_df, test_df, cutoff_date_used)
    """
    df = deals_df.copy()
    df["deal_created_date"] = pd.to_datetime(df["deal_created_date"])
    df = df.sort_values("deal_created_date")

    if cutoff_date is None:
        cutoff = df["deal_created_date"].quantile(0.70)
    else:
        cutoff = pd.to_datetime(cutoff_date)

    cutoff_str = cutoff.strftime("%Y-%m-%d")
    train_df   = df[df["deal_created_date"] <  cutoff].copy()
    test_df    = df[df["deal_created_date"] >= cutoff].copy()
    return train_df, test_df, cutoff_str


def run_backtest(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    signals_list: list
) -> BacktestResult:
    """
    Derive weights from train_df, score test_df, compute metrics.
    """
    from engine.icp_weights import derive_weights
    from engine.scoring_engine import score_lead

    weights = derive_weights(train_df)
    scores, actuals = [], []

    for _, row in test_df.iterrows():
        lead           = row.to_dict()
        actual_outcome = lead.pop("outcome", None)
        lead.pop("deal_id", None)
        lead.pop("deal_closed_date", None)
        as_of = str(row.get("deal_created_date", ""))[:10]

        try:
            result = score_lead(lead, signals_list, weights, graph=None, as_of_date=as_of)
            scores.append(result["score"])
            actuals.append(1 if actual_outcome == "won" else 0)
        except Exception:
            continue

    if len(scores) < 5:
        return BacktestResult(
            spearman_correlation=0.0, spearman_pvalue=1.0,
            precision_at_10=0.0,
            train_size=len(train_df), test_size=len(test_df),
            cutoff_date="auto", won_in_test=sum(actuals),
            disclaimer="Insufficient test data (< 5 scored leads).",
        )

    corr, pval = spearmanr(scores, actuals)
    corr = round(float(corr) if corr == corr else 0.0, 4)   # guard NaN
    pval = round(float(pval) if pval == pval else 1.0, 4)

    paired  = sorted(zip(scores, actuals), key=lambda x: x[0], reverse=True)
    p_at_10 = round(sum(a for _, a in paired[:10]) / min(10, len(paired)), 4)

    disclaimer = (
        f"Trained on {len(train_df)} deals, tested on {len(test_df)} "
        f"({sum(actuals)} won). "
        f"With n={len(test_df)} test deals, Spearman r={corr} is directional — "
        f"treat as indicative, not definitive."
    )

    return BacktestResult(
        spearman_correlation=corr, spearman_pvalue=pval,
        precision_at_10=p_at_10,
        train_size=len(train_df), test_size=len(test_df),
        cutoff_date="auto", won_in_test=sum(actuals),
        disclaimer=disclaimer,
    )
