"""
engine/data_profiler.py
=======================
Read-only dataset statistics for the dashboard.
Never modifies any data.
"""

import pandas as pd
from collections import Counter


def profile_dataset(deals_df: pd.DataFrame) -> dict:
    """Summary stats from historical_deals.csv for GET /dashboard/summary."""
    total    = len(deals_df)
    won      = int((deals_df["outcome"] == "won").sum())
    lost     = int((deals_df["outcome"] == "lost").sum())
    win_rate = round(won / total * 100, 1) if total > 0 else 0.0

    # Industry breakdown
    ind_data = []
    for ind, grp in deals_df.groupby("industry"):
        w = int((grp["outcome"] == "won").sum())
        l = int((grp["outcome"] == "lost").sum())
        t = w + l
        ind_data.append({
            "industry": ind, "won": w, "lost": l, "total": t,
            "win_rate": round(w / t * 100, 1) if t > 0 else 0.0,
        })
    ind_data.sort(key=lambda x: x["total"], reverse=True)

    # Geography breakdown
    geo_data = []
    for geo, grp in deals_df.groupby("geography"):
        w = int((grp["outcome"] == "won").sum())
        l = int((grp["outcome"] == "lost").sum())
        t = w + l
        geo_data.append({
            "region": geo, "won": w, "lost": l, "total": t,
            "win_rate": round(w / t * 100, 1) if t > 0 else 0.0,
        })
    geo_data.sort(key=lambda x: x["total"], reverse=True)

    # Parameter coverage
    coverage = {}
    for col in ["dscr","debt_to_ebitda_ratio","years_in_business",
                "credit_rating","loan_purpose","collateral_type"]:
        if col in deals_df.columns:
            coverage[col] = round(deals_df[col].notna().sum() / total * 100, 1)

    # Date range
    date_range = {"earliest": "unknown", "latest": "unknown"}
    if "deal_created_date" in deals_df.columns:
        dates = pd.to_datetime(deals_df["deal_created_date"], errors="coerce").dropna()
        if len(dates) > 0:
            date_range = {
                "earliest": str(dates.min())[:10],
                "latest":   str(dates.max())[:10],
            }

    return {
        "total_deals":         total,
        "total_won":           won,
        "total_lost":          lost,
        "win_rate_pct":        win_rate,
        "industry_breakdown":  ind_data,
        "geography_breakdown": geo_data,
        "parameter_coverage":  coverage,
        "date_range":          date_range,
    }


def profile_leads(leads_df: pd.DataFrame) -> dict:
    """Summary stats from new_leads.csv for GET /dashboard/summary."""
    total = len(leads_df)
    geo   = Counter(leads_df["geography"].tolist()  if "geography" in leads_df.columns else [])
    ind   = Counter(leads_df["industry"].tolist()   if "industry"  in leads_df.columns else [])
    return {
        "total_leads":         total,
        "geography_breakdown": [{"region":k,"count":v}
                                 for k,v in sorted(geo.items(), key=lambda x:-x[1])],
        "industry_breakdown":  [{"industry":k,"count":v}
                                 for k,v in sorted(ind.items(), key=lambda x:-x[1])],
    }
