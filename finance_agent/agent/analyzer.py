"""
agent/analyzer.py
Core analysis engine: spending totals, trends, anomalies, risk scoring.
"""

import numpy as np
import pandas as pd
from scipy import stats


CATEGORY_BUDGETS = {
    "Food": 0.30,
    "Rent": 0.35,
    "Transport": 0.10,
    "Shopping": 0.10,
    "Bills": 0.10,
    "Others": 0.05,
}

RISK_THRESHOLDS = {
    "low": (0, 40),
    "medium": (40, 65),
    "high": (65, 100),
}


# ── Monthly Summaries ──────────────────────────────────────────────────────────

def monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate spend by month."""
    summary = (
        df.groupby("month_str")["amount"]
        .agg(total="sum", count="count", avg="mean")
        .reset_index()
    )
    summary["total"] = summary["total"].round(2)
    summary["avg"] = summary["avg"].round(2)
    return summary


def category_monthly_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot: rows = month, cols = category totals."""
    pivot = (
        df.pivot_table(
            index="month_str", columns="category",
            values="amount", aggfunc="sum", fill_value=0,
        )
        .reset_index()
    )
    return pivot


def category_totals(df: pd.DataFrame) -> pd.Series:
    """Overall category totals across entire dataset."""
    return df.groupby("category")["amount"].sum().round(2).sort_values(ascending=False)


# ── Trend Detection ────────────────────────────────────────────────────────────

def spending_trend(df: pd.DataFrame) -> dict:
    """
    Compute linear trend across months.
    Returns slope direction and percent change last→current month.
    """
    monthly = df.groupby("month")["amount"].sum().sort_index()
    if len(monthly) < 2:
        return {"direction": "stable", "pct_change": 0.0, "monthly_values": monthly.to_dict()}

    values = monthly.values.astype(float)
    x = np.arange(len(values))
    slope, _, _, _, _ = stats.linregress(x, values)

    last, current = values[-2], values[-1]
    pct_change = ((current - last) / last * 100) if last > 0 else 0.0

    return {
        "direction": "increasing" if slope > 0 else "decreasing",
        "slope": round(float(slope), 2),
        "pct_change": round(float(pct_change), 2),
        "monthly_values": {str(k): round(float(v), 2) for k, v in monthly.items()},
    }


def category_trends(df: pd.DataFrame) -> dict:
    """Return pct change per category between last two months."""
    months = sorted(df["month"].unique())
    if len(months) < 2:
        return {}

    m1, m2 = months[-2], months[-1]
    prev = df[df["month"] == m1].groupby("category")["amount"].sum()
    curr = df[df["month"] == m2].groupby("category")["amount"].sum()

    all_cats = set(prev.index) | set(curr.index)
    trends = {}
    for cat in all_cats:
        p, c = prev.get(cat, 0), curr.get(cat, 0)
        pct = ((c - p) / p * 100) if p > 0 else (100.0 if c > 0 else 0.0)
        trends[cat] = {
            "prev": round(float(p), 2),
            "curr": round(float(c), 2),
            "pct_change": round(float(pct), 2),
            "direction": "up" if pct > 5 else ("down" if pct < -5 else "stable"),
        }
    return trends


# ── Anomaly Detection ─────────────────────────────────────────────────────────

def detect_anomalies(df: pd.DataFrame, z_threshold: float = 2.5) -> pd.DataFrame:
    """
    Flag transactions where amount is z_threshold std devs above category mean.
    Returns subset of df with anomaly flag.
    """
    df = df.copy()
    df["z_score"] = 0.0
    df["is_anomaly"] = False

    for cat, grp in df.groupby("category"):
        if len(grp) < 3:
            continue
        mu, sigma = grp["amount"].mean(), grp["amount"].std()
        if sigma == 0:
            continue
        z = (grp["amount"] - mu) / sigma
        df.loc[grp.index, "z_score"] = z.round(3)
        df.loc[grp.index, "is_anomaly"] = z.abs() > z_threshold

    anomalies = df[df["is_anomaly"]].copy()
    return anomalies[["date", "description", "amount", "category", "z_score"]]


def detect_spending_spikes(df: pd.DataFrame) -> list[dict]:
    """Identify days with unusually high spend."""
    daily = df.groupby("date")["amount"].sum()
    mean, std = daily.mean(), daily.std()
    if std == 0:
        return []

    spikes = daily[daily > mean + 2 * std]
    return [
        {"date": str(d.date()), "amount": round(float(a), 2), "z_score": round((a - mean) / std, 2)}
        for d, a in spikes.items()
    ]


# ── Overspending Detection ─────────────────────────────────────────────────────

def detect_overspending(df: pd.DataFrame) -> dict:
    """
    Compare category spend % vs recommended budget %.
    Returns dict of category → overspend status.
    """
    total = df["amount"].sum()
    if total == 0:
        return {}

    cat_totals = df.groupby("category")["amount"].sum()
    results = {}
    for cat, budget_pct in CATEGORY_BUDGETS.items():
        actual = cat_totals.get(cat, 0)
        actual_pct = actual / total
        over = actual_pct > budget_pct
        results[cat] = {
            "actual_amount": round(float(actual), 2),
            "actual_pct": round(actual_pct * 100, 1),
            "budget_pct": round(budget_pct * 100, 1),
            "overspending": over,
            "excess_pct": round((actual_pct - budget_pct) * 100, 1) if over else 0.0,
        }
    return results


# ── Savings Estimation ─────────────────────────────────────────────────────────

def estimate_savings(df: pd.DataFrame, assumed_income: float = None) -> dict:
    """
    Estimate savings. If income not given, infer as 1.5× monthly avg spend.
    """
    monthly = df.groupby("month")["amount"].sum()
    avg_monthly_spend = float(monthly.mean())

    if assumed_income is None:
        assumed_income = avg_monthly_spend * 1.5

    savings = assumed_income - avg_monthly_spend
    savings_rate = (savings / assumed_income * 100) if assumed_income > 0 else 0

    return {
        "assumed_income": round(assumed_income, 2),
        "avg_monthly_spend": round(avg_monthly_spend, 2),
        "estimated_savings": round(savings, 2),
        "savings_rate_pct": round(savings_rate, 1),
        "status": "healthy" if savings_rate >= 20 else ("moderate" if savings_rate >= 10 else "poor"),
    }


# ── Risk Scoring ──────────────────────────────────────────────────────────────

def compute_risk_score(df: pd.DataFrame) -> dict:
    """
    Financial risk score 0–100 (higher = riskier).
    Factors: savings rate, overspending categories, spending trend, anomalies.
    """
    score = 0
    breakdown = {}

    # Factor 1: Savings rate (0–30 pts)
    savings = estimate_savings(df)
    sr = savings["savings_rate_pct"]
    savings_score = max(0, 30 - int(sr * 1.5))
    score += savings_score
    breakdown["savings_risk"] = savings_score

    # Factor 2: Overspending categories (0–30 pts)
    overspend = detect_overspending(df)
    over_count = sum(1 for v in overspend.values() if v["overspending"])
    over_score = min(30, over_count * 8)
    score += over_score
    breakdown["overspending_risk"] = over_score

    # Factor 3: Spending trend (0–20 pts)
    trend = spending_trend(df)
    trend_score = min(20, max(0, int(trend["pct_change"] / 5)))
    score += trend_score
    breakdown["trend_risk"] = trend_score

    # Factor 4: Anomaly frequency (0–20 pts)
    anomalies = detect_anomalies(df)
    anomaly_score = min(20, len(anomalies) * 4)
    score += anomaly_score
    breakdown["anomaly_risk"] = anomaly_score

    score = min(100, score)
    level = "Low" if score < 40 else ("Medium" if score < 65 else "High")

    return {
        "score": score,
        "level": level,
        "breakdown": breakdown,
        "interpretation": _risk_interpretation(level),
    }


def _risk_interpretation(level: str) -> str:
    mapping = {
        "Low": "Your finances look healthy. Keep maintaining disciplined spending habits.",
        "Medium": "Some areas need attention. Review overspending categories and try to save more.",
        "High": "Your financial health is at risk. Immediate action needed to control expenses and build savings.",
    }
    return mapping.get(level, "")


# ── Full Report ────────────────────────────────────────────────────────────────

def full_analysis_report(df: pd.DataFrame) -> dict:
    """Run all analyses and return unified report dict."""
    return {
        "monthly_summary": monthly_summary(df).to_dict(orient="records"),
        "category_totals": category_totals(df).to_dict(),
        "category_monthly": category_monthly_breakdown(df).to_dict(orient="records"),
        "spending_trend": spending_trend(df),
        "category_trends": category_trends(df),
        "overspending": detect_overspending(df),
        "savings": estimate_savings(df),
        "anomalies": detect_anomalies(df).to_dict(orient="records"),
        "spending_spikes": detect_spending_spikes(df),
        "risk_score": compute_risk_score(df),
    }
