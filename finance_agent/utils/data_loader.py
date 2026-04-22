"""
utils/data_loader.py
Handles CSV loading, text cleaning, and preprocessing for transaction data.
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime


def load_transactions(filepath: str) -> pd.DataFrame:
    """Load transaction CSV and return cleaned DataFrame."""
    df = pd.read_csv(filepath)
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"date", "amount", "description"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "amount"])
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0).abs()
    df["description"] = df["description"].astype(str).apply(clean_text)
    df = df[df["amount"] > 0].reset_index(drop=True)

    df["month"] = df["date"].dt.to_period("M")
    df["month_str"] = df["date"].dt.strftime("%b %Y")
    df["day_of_week"] = df["date"].dt.day_name()
    df["week"] = df["date"].dt.isocalendar().week.astype(int)

    return df


def clean_text(text: str) -> str:
    """Lowercase, strip punctuation, normalize whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def load_training_data(filepath: str) -> tuple[list, list]:
    """Load labeled training data and return (descriptions, labels)."""
    df = pd.read_csv(filepath)
    df["description"] = df["description"].astype(str).apply(clean_text)
    return df["description"].tolist(), df["category"].tolist()


def split_by_month(df: pd.DataFrame) -> dict:
    """Return a dict keyed by month Period with sub-DataFrames."""
    return {period: grp.copy() for period, grp in df.groupby("month")}


def get_recent_months(df: pd.DataFrame, n: int = 2) -> list:
    """Return the n most recent month periods found in df."""
    months = sorted(df["month"].unique())
    return months[-n:] if len(months) >= n else months


def summarize_month(df_month: pd.DataFrame) -> dict:
    """Quick numeric summary for a single month's transactions."""
    return {
        "total_spend": round(df_month["amount"].sum(), 2),
        "num_transactions": len(df_month),
        "avg_transaction": round(df_month["amount"].mean(), 2),
        "max_transaction": round(df_month["amount"].max(), 2),
        "category_totals": (
            df_month.groupby("category")["amount"]
            .sum()
            .round(2)
            .to_dict()
            if "category" in df_month.columns
            else {}
        ),
    }
