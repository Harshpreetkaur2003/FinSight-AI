"""
utils/export.py
Generate downloadable PDF/CSV/Excel financial reports.
"""

import io
import pandas as pd
from datetime import datetime
from agent.analyzer import full_analysis_report, estimate_savings, compute_risk_score
from agent.rules import generate_advice


def export_csv(df: pd.DataFrame) -> bytes:
    """Export categorized transactions as CSV."""
    return df.to_csv(index=False).encode("utf-8")


def export_excel(df: pd.DataFrame) -> bytes:
    """Export multi-sheet Excel: transactions + summary + advice."""
    buf = io.BytesIO()
    report = full_analysis_report(df)
    advice = generate_advice(df)

    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        # Sheet 1: Transactions
        df.drop(columns=["month"], errors="ignore").to_excel(
            writer, sheet_name="Transactions", index=False
        )

        # Sheet 2: Monthly Summary
        pd.DataFrame(report["monthly_summary"]).to_excel(
            writer, sheet_name="Monthly Summary", index=False
        )

        # Sheet 3: Category Totals
        pd.Series(report["category_totals"], name="Amount").reset_index().rename(
            columns={"index": "Category"}
        ).to_excel(writer, sheet_name="Category Totals", index=False)

        # Sheet 4: Advice
        pd.DataFrame(advice)[["priority", "category", "advice", "reason"]].to_excel(
            writer, sheet_name="Recommendations", index=False
        )

        # Sheet 5: Anomalies
        if report["anomalies"]:
            pd.DataFrame(report["anomalies"]).to_excel(
                writer, sheet_name="Anomalies", index=False
            )

    buf.seek(0)
    return buf.read()


def export_text_report(df: pd.DataFrame) -> str:
    """Generate a plain-text financial report."""
    report = full_analysis_report(df)
    savings = report["savings"]
    risk = report["risk_score"]
    advice = generate_advice(df)
    now = datetime.now().strftime("%d %b %Y %H:%M")

    lines = [
        "=" * 60,
        "       FINSIGHT AI — FINANCIAL HEALTH REPORT",
        f"       Generated: {now}",
        "=" * 60,
        "",
        "EXECUTIVE SUMMARY",
        "-" * 40,
        f"  Avg Monthly Spend : ₹{savings['avg_monthly_spend']:>10,.2f}",
        f"  Est. Monthly Saves: ₹{savings['estimated_savings']:>10,.2f}",
        f"  Savings Rate      : {savings['savings_rate_pct']:>9.1f}%",
        f"  Financial Risk    : {risk['score']:>9}/100  ({risk['level']})",
        "",
        "CATEGORY BREAKDOWN",
        "-" * 40,
    ]

    for cat, amt in sorted(report["category_totals"].items(), key=lambda x: -x[1]):
        lines.append(f"  {cat:<15} ₹{amt:>10,.2f}")

    lines += [
        "",
        "MONTHLY TREND",
        "-" * 40,
    ]
    for m in report["monthly_summary"]:
        lines.append(f"  {m['month_str']:<12} ₹{m['total']:>10,.2f}  ({m['count']} txns)")

    lines += [
        "",
        "TOP RECOMMENDATIONS",
        "-" * 40,
    ]
    for i, a in enumerate(advice[:5], 1):
        lines.append(f"  {i}. [{a['priority']}] {a['category']}")
        lines.append(f"     → {a['advice']}")
        lines.append(f"     Reason: {a['reason']}")
        lines.append("")

    if report["anomalies"]:
        lines += ["ANOMALOUS TRANSACTIONS", "-" * 40]
        for anm in report["anomalies"]:
            lines.append(
                f"  {anm['date']} | {anm['description'][:30]:<30} | "
                f"₹{anm['amount']:>8,.2f} | Z={anm['z_score']:.2f}"
            )
        lines.append("")

    lines.append("=" * 60)
    lines.append("      End of Report — FinSight AI")
    lines.append("=" * 60)

    return "\n".join(lines)
