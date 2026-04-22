"""
agent/rules.py
Rule-based advice engine that generates concrete financial recommendations
based on analysis output — no LLM required.
"""

from agent.analyzer import (
    detect_overspending,
    spending_trend,
    category_trends,
    estimate_savings,
    compute_risk_score,
    detect_anomalies,
)
import pandas as pd


SAVING_TIPS = {
    "Food": [
        "Meal-prep at home to cut food delivery costs by up to 60%.",
        "Limit food delivery apps to 2–3 times per week.",
        "Use grocery stores instead of quick-commerce apps for regular items.",
        "Set a weekly food budget and track it actively.",
    ],
    "Shopping": [
        "Implement a 48-hour rule before making non-essential purchases.",
        "Unsubscribe from promotional emails to reduce impulse buying.",
        "Compare prices across platforms before purchasing.",
        "Create a monthly shopping budget and stick to it.",
    ],
    "Transport": [
        "Use public transport or monthly passes for regular commutes.",
        "Carpool with colleagues to split cab costs.",
        "Walk or cycle for short distances (< 2 km).",
        "Book cabs in advance to avoid surge pricing.",
    ],
    "Bills": [
        "Audit your subscriptions — cancel ones you haven't used in 30 days.",
        "Switch to annual plans for streaming services to save 20–30%.",
        "Compare telecom plans annually for better deals.",
        "Install energy-saving appliances to reduce electricity bills.",
    ],
    "Rent": [
        "Consider PG or house-sharing to halve accommodation costs.",
        "Negotiate rent with your landlord — many are open to it.",
        "Explore relocating slightly farther from city center for cheaper rent.",
    ],
    "Others": [
        "Track all miscellaneous expenses to find hidden spending leaks.",
        "Set a monthly cap for unplanned/impulse purchases.",
    ],
}


def generate_advice(df: pd.DataFrame, income: float = None) -> list[dict]:
    """
    Return a list of actionable advice dicts with priority and category.
    Each dict: {priority, category, advice, reason}
    """
    advice_list = []

    overspend = detect_overspending(df)
    trend = spending_trend(df)
    cat_trends = category_trends(df)
    savings = estimate_savings(df, income)
    risk = compute_risk_score(df)
    anomalies = detect_anomalies(df)

    # ── Overspending Advice ───────────────────────────────────────────────
    for cat, data in overspend.items():
        if data["overspending"]:
            tips = SAVING_TIPS.get(cat, SAVING_TIPS["Others"])
            advice_list.append({
                "priority": "HIGH",
                "category": cat,
                "advice": tips[0],
                "reason": (
                    f"You spent {data['actual_pct']}% on {cat}, "
                    f"exceeding the recommended {data['budget_pct']}% "
                    f"by {data['excess_pct']}%."
                ),
            })

    # ── Rising Category Trends ────────────────────────────────────────────
    for cat, t in cat_trends.items():
        if t["direction"] == "up" and t["pct_change"] > 20:
            tips = SAVING_TIPS.get(cat, SAVING_TIPS["Others"])
            advice_list.append({
                "priority": "MEDIUM",
                "category": cat,
                "advice": tips[1] if len(tips) > 1 else tips[0],
                "reason": (
                    f"{cat} spending rose {t['pct_change']}% "
                    f"(₹{t['prev']} → ₹{t['curr']}) this month."
                ),
            })

    # ── Overall Spending Trend ────────────────────────────────────────────
    if trend["direction"] == "increasing" and trend["pct_change"] > 15:
        advice_list.append({
            "priority": "HIGH",
            "category": "General",
            "advice": "Create a monthly budget and review it weekly to catch overruns early.",
            "reason": (
                f"Your total spending increased {trend['pct_change']}% "
                "vs last month."
            ),
        })

    # ── Savings Rate ──────────────────────────────────────────────────────
    if savings["status"] == "poor":
        advice_list.append({
            "priority": "HIGH",
            "category": "Savings",
            "advice": (
                "Apply the 50/30/20 rule: 50% needs, 30% wants, 20% savings. "
                "Automate a SIP or recurring deposit on salary day."
            ),
            "reason": (
                f"Estimated savings rate is {savings['savings_rate_pct']}% — "
                "well below the healthy 20% benchmark."
            ),
        })
    elif savings["status"] == "moderate":
        advice_list.append({
            "priority": "MEDIUM",
            "category": "Savings",
            "advice": "Increase savings by 5% each month by cutting one discretionary category.",
            "reason": (
                f"Savings rate of {savings['savings_rate_pct']}% is acceptable "
                "but can be improved."
            ),
        })

    # ── Anomalies ─────────────────────────────────────────────────────────
    if len(anomalies) > 0:
        top = anomalies.sort_values("amount", ascending=False).iloc[0]
        advice_list.append({
            "priority": "MEDIUM",
            "category": top["category"],
            "advice": "Review unusually large transactions and verify they are legitimate.",
            "reason": (
                f"Detected {len(anomalies)} unusual transaction(s). "
                f"Largest: ₹{top['amount']} on {top['description']}."
            ),
        })

    # ── Risk Score ────────────────────────────────────────────────────────
    if risk["level"] == "High":
        advice_list.append({
            "priority": "HIGH",
            "category": "Risk",
            "advice": "Build a 3-month emergency fund before making any discretionary purchases.",
            "reason": f"Your financial risk score is {risk['score']}/100 (High).",
        })

    # Sort: HIGH → MEDIUM → LOW
    priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    advice_list.sort(key=lambda x: priority_order.get(x["priority"], 3))

    # Deduplicate by advice text
    seen = set()
    unique = []
    for a in advice_list:
        if a["advice"] not in seen:
            seen.add(a["advice"])
            unique.append(a)

    return unique


def generate_summary_text(df: pd.DataFrame, income: float = None) -> str:
    """Plain-text financial health summary (no LLM required)."""
    savings = estimate_savings(df, income)
    risk = compute_risk_score(df)
    overspend = detect_overspending(df)
    trend = spending_trend(df)

    over_cats = [cat for cat, v in overspend.items() if v["overspending"]]

    lines = [
        f"📊 Financial Health Summary",
        f"─────────────────────────────",
        f"• Avg Monthly Spend : ₹{savings['avg_monthly_spend']:,.0f}",
        f"• Est. Monthly Savings: ₹{savings['estimated_savings']:,.0f} ({savings['savings_rate_pct']}%)",
        f"• Risk Score        : {risk['score']}/100 ({risk['level']})",
        f"• Spending Trend    : {trend['direction'].title()} ({trend['pct_change']:+.1f}%)",
    ]

    if over_cats:
        lines.append(f"• Overspending In   : {', '.join(over_cats)}")
    else:
        lines.append("• No category overspending detected ✅")

    lines.append("")
    lines.append(risk["interpretation"])

    return "\n".join(lines)


def answer_question_rules(question: str, df: pd.DataFrame) -> str:
    """
    Simple keyword-matching Q&A over rule engine output.
    Used as fallback when LLM is not available.
    """
    q = question.lower()
    advice = generate_advice(df)
    savings = estimate_savings(df)
    risk = compute_risk_score(df)
    overspend = detect_overspending(df)
    trend = spending_trend(df)
    cat_totals = df.groupby("category")["amount"].sum().sort_values(ascending=False)

    if any(w in q for w in ["overspend", "too much", "where am i spending"]):
        over_cats = [
            f"{cat} ({v['actual_pct']}% vs budget {v['budget_pct']}%)"
            for cat, v in overspend.items() if v["overspending"]
        ]
        if over_cats:
            return "You are overspending in:\n• " + "\n• ".join(over_cats)
        return "Great news! You're within budget across all categories."

    if any(w in q for w in ["save", "saving", "savings"]):
        tips = [a["advice"] for a in advice if a["category"] == "Savings"]
        base = f"Your current savings rate is estimated at {savings['savings_rate_pct']}%.\n"
        if tips:
            return base + "Tips to save more:\n• " + "\n• ".join(tips)
        high_cat = cat_totals.index[0] if len(cat_totals) > 0 else "Shopping"
        return base + f"Focus on reducing {high_cat} — it's your largest expense."

    if any(w in q for w in ["analyze", "analysis", "overview", "summary", "report"]):
        return generate_summary_text(df)

    if any(w in q for w in ["risk", "score", "health"]):
        return (
            f"Your financial risk score is {risk['score']}/100 ({risk['level']}).\n"
            f"{risk['interpretation']}"
        )

    if any(w in q for w in ["trend", "increasing", "decreasing", "change"]):
        return (
            f"Your spending is {trend['direction']} "
            f"({trend['pct_change']:+.1f}% vs last month)."
        )

    if any(w in q for w in ["tip", "advice", "suggest", "recommend", "how can"]):
        top_advice = advice[:3]
        if not top_advice:
            return "Your finances look good! Keep tracking your expenses."
        lines = [f"[{a['priority']}] {a['category']}: {a['advice']}" for a in top_advice]
        return "Top recommendations:\n" + "\n".join(lines)

    # Default: full summary
    return generate_summary_text(df)
