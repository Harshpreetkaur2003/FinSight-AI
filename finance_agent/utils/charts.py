"""
utils/charts.py
Matplotlib chart builders for the Streamlit dashboard.
Returns matplotlib Figure objects.
"""

import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── Palette ────────────────────────────────────────────────────────────────────
BG     = "#0D0F1A"
BG2    = "#1A1D2E"
ACCENT = "#6C63FF"
TEXT   = "#E8EAF6"
GRID   = "#2A2D3E"
MUTED  = "#8B9CBF"

CATEGORY_COLORS = {
    "Food":      "#FF6584",
    "Rent":      "#6C63FF",
    "Transport": "#43B97F",
    "Shopping":  "#F7B731",
    "Bills":     "#4FC3F7",
    "Others":    "#BA68C8",
}


def _style(fig, ax_list=None):
    fig.patch.set_facecolor(BG)
    for ax in (ax_list or fig.get_axes()):
        ax.set_facecolor(BG2)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)
        ax.title.set_color(TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
        ax.yaxis.grid(True, color=GRID, linewidth=0.6, linestyle="--")
        ax.set_axisbelow(True)


def pie_chart(df: pd.DataFrame) -> plt.Figure:
    totals = df.groupby("category")["amount"].sum()
    labels = totals.index.tolist()
    sizes  = totals.values
    colors = [CATEGORY_COLORS.get(l, "#999999") for l in labels]

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    wedges, _, autotexts = ax.pie(
        sizes, labels=None, colors=colors,
        autopct="%1.1f%%", startangle=140,
        wedgeprops=dict(edgecolor=BG, linewidth=2),
        pctdistance=0.82,
    )
    for at in autotexts:
        at.set_color(TEXT)
        at.set_fontsize(9)

    circle = plt.Circle((0, 0), 0.58, color=BG)
    ax.add_patch(circle)
    ax.text(0, 0, f"₹{sizes.sum():,.0f}", ha="center", va="center",
            fontsize=11, color=TEXT, fontweight="bold")

    patches = [mpatches.Patch(color=CATEGORY_COLORS.get(l, "#999"), label=l) for l in labels]
    ax.legend(handles=patches, loc="lower center", bbox_to_anchor=(0.5, -0.12),
              ncol=3, frameon=False, fontsize=8, labelcolor=TEXT)
    ax.set_title("Spending by Category", color=TEXT, fontsize=12, pad=10)
    fig.tight_layout()
    return fig


def bar_chart_monthly(df: pd.DataFrame) -> plt.Figure:
    monthly = df.groupby("month_str")["amount"].sum()

    fig, ax = plt.subplots(figsize=(7, 4))
    _style(fig, [ax])

    x    = np.arange(len(monthly))
    bars = ax.bar(x, monthly.values, color=ACCENT, width=0.55, linewidth=0, zorder=3)

    max_val = monthly.values.max() if len(monthly) else 1
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max_val * 0.015,
                f"₹{bar.get_height():,.0f}",
                ha="center", va="bottom", fontsize=8, color=MUTED)

    ax.set_xticks(x)
    ax.set_xticklabels(monthly.index, rotation=30, ha="right")
    ax.set_ylabel("Total Spend (₹)")
    ax.set_title("Monthly Spending", color=TEXT, fontsize=12)
    fig.tight_layout()
    return fig


def category_trend_bar(df: pd.DataFrame) -> plt.Figure:
    pivot = df.pivot_table(
        index="month_str", columns="category",
        values="amount", aggfunc="sum", fill_value=0,
    )
    categories = pivot.columns.tolist()
    months     = pivot.index.tolist()
    x          = np.arange(len(months))

    fig, ax = plt.subplots(figsize=(8, 4))
    _style(fig, [ax])

    bottom = np.zeros(len(months))
    for cat in categories:
        vals = pivot[cat].values
        ax.bar(x, vals, bottom=bottom, label=cat,
               color=CATEGORY_COLORS.get(cat, "#999999"), linewidth=0, zorder=3)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(months, rotation=30, ha="right")
    ax.set_ylabel("Amount (₹)")
    ax.set_title("Category Trends by Month", color=TEXT, fontsize=12)
    ax.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=TEXT, ncol=2)
    fig.tight_layout()
    return fig


def spending_gauge(risk_score: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4, 2.5), subplot_kw={"aspect": "equal"})
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis("off")

    theta = np.linspace(np.pi, 0, 300)
    ax.plot(np.cos(theta), np.sin(theta),
            color=GRID, linewidth=14, solid_capstyle="round")

    color = "#43B97F" if risk_score < 40 else ("#F7B731" if risk_score < 65 else "#FF6584")
    frac  = risk_score / 100.0
    end   = np.pi - frac * np.pi
    t2    = np.linspace(np.pi, end, max(2, int(frac * 300)))
    ax.plot(np.cos(t2), np.sin(t2),
            color=color, linewidth=14, solid_capstyle="round", zorder=5)

    level = "Low" if risk_score < 40 else ("Medium" if risk_score < 65 else "High")
    ax.text(0, -0.15, f"{risk_score}", ha="center", va="center",
            fontsize=26, fontweight="bold", color=color)
    ax.text(0, -0.45, f"/ 100  •  {level} Risk",
            ha="center", va="center", fontsize=9, color=MUTED)
    ax.text(-1.1, -0.05, "0",   ha="center", fontsize=7, color=MUTED)
    ax.text( 1.1, -0.05, "100", ha="center", fontsize=7, color=MUTED)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.6, 1.15)
    fig.tight_layout(pad=0.2)
    return fig


def anomaly_scatter(df: pd.DataFrame, anomalies: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 4))
    _style(fig, [ax])

    anomaly_idx = set() if len(anomalies) == 0 else set(anomalies.index)
    normal = df[~df.index.isin(anomaly_idx)]

    ax.scatter(normal["date"], normal["amount"],
               color=ACCENT, s=25, alpha=0.65, linewidths=0, label="Normal", zorder=3)

    if len(anomalies) > 0:
        ax.scatter(anomalies["date"], anomalies["amount"],
                   color="#FF6584", s=70, marker="X", zorder=5,
                   linewidths=0.5, edgecolors="white", label="Anomaly")

    ax.set_ylabel("Amount (₹)")
    ax.set_title("Transaction Anomalies", color=TEXT, fontsize=12)
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT)
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    return fig


def fig_to_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.read()
