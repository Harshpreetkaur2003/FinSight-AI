"""
app.py
Expense Intelligence & Financial Health AI Agent — Streamlit UI
Run: streamlit run app.py
"""

import os
import sys
import io
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from utils.data_loader import load_transactions
from models.predict import predict_dataframe
from agent.analyzer import full_analysis_report, detect_anomalies, compute_risk_score
from agent.rules import generate_advice, generate_summary_text
from agent.llm_agent import FinancialAgent
from utils.charts import (
    pie_chart, bar_chart_monthly, category_trend_bar,
    spending_gauge, anomaly_scatter,
)
from utils.export import export_csv, export_excel, export_text_report

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinSight AI",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=Space+Mono&display=swap');

* {
    font-family: 'Sora', sans-serif !important;
    box-sizing: border-box;
}

html, body, [class*="css"] {
    background-color: #0D0F1A !important;
    color: #E8EAF6;
}

/* Main container */
.block-container {
    padding-top: 1rem !important;
    padding-bottom: 2rem !important;
    max-width: 100% !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#13162B 0%,#0D0F1A 100%) !important;
    border-right: 1px solid rgba(108,99,255,0.3);
}

[data-testid="stSidebar"] * {
    color: #E8EAF6 !important;
}

/* Metric Cards */
.metric-card {
    background: linear-gradient(135deg,#1A1D2E 0%,#13162B 100%);
    border: 1px solid rgba(108,99,255,0.25);
    border-radius: 16px;
    padding: 18px 14px;
    text-align: center;
    margin-bottom: 10px;
    min-height: 130px;

    display: flex;
    flex-direction: column;
    justify-content: center;

    overflow-wrap: break-word;
    word-wrap: break-word;
}

.metric-value {
    font-size: clamp(1rem, 2vw, 1.6rem);
    font-weight: 700;
    line-height: 1.3;

    background: linear-gradient(90deg,#6C63FF,#FF6584);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;

    white-space: normal;
    overflow-wrap: break-word;
}

.metric-label {
    font-size: 0.72rem;
    color: #8B9CBF;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 8px;

    white-space: normal;
}

.metric-delta {
    font-size: 0.72rem;
    color: #8B9CBF;
    margin-top: 6px;

    white-space: normal;
    line-height: 1.4;
}

/* Chat bubbles */
.chat-user {
    background: linear-gradient(135deg,#2D2060,#1A1D2E);
    border: 1px solid rgba(108,99,255,0.4);
    border-radius: 18px 18px 4px 18px;
    padding: 12px 16px;
    margin: 8px 0 8px 8%;
    font-size: 0.92rem;

    width: auto;
    max-width: 92%;

    overflow-wrap: break-word;
    word-break: break-word;
}

.chat-ai {
    background: linear-gradient(135deg,#1A2A1A,#1A1D2E);
    border: 1px solid rgba(67,185,127,0.35);
    border-radius: 18px 18px 18px 4px;
    padding: 12px 16px;
    margin: 8px 8% 8px 0;

    font-size: 0.92rem;
    white-space: pre-wrap;

    width: auto;
    max-width: 92%;

    overflow-wrap: break-word;
    word-break: break-word;
}

.chat-sender {
    font-size: 0.7rem;
    color: #8B9CBF;
    margin-bottom: 4px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* Advice cards */
.advice-HIGH {
    border-left:4px solid #FF6584;
    background:rgba(255,101,132,0.08);
    border-radius:0 12px 12px 0;
    padding:12px 16px;
    margin:8px 0;
}

.advice-MEDIUM {
    border-left:4px solid #F7B731;
    background:rgba(247,183,49,0.08);
    border-radius:0 12px 12px 0;
    padding:12px 16px;
    margin:8px 0;
}

.advice-LOW {
    border-left:4px solid #43B97F;
    background:rgba(67,185,127,0.08);
    border-radius:0 12px 12px 0;
    padding:12px 16px;
    margin:8px 0;
}

/* Section titles */
.section-title {
    font-size: 1.05rem;
    font-weight: 700;
    color:#E8EAF6;
    margin-bottom:12px;
    padding-bottom:8px;
    border-bottom:1px solid rgba(108,99,255,0.3);

    overflow-wrap: break-word;
}

/* Inputs */
.stTextInput input,
.stTextArea textarea {
    background-color: #1A1D2E !important;
    color: #E8EAF6 !important;
    border: 1px solid rgba(108,99,255,0.4) !important;
    border-radius: 10px !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg,#6C63FF,#5550CC) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;

    width: 100%;
    min-height: 42px;

    overflow-wrap: break-word;
    white-space: normal;
}

.stDownloadButton > button {
    background: linear-gradient(135deg,#43B97F,#2E9060) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;

    width: 100%;
    min-height: 42px;

    white-space: normal;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    flex-wrap: wrap;
}

.stTabs [data-baseweb="tab"] {
    color: #8B9CBF !important;
    font-weight: 600;
    border-radius: 10px;
    padding: 10px 14px;

    white-space: nowrap;
}

.stTabs [aria-selected="true"] {
    color: #6C63FF !important;
    border-bottom: 2px solid #6C63FF !important;
}

/* Dataframes */
div[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}

/* Prevent chart overflow */
canvas {
    max-width: 100% !important;
}

/* Mobile responsiveness */
@media screen and (max-width: 768px) {

    .metric-card {
        min-height: auto;
        padding: 14px 10px;
    }

    .metric-value {
        font-size: 1.1rem;
    }

    .chat-user,
    .chat-ai {
        margin-left: 0 !important;
        margin-right: 0 !important;
        max-width: 100%;
    }

    .section-title {
        font-size: 0.95rem;
    }

    .stTabs [data-baseweb="tab"] {
        font-size: 0.78rem;
        padding: 8px 10px;
    }
}
</style>
""", unsafe_allow_html=True)


# ── Session State ──────────────────────────────────────────────────────────────
def _init():
    for k, v in {
        "df": None, "report": None,
        "agent": FinancialAgent(os.getenv("OPENAI_API_KEY", "")),
        "chat_history": [], "data_loaded": False,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:20px 0 10px'>
        <div style='font-size:2.5rem'>💰</div>
        <div style='font-size:1.3rem;font-weight:700;
                    background:linear-gradient(90deg,#6C63FF,#FF6584);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                    background-clip:text;'>FinSight AI</div>
        <div style='font-size:0.75rem;color:#8B9CBF;margin-top:4px;'>
            Expense Intelligence Agent</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown("### 📂 Upload Transactions")
    uploaded = st.file_uploader(
        "CSV (date, amount, description)",
        type=["csv"], label_visibility="collapsed",
    )

    if st.button("📊 Load Sample Data", use_container_width=True):
        path = os.path.join(os.path.dirname(__file__), "data", "sample_transactions.csv")
        if os.path.exists(path):
            with open(path, "rb") as f:
                buf = io.BytesIO(f.read())
                buf.name = "sample_transactions.csv"
                uploaded = buf

    st.divider()
    st.markdown("### ⚙️ Settings")
    income = st.number_input("Monthly Income (₹)", min_value=0, value=50000, step=1000)
    openai_key = st.text_input("OpenAI API Key (optional)", type="password", placeholder="sk-...")
    if openai_key and openai_key != st.session_state.agent.api_key:
        st.session_state.agent = FinancialAgent(openai_api_key=openai_key)

    st.divider()
    if st.session_state.data_loaded:
        st.markdown("### 📥 Export")
        df_e = st.session_state.df
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("📄 CSV", data=export_csv(df_e),
                file_name="transactions.csv", mime="text/csv", use_container_width=True)
        with c2:
            st.download_button("📊 Excel", data=export_excel(df_e),
                file_name="finsight_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True)
        st.download_button("📋 Text Report", data=export_text_report(df_e).encode(),
            file_name="financial_report.txt", mime="text/plain", use_container_width=True)

    st.divider()
    st.markdown("<div style='font-size:0.7rem;color:#555;text-align:center;'>"
                "FinSight AI v1.0 | Built with ❤️</div>", unsafe_allow_html=True)


# ── Data Loading ───────────────────────────────────────────────────────────────
if uploaded is not None and not st.session_state.data_loaded:
    with st.spinner("🔄 Processing transactions..."):
        try:
            model_path = os.path.join(os.path.dirname(__file__), "models", "expense_classifier.pkl")
            if not os.path.exists(model_path):
                from train_model import train_and_save
                train_and_save()

            df = load_transactions(uploaded)
            df = predict_dataframe(df)
            report = full_analysis_report(df)

            st.session_state.df = df
            st.session_state.report = report
            st.session_state.data_loaded = True
            st.session_state.chat_history = []
            st.session_state.agent.load_data(df, income if income > 0 else None)
            st.success(f"✅ Loaded {len(df)} transactions across {df['month'].nunique()} month(s)")
        except Exception as e:
            st.error(f"❌ {e}")
            st.exception(e)


# ── Landing page ───────────────────────────────────────────────────────────────
if not st.session_state.data_loaded:
    st.markdown("""
    <div style='text-align:center;padding:80px 20px 40px'>
        <div style='font-size:4rem;margin-bottom:16px'>💰</div>
        <h1 style='font-size:2.5rem;font-weight:700;
                   background:linear-gradient(90deg,#6C63FF,#FF6584);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                   background-clip:text;'>FinSight AI</h1>
        <p style='color:#8B9CBF;font-size:1.1rem;max-width:480px;margin:0 auto;'>
            Upload your transaction CSV or load sample data<br>
            for AI-powered financial insights and advice.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    for col, icon, title, desc in [
        (c1, "🤖", "ML Categorization", "Auto-categorizes every transaction"),
        (c2, "📊", "Smart Analytics", "Trends, anomalies, overspending"),
        (c3, "💬", "Chat Advisor", "Ask anything about your finances"),
    ]:
        with col:
            st.markdown(f"""<div class='metric-card'>
                <div style='font-size:2rem;margin-bottom:8px'>{icon}</div>
                <div style='font-weight:700;font-size:0.95rem'>{title}</div>
                <div style='color:#8B9CBF;font-size:0.8rem;margin-top:6px'>{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("""<div style='text-align:center;margin-top:32px;color:#8B9CBF;font-size:0.85rem'>
        📁 Expected CSV format: <code style='color:#6C63FF'>date, amount, description</code>
    </div>""", unsafe_allow_html=True)
    st.stop()


# ── Main Dashboard ─────────────────────────────────────────────────────────────
df     = st.session_state.df
report = st.session_state.report

tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🔍 Analysis", "⚠️ Alerts", "💬 AI Advisor"])


# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    savings  = report["savings"]
    risk     = report["risk_score"]
    trend    = report["spending_trend"]
    total_sp = sum(v["total"] for v in report["monthly_summary"])
    n_anoms  = len(report["anomalies"])

    # KPI row
    cols = st.columns(5)
    metrics = [
        ("💸 Total Spend",    f"₹{total_sp:,.0f}",                       None),
        ("📅 Avg Monthly",    f"₹{savings['avg_monthly_spend']:,.0f}",    None),
        ("💰 Est. Savings",   f"₹{savings['estimated_savings']:,.0f}",    f"{savings['savings_rate_pct']}% rate"),
        ("📈 Trend",          trend["direction"].title(),                  f"{trend['pct_change']:+.1f}% vs last month"),
        ("⚠️ Anomalies",      str(n_anoms),                               "flagged transactions"),
    ]
    for col, (label, val, delta) in zip(cols, metrics):
        with col:
            d_html = f"<div class='metric-delta'>{delta}</div>" if delta else ""
            st.markdown(f"""<div class='metric-card'>
                <div class='metric-value'>{val}</div>
                <div class='metric-label'>{label}</div>{d_html}
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts row 1: pie + gauge
    c_pie, c_gauge = st.columns([3, 1])
    with c_pie:
        st.markdown("<div class='section-title'>🥧 Spending by Category</div>", unsafe_allow_html=True)
        st.pyplot(pie_chart(df), use_container_width=True)
    with c_gauge:
        st.markdown("<div class='section-title'>🎯 Risk Score</div>", unsafe_allow_html=True)
        st.pyplot(spending_gauge(risk["score"]), use_container_width=True)
        lvl_color = {"Low": "#43B97F", "Medium": "#F7B731", "High": "#FF6584"}.get(risk["level"], "#999")
        st.markdown(f"""<div style='text-align:center;padding:8px;
                        color:{lvl_color};font-weight:700;font-size:0.85rem;'>
            {risk["level"]} Risk<br>
            <span style='font-size:0.73rem;color:#8B9CBF;font-weight:400;'>
            {risk["interpretation"][:75]}...</span>
        </div>""", unsafe_allow_html=True)

    # Charts row 2: monthly bar + stacked
    c_bar, c_stack = st.columns(2)
    with c_bar:
        st.markdown("<div class='section-title'>📅 Monthly Spending</div>", unsafe_allow_html=True)
        st.pyplot(bar_chart_monthly(df), use_container_width=True)
    with c_stack:
        st.markdown("<div class='section-title'>📊 Category Trends</div>", unsafe_allow_html=True)
        st.pyplot(category_trend_bar(df), use_container_width=True)

    # Proactive insights
    insights = st.session_state.agent.get_proactive_insights()
    if insights:
        st.markdown("<div class='section-title'>💡 Key Insights</div>", unsafe_allow_html=True)
        for ins in insights:
            st.markdown(ins)


# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-title'>📋 Transaction Data</div>", unsafe_allow_html=True)

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        cats = ["All"] + sorted(df["category"].unique().tolist())
        sel_cat = st.selectbox("Category", cats)
    with fc2:
        months = ["All"] + sorted(df["month_str"].unique().tolist())
        sel_month = st.selectbox("Month", months)
    with fc3:
        sort_by = st.selectbox("Sort by", ["date", "amount", "category"])

    disp = df.copy()
    if sel_cat   != "All": disp = disp[disp["category"]  == sel_cat]
    if sel_month != "All": disp = disp[disp["month_str"] == sel_month]
    disp = disp.sort_values(sort_by, ascending=(sort_by != "amount"))
    st.dataframe(disp[["date","description","amount","category"]].reset_index(drop=True),
                 use_container_width=True, height=320)

    st.markdown("<br><div class='section-title'>🔴 Overspending Analysis</div>", unsafe_allow_html=True)
    rows = []
    for cat, v in report["overspending"].items():
        rows.append({
            "Category": cat,
            "Spent (₹)": f"₹{v['actual_amount']:,.0f}",
            "Spend %":   f"{v['actual_pct']}%",
            "Budget %":  f"{v['budget_pct']}%",
            "Status":    "🔴 Over" if v["overspending"] else "✅ OK",
            "Excess":    f"+{v['excess_pct']}%" if v["overspending"] else "—",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("<br><div class='section-title'>📊 Category Deep Dive</div>", unsafe_allow_html=True)
    cat_sel = st.selectbox("Select Category", sorted(df["category"].unique()), key="dive")
    cdf = df[df["category"] == cat_sel]
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Spent",    f"₹{cdf['amount'].sum():,.0f}")
    m2.metric("Transactions",   len(cdf))
    m3.metric("Avg Transaction",f"₹{cdf['amount'].mean():,.0f}")
    st.dataframe(
        cdf[["date","description","amount","category"]].sort_values("amount", ascending=False).reset_index(drop=True),
        use_container_width=True, height=240,
    )


# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-title'>⚠️ Anomaly Detection</div>", unsafe_allow_html=True)

    anomalies = detect_anomalies(df)
    c_sc, c_info = st.columns([3, 1])
    with c_sc:
        st.pyplot(anomaly_scatter(df, anomalies), use_container_width=True)
    with c_info:
        st.metric("Anomalies Found", len(anomalies))
        if len(anomalies) > 0:
            st.markdown("**Top Anomalies:**")
            for _, row in anomalies.sort_values("amount", ascending=False).head(3).iterrows():
                st.markdown(f"""<div style='background:rgba(255,101,132,0.1);
                    border:1px solid rgba(255,101,132,0.3);border-radius:8px;
                    padding:8px;margin:5px 0;font-size:0.8rem;'>
                    <b>₹{row['amount']:,.0f}</b><br>
                    <span style='color:#8B9CBF'>{row['description'][:28]}</span><br>
                    <span style='color:#FF6584'>Z={row['z_score']:.2f}</span>
                </div>""", unsafe_allow_html=True)
        else:
            st.success("No anomalies ✅")

    if len(anomalies) > 0:
        st.markdown("**All Anomalous Transactions:**")
        st.dataframe(anomalies.reset_index(drop=True), use_container_width=True)

    spikes = report["spending_spikes"]
    st.markdown("<br><div class='section-title'>⚡ Spending Spikes</div>", unsafe_allow_html=True)
    if spikes:
        st.dataframe(pd.DataFrame(spikes), use_container_width=True, hide_index=True)
    else:
        st.info("No unusual spending spikes detected.")

    st.markdown("<br><div class='section-title'>💡 Recommendations</div>", unsafe_allow_html=True)
    advice_list = generate_advice(df, income if income > 0 else None)
    if not advice_list:
        st.success("🎉 Your finances look healthy! No major issues detected.")
    for a in advice_list:
        icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(a["priority"], "ℹ️")
        st.markdown(f"""<div class='advice-{a["priority"]}'>
            <div style='font-weight:700;font-size:0.9rem;'>
                {icon} [{a["priority"]}] {a["category"]}</div>
            <div style='margin:4px 0 2px'>{a["advice"]}</div>
            <div style='font-size:0.8rem;color:#8B9CBF'>{a["reason"]}</div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    llm_on = st.session_state.agent.use_llm
    badge = (
        '<span style="background:rgba(67,185,127,0.2);border:1px solid #43B97F;'
        'color:#43B97F;border-radius:20px;padding:2px 10px;font-size:0.75rem;">GPT Mode</span>'
        if llm_on else
        '<span style="background:rgba(108,99,255,0.2);border:1px solid #6C63FF;'
        'color:#6C63FF;border-radius:20px;padding:2px 10px;font-size:0.75rem;">Rule-Based AI</span>'
    )
    st.markdown(f"<div class='section-title'>💬 AI Financial Advisor &nbsp;{badge}</div>",
                unsafe_allow_html=True)

    # Quick buttons
    st.markdown("**Quick Questions:**")
    qcols = st.columns(4)
    quick_qs = ["Where am I overspending?", "How can I save money?",
                "Analyze my expenses", "What is my risk score?"]
    triggered = None
    for col, q in zip(qcols, quick_qs):
        with col:
            if st.button(q, key=f"q_{q}", use_container_width=True):
                triggered = q

    # Chat history
    if not st.session_state.chat_history:
        st.markdown("""<div style='text-align:center;padding:40px;color:#8B9CBF;'>
            <div style='font-size:2rem;margin-bottom:10px'>🤖</div>
            Hi! I'm your AI financial advisor.<br>Ask me anything about your expenses.
        </div>""", unsafe_allow_html=True)
    else:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"""<div class='chat-user'>
                    <div class='chat-sender'>You</div>{msg['content']}
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class='chat-ai'>
                    <div class='chat-sender'>FinSight AI 🤖</div>{msg['content']}
                </div>""", unsafe_allow_html=True)

    # Input form
    with st.form("chat_form", clear_on_submit=True):
        ci, cb = st.columns([5, 1])
        with ci:
            user_input = st.text_input("Message", placeholder="Ask about your finances...",
                                       label_visibility="collapsed")
        with cb:
            submitted = st.form_submit_button("Send 🚀", use_container_width=True)

    query = triggered or (user_input.strip() if submitted and user_input.strip() else None)
    if query:
        with st.spinner("🤔 Thinking..."):
            reply = st.session_state.agent.chat(query)
        st.session_state.chat_history.append({"role": "user", "content": query})
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.rerun()

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.session_state.agent.reset_history()
            st.rerun()
