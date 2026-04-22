"""
agent/llm_agent.py
LangChain-based conversational financial advisor.
Falls back to rule-based engine if OpenAI key is not set.
"""

import os
import json
import pandas as pd
from typing import Optional

from agent.analyzer import full_analysis_report
from agent.rules import generate_advice, generate_summary_text, answer_question_rules

# ── LangChain / OpenAI imports (optional) ────────────────────────────────────
try:
    from langchain_openai import ChatOpenAI
    from langchain.schema import HumanMessage, SystemMessage, AIMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


SYSTEM_PROMPT = """You are an expert AI financial advisor called "FinSight AI".
You have access to the user's complete expense analysis report.
Your role is to:
1. Answer questions about their spending patterns clearly and concisely
2. Identify overspending and financial risks
3. Provide actionable, specific advice — not vague platitudes
4. Be empathetic but direct
5. Use ₹ (Indian Rupee) for currency
6. Keep responses under 200 words unless a detailed report is requested

Always ground your answers in the actual data provided. Do not fabricate numbers.
Format responses with bullet points where appropriate for readability."""


def _build_context(df: pd.DataFrame, income: float = None) -> str:
    """Serialize analysis report into a compact context string for the LLM."""
    report = full_analysis_report(df)
    advice = generate_advice(df, income)

    ctx = {
        "monthly_spend_summary": report["monthly_summary"],
        "category_totals": report["category_totals"],
        "spending_trend": report["spending_trend"],
        "category_trends": report["category_trends"],
        "overspending_analysis": report["overspending"],
        "savings_estimate": report["savings"],
        "risk_score": report["risk_score"],
        "anomalies_detected": len(report["anomalies"]),
        "spending_spikes": report["spending_spikes"],
        "top_recommendations": [
            {"priority": a["priority"], "category": a["category"], "advice": a["advice"]}
            for a in advice[:5]
        ],
    }
    return json.dumps(ctx, indent=2, default=str)


class FinancialAgent:
    """
    Conversational financial AI agent.
    Uses OpenAI LLM if API key is set, otherwise uses rule-based engine.
    """

    def __init__(self, openai_api_key: Optional[str] = None):
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
        self.use_llm = LANGCHAIN_AVAILABLE and bool(self.api_key)
        self._llm = None
        self._chat_history: list = []
        self._df: Optional[pd.DataFrame] = None
        self._income: Optional[float] = None
        self._context_str: str = ""

        if self.use_llm:
            self._llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.4,
                api_key=self.api_key,
                max_tokens=400,
            )

    def load_data(self, df: pd.DataFrame, income: float = None):
        """Load transaction data and pre-compute context."""
        self._df = df
        self._income = income
        self._context_str = _build_context(df, income)
        self._chat_history = []

    @property
    def is_ready(self) -> bool:
        return self._df is not None

    def chat(self, user_message: str) -> str:
        """Process a user message and return AI response."""
        if not self.is_ready:
            return "Please upload your transaction data first before asking questions."

        if self.use_llm:
            return self._llm_response(user_message)
        return self._rule_response(user_message)

    def _rule_response(self, user_message: str) -> str:
        """Fallback rule-based response."""
        return answer_question_rules(user_message, self._df)

    def _llm_response(self, user_message: str) -> str:
        """LangChain OpenAI response with conversation memory."""
        try:
            system_msg = SystemMessage(content=(
                f"{SYSTEM_PROMPT}\n\n"
                f"=== FINANCIAL DATA CONTEXT ===\n{self._context_str}\n"
                f"=== END CONTEXT ==="
            ))

            # Build message list
            messages = [system_msg]
            for h in self._chat_history[-6:]:  # keep last 3 turns
                if h["role"] == "user":
                    messages.append(HumanMessage(content=h["content"]))
                else:
                    messages.append(AIMessage(content=h["content"]))
            messages.append(HumanMessage(content=user_message))

            response = self._llm.invoke(messages)
            assistant_reply = response.content

            # Update history
            self._chat_history.append({"role": "user", "content": user_message})
            self._chat_history.append({"role": "assistant", "content": assistant_reply})

            return assistant_reply

        except Exception as e:
            # Graceful fallback
            return (
                f"⚠️ LLM unavailable ({type(e).__name__}). Using rule-based engine:\n\n"
                + answer_question_rules(user_message, self._df)
            )

    def reset_history(self):
        self._chat_history = []

    def get_proactive_insights(self) -> list[str]:
        """Return 3 proactive insights to display on dashboard load."""
        if not self.is_ready:
            return []

        advice = generate_advice(self._df, self._income)
        insights = []
        for a in advice[:3]:
            emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(a["priority"], "ℹ️")
            insights.append(f"{emoji} **{a['category']}**: {a['advice']}\n  _{a['reason']}_")
        return insights
