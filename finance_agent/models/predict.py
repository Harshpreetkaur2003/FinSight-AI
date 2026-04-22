"""
models/predict.py
Loads saved pipeline and predicts expense categories.
Falls back to rule-based labeling if model is missing.
"""

import os
import pickle
import re
from typing import Union
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(__file__), "expense_classifier.pkl")

KEYWORD_RULES = {
    "Food": [
        "swiggy", "zomato", "mcdonalds", "dominos", "kfc", "burger", "pizza",
        "subway", "restaurant", "food", "lunch", "dinner", "breakfast",
        "grocery", "bigbasket", "blinkit", "zepto", "dunzo", "dmart",
        "reliance fresh", "more supermarket", "big bazaar", "canteen", "chai",
        "snacks", "cafe", "coffee", "bakery", "hotel", "dhaba", "biryani",
    ],
    "Rent": [
        "rent", "pg accommodation", "house rent", "landlord", "flat rent",
        "room rent", "monthly rent", "accommodation",
    ],
    "Transport": [
        "ola", "uber", "rapido", "auto", "rickshaw", "petrol", "metro",
        "bus pass", "train", "flight", "parking", "toll", "cab", "taxi",
        "city bus", "airport", "fuel", "rapido", "carpool",
    ],
    "Shopping": [
        "amazon", "flipkart", "myntra", "meesho", "ajio", "nykaa",
        "bewakoof", "apple", "samsung", "electronics", "clothes", "fashion",
        "shoes", "gadget", "accessories", "sale", "shopping",
    ],
    "Bills": [
        "electricity", "water", "gas", "lpg", "internet", "broadband",
        "mobile", "phone bill", "netflix", "spotify", "dth", "subscription",
        "jio", "airtel", "vodafone", "bsnl", "recharge", "insurance",
        "emi", "loan", "postpaid", "prime", "hotstar",
    ],
}

_model = None


def _load_model():
    global _model
    if _model is None and os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
    return _model


def _clean(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text)


def rule_based_predict(description: str) -> str:
    desc = _clean(description)
    for category, keywords in KEYWORD_RULES.items():
        if any(kw in desc for kw in keywords):
            return category
    return "Others"


def predict_category(description: Union[str, list]) -> Union[str, list]:
    """
    Predict category for a single string or list of strings.
    Uses ML model if available, else falls back to rule-based logic.
    """
    single = isinstance(description, str)
    items = [description] if single else description

    model = _load_model()
    if model is not None:
        cleaned = [_clean(d) for d in items]
        preds = model.predict(cleaned).tolist()
    else:
        preds = [rule_based_predict(d) for d in items]

    return preds[0] if single else preds


def predict_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'category' column to transaction DataFrame."""
    df = df.copy()
    df["category"] = predict_category(df["description"].tolist())
    return df


def get_confidence(descriptions: list) -> list[dict]:
    """Return category probabilities for each description."""
    model = _load_model()
    if model is None:
        return [{"category": rule_based_predict(d), "confidence": 1.0} for d in descriptions]

    cleaned = [_clean(d) for d in descriptions]
    probs = model.predict_proba(cleaned)
    classes = model.classes_
    results = []
    for p in probs:
        top_idx = p.argmax()
        results.append({
            "category": classes[top_idx],
            "confidence": round(float(p[top_idx]), 3),
            "all_probs": {c: round(float(v), 3) for c, v in zip(classes, p)},
        })
    return results
