"""
train_model.py
Trains a TF-IDF + RandomForest pipeline for expense categorization.
Run: python train_model.py
"""

import os
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report
from utils.data_loader import load_training_data

MODEL_PATH = "models/expense_classifier.pkl"
TRAINING_DATA_PATH = "data/training_data.csv"

CATEGORIES = ["Food", "Rent", "Transport", "Shopping", "Bills", "Others"]

# Augmented keyword rules used to enrich training set
KEYWORD_RULES = {
    "Food": [
        "swiggy", "zomato", "mcdonalds", "dominos", "kfc", "burger", "pizza",
        "subway", "restaurant", "food", "lunch", "dinner", "breakfast",
        "grocery", "bigbasket", "blinkit", "zepto", "dunzo", "dmart",
        "reliance fresh", "more supermarket", "big bazaar", "canteen", "chai",
        "snacks", "cafe", "coffee", "bakery", "hotel", "dhaba",
    ],
    "Rent": [
        "rent", "pg accommodation", "house rent", "landlord", "flat rent",
        "room rent", "monthly rent", "accommodation",
    ],
    "Transport": [
        "ola", "uber", "rapido", "auto", "rickshaw", "petrol", "metro",
        "bus pass", "train", "flight", "parking", "toll", "cab", "taxi",
        "rapido", "city bus", "airport", "fuel",
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
        "emi", "loan", "postpaid",
    ],
}


def rule_based_label(description: str) -> str:
    desc = description.lower()
    for category, keywords in KEYWORD_RULES.items():
        if any(kw in desc for kw in keywords):
            return category
    return "Others"


def augment_training_data(descriptions: list, labels: list) -> tuple[list, list]:
    """Add synthetic examples to improve generalization."""
    extras_desc = [
        "online food order", "meal delivery app", "restaurant bill",
        "house monthly rent", "apartment rent", "room booking",
        "fuel station", "cab booking", "train ticket booking",
        "online store purchase", "ecommerce order", "clothing purchase",
        "utility payment", "subscription renewal", "recharge prepaid",
        "atm withdrawal", "bank transfer", "salary credit",
        "medical expense", "hospital bill", "pharmacy purchase",
        "gym membership", "fitness class", "yoga subscription",
    ]
    extras_label = [
        "Food", "Food", "Food",
        "Rent", "Rent", "Rent",
        "Transport", "Transport", "Transport",
        "Shopping", "Shopping", "Shopping",
        "Bills", "Bills", "Bills",
        "Others", "Others", "Others",
        "Others", "Others", "Others",
        "Others", "Others", "Others",
    ]
    return descriptions + extras_desc, labels + extras_label


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True,
            min_df=1,
        )),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            random_state=42,
            n_jobs=-1,
        )),
    ])


def train_and_save():
    print("Loading training data...")
    descriptions, labels = load_training_data(TRAINING_DATA_PATH)
    descriptions, labels = augment_training_data(descriptions, labels)

    print(f"Total samples: {len(descriptions)}")
    print(f"Category distribution: {pd.Series(labels).value_counts().to_dict()}")

    pipeline = build_pipeline()

    print("\nCross-validating...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, descriptions, labels, cv=cv, scoring="accuracy")
    print(f"CV Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    print("\nTraining final model on all data...")
    pipeline.fit(descriptions, labels)

    # Quick evaluation on training set
    preds = pipeline.predict(descriptions)
    print("\nTraining Set Report:")
    print(classification_report(labels, preds))

    os.makedirs("models", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"\nModel saved to: {MODEL_PATH}")
    return pipeline


if __name__ == "__main__":
    train_and_save()
