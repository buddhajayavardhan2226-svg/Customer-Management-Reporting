"""
ml/churn.py
-----------
Logistic Regression churn predictor.
Features: complaint_count, negative_sentiment_count, return_requests, low_rating_count
Output: Low Risk / Medium Risk / High Risk
"""

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

# ── Synthetic training data ────────────────────────────────────────────────────
# [complaints, neg_sentiments, returns, low_ratings(1-2)]  → label
_X_TRAIN = np.array([
    # Low risk
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [1, 0, 0, 1],
    [0, 0, 1, 0],
    [1, 1, 0, 0],
    [0, 0, 0, 2],

    # Medium risk
    [2, 1, 0, 1],
    [1, 2, 1, 0],
    [2, 2, 0, 0],
    [1, 1, 1, 1],
    [3, 0, 0, 0],
    [0, 2, 2, 1],
    [2, 0, 1, 2],
    [1, 3, 0, 1],

    # High risk
    [4, 3, 2, 2],
    [5, 4, 1, 3],
    [3, 3, 3, 2],
    [6, 2, 2, 4],
    [4, 5, 3, 1],
    [5, 3, 4, 3],
    [3, 4, 2, 3],
    [7, 3, 1, 4],
], dtype=float)

_Y_TRAIN = (
    ["low"] * 8 +
    ["medium"] * 8 +
    ["high"] * 8
)

_churn_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression(
        solver="lbfgs",
        max_iter=500, random_state=42
    )),
])
_churn_pipeline.fit(_X_TRAIN, _Y_TRAIN)

RISK_COLOURS = {
    "low":    "#43e97b",
    "medium": "#ffa500",
    "high":   "#ff4757",
}

RISK_THRESHOLDS = {
    "low":    "0–1 complaints, minimal negative interactions",
    "medium": "2–3 complaints or multiple returns",
    "high":   "4+ complaints, repeated negative sentiment",
}


def predict_churn(
    complaint_count: int,
    negative_sentiment_count: int,
    return_requests: int,
    low_rating_count: int,
) -> dict:
    """
    Returns: { risk, confidence, color, features }
    """
    X = np.array([[complaint_count, negative_sentiment_count,
                   return_requests, low_rating_count]], dtype=float)

    proba  = _churn_pipeline.predict_proba(X)[0]
    labels = _churn_pipeline.classes_
    idx    = int(np.argmax(proba))
    risk   = labels[idx]
    conf   = float(proba[idx])

    return {
        "risk":        risk,
        "confidence":  round(conf, 3),
        "color":       RISK_COLOURS[risk],
        "all_proba":   {l: round(float(p), 3) for l, p in zip(labels, proba)},
        "features": {
            "complaints":   complaint_count,
            "neg_sentiment": negative_sentiment_count,
            "returns":      return_requests,
            "low_ratings":  low_rating_count,
        },
    }


def batch_predict(sessions: list[dict]) -> list[dict]:
    """
    sessions: list of dicts with keys:
        session_id, complaint_count, negative_sentiment_count, return_requests, low_rating_count
    Returns: list of { session_id, risk, confidence }
    """
    results = []
    for s in sessions:
        r = predict_churn(
            s.get("complaint_count", 0),
            s.get("negative_sentiment_count", 0),
            s.get("return_requests", 0),
            s.get("low_rating_count", 0),
        )
        results.append({"session_id": s["session_id"], **r})
    return results


if __name__ == "__main__":
    cases = [
        (0, 0, 0, 0, "Ideal customer"),
        (1, 1, 0, 0, "Slight concern"),
        (2, 2, 1, 1, "At risk"),
        (5, 4, 3, 3, "High churn risk"),
    ]
    for c, n, r, lr, label in cases:
        res = predict_churn(c, n, r, lr)
        print(f"[{res['risk'].upper():6s}] {res['confidence']:.2f} | {label}")