"""
ml/clustering.py
----------------
Unsupervised KMeans clustering of complaint text.
Groups complaints into meaningful patterns automatically.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
import numpy as np
import re

# Pre-defined cluster labels (named after discovering centroids on seed data)
CLUSTER_NAMES = {
    0: "Size & Fit Issues",
    1: "Product Quality & Damage",
    2: "Delivery & Shipping Issues",
    3: "Wrong Item / Order Errors",
}

CLUSTER_ICONS = {
    0: "📏",
    1: "🔨",
    2: "🚚",
    3: "📦",
}

# Seed complaints to bootstrap the model (extends with real data)
_SEED_COMPLAINTS = [
    "shoes are too small, size 9 feels like size 8",
    "size chart is wrong, ordered size 10 but it doesn't fit",
    "the shoe is very tight, width is narrow, wrong size",
    "size mismatch between left and right shoe",
    "the size I ordered doesn't match what arrived",

    "sole completely detached after one day of use",
    "stitching came apart, very poor build quality",
    "shoe broke at the seam within a week",
    "material is fake and cheap, peeled off",
    "heel cushion collapsed, very poor quality",

    "order arrived 3 weeks late, tracking was wrong",
    "package was damaged during shipping",
    "delivery delayed multiple times with no updates",
    "shoes were wet and damaged when delivered",
    "shipment stuck in transit for two weeks",

    "received completely wrong model, not what I ordered",
    "wrong color was sent, ordered black got brown",
    "different brand was shipped instead",
    "wrong quantity in package, missing one shoe",
    "label says size 9 but shoes inside are size 7",
]

K = 4

_tfidf = TfidfVectorizer(
    ngram_range=(1, 2), max_features=500,
    stop_words="english", lowercase=True,
)

_kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)

def _fit(texts):
    X = _tfidf.fit_transform(texts)
    _kmeans.fit(X)

_fit(_SEED_COMPLAINTS)


def cluster_complaints(texts: list[str]) -> dict:
    """
    Cluster a list of complaint strings.
    Returns cluster assignments and summary per cluster.

    If fewer than K samples, falls back to seed data + provided texts.
    """
    if not texts:
        return {"clusters": [], "cluster_summary": {}}

    all_texts = _SEED_COMPLAINTS + texts
    X_all = _tfidf.transform(all_texts)

    if len(texts) >= K:
        _kmeans.fit(X_all)

    labels = _kmeans.predict(X_all)
    real_labels = labels[len(_SEED_COMPLAINTS):]  # only real complaint assignments

    clusters = {}
    for idx, (text, label) in enumerate(zip(texts, real_labels)):
        label = int(label)
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(text)

    summary = {}
    for label, items in clusters.items():
        name = CLUSTER_NAMES.get(label, f"Category {label}")
        icon = CLUSTER_ICONS.get(label, "📌")
        summary[label] = {
            "name":  name,
            "icon":  icon,
            "count": len(items),
            "samples": items[:3],
        }

    # Ensure all K clusters exist in output even if empty
    for k in range(K):
        if k not in summary:
            summary[k] = {
                "name":    CLUSTER_NAMES.get(k, f"Category {k}"),
                "icon":    CLUSTER_ICONS.get(k, "📌"),
                "count":   0,
                "samples": [],
            }

    return {
        "clusters": [int(l) for l in real_labels],
        "cluster_summary": summary,
    }


def predict_cluster(text: str) -> int:
    """Predict cluster for a single complaint text."""
    X = _tfidf.transform([text])
    return int(_kmeans.predict(X)[0])


if __name__ == "__main__":
    tests = [
        "The sole fell off after two days",
        "Wrong size delivered, size chart is incorrect",
        "Package arrived damaged and very late",
        "Received completely wrong product",
    ]
    result = cluster_complaints(tests)
    for label, info in result["cluster_summary"].items():
        print(f"{info['icon']} Cluster {label}: {info['name']} — {info['count']} items")