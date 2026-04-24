"""
ml/sentiment.py
---------------
Sentiment analysis for customer messages.
Hybrid approach: keyword scoring + Naive Bayes classifier.
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np

# ── Training data ──────────────────────────────────────────────────────────────
_SENT_DATA = [
    ("absolutely terrible quality broke immediately",      "negative"),
    ("very disappointed with this purchase",               "negative"),
    ("the shoes fell apart after one day",                 "negative"),
    ("worst product I have ever bought",                   "negative"),
    ("extremely unhappy with the service",                 "negative"),
    ("damaged wrong size not what I ordered",              "negative"),
    ("fake product poor stitching sole came off",          "negative"),
    ("delivery was delayed and product was broken",        "negative"),
    ("never buying again terrible experience",             "negative"),
    ("disgraceful quality completely unacceptable",        "negative"),
    ("the color faded after first wash terrible",          "negative"),
    ("size was wrong and customer service ignored me",     "negative"),

    ("really happy with the quality of these shoes",       "positive"),
    ("excellent product fast delivery very satisfied",     "positive"),
    ("love these sneakers comfortable and stylish",        "positive"),
    ("great value for money highly recommended",           "positive"),
    ("perfect fit amazing quality thank you",              "positive"),
    ("outstanding service quick response very pleased",    "positive"),
    ("shoes look exactly like the picture wonderful",      "positive"),
    ("good quality nice design happy with purchase",       "positive"),
    ("smooth ordering experience great product",           "positive"),
    ("very comfortable shoes excellent build quality",     "positive"),
    ("fantastic sneakers worth every rupee",               "positive"),
    ("impressed with the quality and fast shipping",       "positive"),
]

_st = [t for t, _ in _SENT_DATA]
_sl = [l for _, l in _SENT_DATA]

_sent_pipeline = Pipeline([
    ("vec", CountVectorizer(ngram_range=(1, 2), lowercase=True)),
    ("clf", MultinomialNB(alpha=0.5)),
])
_sent_pipeline.fit(_st, _sl)

# ── Keyword lexicons ───────────────────────────────────────────────────────────
_NEG = {
    "terrible","awful","horrible","broken","damaged","wrong","fake","worst",
    "disappointed","unhappy","angry","disgusted","poor","bad","useless",
    "unacceptable","defective","pathetic","rude","delay","late","missing",
    "scam","fraud","never","hate","disgusting","inferior","shoddy","failure",
}
_POS = {
    "great","excellent","amazing","love","happy","satisfied","perfect","wonderful",
    "fantastic","superb","outstanding","good","comfortable","smooth","quick",
    "fast","beautiful","stylish","recommend","impressed","pleased","nice",
    "quality","genuine","authentic","brilliant","best","awesome","thank",
}


def analyze_sentiment(text: str) -> dict:
    """
    Returns: { label: 'positive'|'negative'|'neutral', score: float, method: str }
    """
    tokens = set(text.lower().split())
    pos_hits = len(tokens & _POS)
    neg_hits = len(tokens & _NEG)

    # ML prediction
    proba  = _sent_pipeline.predict_proba([text])[0]
    labels = _sent_pipeline.classes_
    ml_idx = int(np.argmax(proba))
    ml_label = labels[ml_idx]
    ml_conf  = float(proba[ml_idx])

    # Hybrid: if keyword signal is strong, use it; else trust ML
    if neg_hits > pos_hits and neg_hits >= 2:
        label, score = "negative", min(0.55 + neg_hits * 0.08, 0.99)
    elif pos_hits > neg_hits and pos_hits >= 2:
        label, score = "positive", min(0.55 + pos_hits * 0.08, 0.99)
    elif ml_conf >= 0.65:
        label, score = ml_label, ml_conf
    elif neg_hits > pos_hits:
        label, score = "negative", 0.55
    elif pos_hits > neg_hits:
        label, score = "positive", 0.55
    else:
        label, score = "neutral", 0.50

    return {
        "label":  label,
        "score":  round(score, 3),
        "pos_kw": pos_hits,
        "neg_kw": neg_hits,
    }


if __name__ == "__main__":
    samples = [
        "The shoes broke after one day, terrible quality",
        "Absolutely love these sneakers, very comfortable",
        "Size was wrong but okay overall",
        "Worst purchase ever, fake product disgusting",
    ]
    for s in samples:
        r = analyze_sentiment(s)
        print(f"[{r['label'].upper():8s}] {r['score']:.2f} | {s}")