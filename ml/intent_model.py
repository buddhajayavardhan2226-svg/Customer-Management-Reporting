"""
ml/intent_model.py
------------------
Multinomial Naive Bayes intent classifier for 7 customer support intents.
Uses CountVectorizer with bigrams for feature extraction.
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np

INTENTS = ["buy", "track", "cancel", "return", "complaint", "feedback", "call"]

TRAINING_DATA = [
    # ── BUY ──────────────────────────────────────────────────────────────
    ("I want to buy shoes",                         "buy"),
    ("I'd like to purchase some sneakers",          "buy"),
    ("Can I place an order for shoes?",             "buy"),
    ("Show me shoes to buy",                        "buy"),
    ("I want to order a pair of Nikes",             "buy"),
    ("Buy shoes online",                            "buy"),
    ("I need new running shoes",                    "buy"),
    ("Help me choose and purchase shoes",           "buy"),
    ("Add shoes to cart",                           "buy"),
    ("I want to shop for footwear",                 "buy"),
    ("Looking for formal shoes to purchase",        "buy"),
    ("I want to get some boots",                    "buy"),

    # ── TRACK ────────────────────────────────────────────────────────────
    ("Where is my order?",                          "track"),
    ("Track my order",                              "track"),
    ("What is the status of my delivery?",          "track"),
    ("I want to know where my shoes are",           "track"),
    ("Track shipment",                              "track"),
    ("My order hasn't arrived yet, where is it?",   "track"),
    ("Check delivery status",                       "track"),
    ("When will my shoes be delivered?",            "track"),
    ("Order tracking",                              "track"),
    ("Has my package been shipped?",                "track"),
    ("Give me the current location of my parcel",   "track"),
    ("I need to track my shoe order",               "track"),

    # ── CANCEL ───────────────────────────────────────────────────────────
    ("I want to cancel my order",                   "cancel"),
    ("Cancel my shoe purchase",                     "cancel"),
    ("Please cancel the order I just placed",       "cancel"),
    ("Stop my order",                               "cancel"),
    ("I don't want these shoes anymore",            "cancel"),
    ("Cancel order immediately",                    "cancel"),
    ("I made a mistake, cancel my order",           "cancel"),
    ("Cancellation request",                        "cancel"),
    ("How do I cancel my order?",                   "cancel"),
    ("I need to cancel before it ships",            "cancel"),
    ("Please stop my delivery",                     "cancel"),
    ("Order cancellation",                          "cancel"),

    # ── RETURN ───────────────────────────────────────────────────────────
    ("I want to return my shoes",                   "return"),
    ("How do I return a product?",                  "return"),
    ("These shoes don't fit, I want to return them","return"),
    ("Return policy for shoes",                     "return"),
    ("I'd like to initiate a return",               "return"),
    ("Send back the shoes I received",              "return"),
    ("Return and refund please",                    "return"),
    ("I received wrong shoes and want to return",   "return"),
    ("Replacement for defective shoes",             "return"),
    ("I want to exchange my purchase",              "return"),
    ("Return damaged product",                      "return"),
    ("I need a refund for my order",                "return"),

    # ── COMPLAINT ────────────────────────────────────────────────────────
    ("I have a complaint about my shoes",           "complaint"),
    ("My shoes are damaged",                        "complaint"),
    ("The quality is terrible",                     "complaint"),
    ("I received the wrong size",                   "complaint"),
    ("These shoes broke after one day",             "complaint"),
    ("I'm very unhappy with my purchase",           "complaint"),
    ("The stitching came apart",                    "complaint"),
    ("Wrong item was delivered",                    "complaint"),
    ("Sole fell off my sneakers",                   "complaint"),
    ("I want to file a complaint",                  "complaint"),
    ("Product quality issue",                       "complaint"),
    ("Shoes are fake, not genuine",                 "complaint"),

    # ── FEEDBACK ─────────────────────────────────────────────────────────
    ("I want to give feedback",                     "feedback"),
    ("Here is my review of the product",            "feedback"),
    ("I'd like to share my experience",             "feedback"),
    ("Rating and review for shoes",                 "feedback"),
    ("Great shoes, wanted to provide feedback",     "feedback"),
    ("Could be better, here is my opinion",         "feedback"),
    ("Product feedback and suggestions",            "feedback"),
    ("Leave a review for my purchase",              "feedback"),
    ("I have some suggestions",                     "feedback"),
    ("Share customer experience",                   "feedback"),
    ("I want to rate the product I bought",         "feedback"),
    ("Tell you about my experience",                "feedback"),

    # ── CALL ─────────────────────────────────────────────────────────────
    ("I want to talk to customer service",          "call"),
    ("Can I speak with a representative?",          "call"),
    ("Schedule a call with support",                "call"),
    ("I need human assistance",                     "call"),
    ("Connect me to an agent",                      "call"),
    ("Talk to a real person",                       "call"),
    ("I need to speak with someone",                "call"),
    ("Customer support call please",                "call"),
    ("Book a call with your team",                  "call"),
    ("I want a callback",                           "call"),
    ("Can someone call me?",                        "call"),
    ("I prefer to speak on phone",                  "call"),
]

_texts  = [t for t, _ in TRAINING_DATA]
_labels = [l for _, l in TRAINING_DATA]

_pipeline = Pipeline([
    ("vec", CountVectorizer(ngram_range=(1, 2), lowercase=True, min_df=1)),
    ("clf", MultinomialNB(alpha=0.4)),
])
_pipeline.fit(_texts, _labels)

_INTENT_LABELS = _pipeline.classes_


def predict_intent(text: str) -> dict:
    """
    Returns dict with intent, confidence, and all class probabilities.
    """
    proba      = _pipeline.predict_proba([text])[0]
    intent_idx = int(np.argmax(proba))
    intent     = _INTENT_LABELS[intent_idx]
    confidence = float(proba[intent_idx])

    return {
        "intent":     intent,
        "confidence": round(confidence, 3),
        "all_proba":  {cls: round(float(p), 3) for cls, p in zip(_INTENT_LABELS, proba)},
    }


if __name__ == "__main__":
    tests = [
        "I want to buy some Adidas sneakers",
        "Where is my order? It's been a week",
        "Cancel my order please",
        "I want to return these broken shoes",
        "The sole fell off after one use, terrible quality",
        "Amazing shoes, very comfortable, great product",
        "I need to speak with a human agent",
    ]
    for t in tests:
        r = predict_intent(t)
        print(f"[{r['intent'].upper():10s}] {r['confidence']:.2f} | {t}")