"""
app.py
------
NexaShoe Customer Management & Hybrid Chatbot System
Flask entry point — all REST endpoints.
"""

import os, sys, uuid
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, request, jsonify, render_template, session

from ml.intent_model import predict_intent
from ml.sentiment    import analyze_sentiment
from ml.clustering   import cluster_complaints, predict_cluster
from ml.churn        import predict_churn, batch_predict
from ml.insights     import generate_insights

from db.database import (
    init_db, get_conn,
    save_chat, save_complaint, save_feedback, save_call_request,
    get_order, cancel_order, initiate_return,
    get_all_sessions, get_analytics,
    STATUS_DISPLAY,
)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "nexashoe-secret-2024")
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(__file__), "static", "uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

init_db()

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp"}


def _sid():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())[:10]
    return session["session_id"]


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ── Pages ──────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/admin-dashboard")
def admin_dashboard():
    return render_template("admin.html")


# ── POST /chat  ────────────────────────────────────────────────────────────────
@app.route("/chat", methods=["POST"])
def chat():
    """
    Natural language entry point. Detects intent and returns the first step
    of the appropriate guided flow.
    """
    body    = request.get_json(force=True) or {}
    message = (body.get("message") or "").strip()
    if not message:
        return jsonify({"error": "Empty message"}), 400

    sid    = body.get("session_id") or _sid()
    result = predict_intent(message)
    intent = result["intent"]
    conf   = result["confidence"]

    save_chat(sid, message, intent, conf, source="text")

    # Map intent → first step of guided flow
    flow_map = {
        "buy":       _flow_buy,
        "track":     _flow_track,
        "cancel":    _flow_cancel,
        "return":    _flow_return,
        "complaint": _flow_complaint,
        "feedback":  _flow_feedback,
        "call":      _flow_call,
    }

    handler   = flow_map.get(intent, _flow_buy)
    flow_resp = handler(step=0, selection=None, data={}, session_id=sid)
    flow_resp.update({
        "detected_intent": intent,
        "confidence":      conf,
        "session_id":      sid,
    })
    return jsonify(flow_resp)


# ── POST /flow  ────────────────────────────────────────────────────────────────
@app.route("/flow", methods=["POST"])
def flow():
    """
    Guided chatbot flow handler. Each button click / text submit goes here.
    Body: { session_id, flow, step, selection, data }
    """
    body      = request.get_json(force=True) or {}
    sid       = body.get("session_id") or _sid()
    flow_name = body.get("flow", "buy")
    step      = int(body.get("step", 0))
    selection = body.get("selection", None)
    data      = body.get("data", {})

    save_chat(sid, str(selection or ""), intent=flow_name, source="button")

    handlers = {
        "buy":       _flow_buy,
        "track":     _flow_track,
        "cancel":    _flow_cancel,
        "return":    _flow_return,
        "complaint": _flow_complaint,
        "feedback":  _flow_feedback,
        "call":      _flow_call,
    }
    handler = handlers.get(flow_name, _flow_buy)
    resp = handler(step=step, selection=selection, data=data, session_id=sid)
    resp["session_id"] = sid
    return jsonify(resp)


# ── POST /upload-image  ────────────────────────────────────────────────────────
@app.route("/upload-image", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return jsonify({"error": "No file"}), 400
    f = request.files["image"]
    if not f.filename or not allowed_file(f.filename):
        return jsonify({"error": "Invalid file type"}), 400
    ext      = f.filename.rsplit(".", 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    f.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
    return jsonify({"path": f"static/uploads/{filename}", "filename": filename})


# ── GET /admin  ────────────────────────────────────────────────────────────────
@app.route("/admin", methods=["GET"])
def admin_api():
    analytics = get_analytics()

    # Run ML clustering on complaints
    complaint_texts = [r["issue"] for r in analytics["complaints_raw"] if r.get("issue")]
    if complaint_texts:
        cluster_result = cluster_complaints(complaint_texts)
        analytics["complaint_clusters"] = cluster_result["cluster_summary"]
    else:
        analytics["complaint_clusters"] = {}

    # Run churn prediction for every session
    sessions = analytics.get("sessions", [])
    churn_results = batch_predict(sessions) if sessions else []

    # Update session churn risk in DB
    if churn_results:
        with get_conn() as conn:
            for r in churn_results:
                conn.execute(
                    "UPDATE sessions SET churn_risk=? WHERE session_id=?",
                    (r["risk"], r["session_id"]),
                )

    analytics["churn_predictions"] = churn_results

    # Re-fetch churn dist after update
    churn_dist_counts = {"low": 0, "medium": 0, "high": 0}
    for r in churn_results:
        churn_dist_counts[r["risk"]] = churn_dist_counts.get(r["risk"], 0) + 1
    analytics["churn_distribution"] = churn_dist_counts

    # Generate smart insights
    insights = generate_insights(analytics)
    analytics["insights"] = insights

    # Clean up large raw fields not needed by frontend
    analytics.pop("complaints_raw", None)
    analytics.pop("sessions", None)

    return jsonify(analytics)


# ═══════════════════════════════════════════════════════════════════════════════
# FLOW HANDLERS
# ═══════════════════════════════════════════════════════════════════════════════

# ── BUY FLOW ──────────────────────────────────────────────────────────────────
SHOE_TYPES  = ["Sneakers", "Formal", "Sports", "Casual", "Boots"]
BRANDS      = ["Nike", "Adidas", "Puma", "Reebok", "New Balance", "Converse"]
SIZES       = ["UK 6", "UK 7", "UK 8", "UK 9", "UK 10", "UK 11", "UK 12"]

PRODUCTS = {
    ("Sneakers", "Nike"):      [("Nike Air Max 270",   "₹8,995",  "https://www.nike.com"),
                                ("Nike Air Force 1",    "₹7,495",  "https://www.nike.com")],
    ("Sneakers", "Adidas"):    [("Adidas Ultraboost",  "₹11,999", "https://www.adidas.co.in"),
                                ("Adidas Stan Smith",   "₹6,499",  "https://www.adidas.co.in")],
    ("Sports",   "Puma"):      [("Puma Velocity Nitro","₹9,999",  "https://in.puma.com"),
                                ("Puma RS-X",           "₹7,999",  "https://in.puma.com")],
    ("Formal",   "Reebok"):    [("Reebok Club C 85",   "₹5,999",  "https://www.reebok.in")],
    ("Casual",   "Converse"):  [("Converse Chuck 70",  "₹5,495",  "https://www.converse.in")],
    ("Boots",    "New Balance"):[("NB 574 Boot",        "₹8,999",  "https://www.newbalance.in")],
}
DEFAULT_PRODUCTS = [
    ("Brand Shoe Classic",  "₹4,999",  "https://example.com/shop"),
    ("Street Runner Pro",   "₹5,999",  "https://example.com/shop"),
]

def _flow_buy(step, selection, data, session_id):
    if step == 0:
        return _btn("👟 What type of shoe are you looking for?",
                    SHOE_TYPES, flow="buy", step=1, data=data, icons=["👟","👞","🏃","🚶","🥾"])
    if step == 1:
        data["shoe_type"] = selection
        return _btn(f"Great choice! Which brand do you prefer?",
                    BRANDS, flow="buy", step=2, data=data)
    if step == 2:
        data["brand"] = selection
        return _btn(f"What is your size?", SIZES, flow="buy", step=3, data=data)
    if step == 3:
        data["size"] = selection
        shoe_type = data.get("shoe_type", "")
        brand     = data.get("brand", "")
        products  = PRODUCTS.get((shoe_type, brand), DEFAULT_PRODUCTS)
        cards = [{
            "name":  p[0],
            "price": p[1],
            "link":  p[2],
            "brand": brand,
            "type":  shoe_type,
            "size":  selection,
        } for p in products]
        return {
            "type":    "products",
            "message": f"Here are {shoe_type} shoes from {brand} in {selection}:",
            "products": cards,
            "flow":    "buy",
            "step":    4,
            "data":    data,
            "done":    False,
        }
    return _done("✅ Redirecting you to checkout! Happy shopping! 🛍️")


# ── TRACK FLOW ─────────────────────────────────────────────────────────────────
def _flow_track(step, selection, data, session_id):
    if step == 0:
        return _input("🔍 Please enter your Order ID (e.g. ORD-1001):",
                      flow="track", step=1, data=data, placeholder="ORD-XXXX")
    if step == 1:
        order_id = (selection or "").strip().upper()
        order    = get_order(order_id)
        if not order:
            return _done(f"❌ Order **{order_id}** not found. Please check the ID and try again.",
                         error=True)
        icon, label, msg = STATUS_DISPLAY.get(
            order["status"], ("📦", order["status"], ""))
        return {
            "type":    "order_status",
            "message": f"Order found! Here are the details:",
            "order": {
                "order_id":      order["order_id"],
                "product":       order["product"],
                "status_raw":    order["status"],
                "status_icon":   icon,
                "status_label":  label,
                "status_msg":    msg,
                "delivery_date": order["delivery_date"],
                "location":      order["location"],
            },
            "flow": "track",
            "step": 2,
            "data": data,
            "done": True,
        }
    return _done("✅ Track another order anytime you need!")


# ── CANCEL FLOW ────────────────────────────────────────────────────────────────
CANCEL_REASONS = [
    "Changed my mind",
    "Ordered by mistake",
    "Found a better price",
    "Delivery too slow",
    "Wrong item ordered",
]

def _flow_cancel(step, selection, data, session_id):
    if step == 0:
        return _btn("😔 We're sorry to hear that. What's the reason for cancellation?",
                    CANCEL_REASONS, flow="cancel", step=1, data=data)
    if step == 1:
        data["reason"] = selection
        return _input("Please enter your Order ID to proceed:",
                      flow="cancel", step=2, data=data, placeholder="ORD-XXXX")
    if step == 2:
        order_id = (selection or "").strip().upper()
        data["order_id"] = order_id
        order = get_order(order_id)
        if not order:
            return _done(f"❌ Order **{order_id}** not found.", error=True)
        if order["status"] in ("delivered", "cancelled"):
            return _done(
                f"⚠️ Order **{order_id}** cannot be cancelled — it is already **{order['status']}**.",
                error=True,
            )
        return _confirm(
            f"Are you sure you want to cancel **{order['product']}** (#{order_id})?",
            flow="cancel", step=3, data=data,
        )
    if step == 3:
        if selection == "yes":
            order_id = data.get("order_id", "")
            ok = cancel_order(order_id)
            if ok:
                return _done(f"✅ Order **{order_id}** has been successfully cancelled. "
                             f"Refund will be processed within 5–7 business days.")
            return _done("❌ Cancellation failed. Please contact support.", error=True)
        return _done("👍 No problem! Your order remains active.")
    return _done("Done!")


# ── RETURN FLOW ────────────────────────────────────────────────────────────────
def _flow_return(step, selection, data, session_id):
    if step == 0:
        return _input("↩️ Let's process your return. Please enter your Order ID:",
                      flow="return", step=1, data=data, placeholder="ORD-XXXX")
    if step == 1:
        order_id = (selection or "").strip().upper()
        data["order_id"] = order_id
        result = initiate_return(order_id)

        if not result["success"]:
            return _done(f"❌ {result['reason']}", error=True)

        order = get_order(order_id)
        return _btn(
            f"✅ Return window is valid for **{order['product']}**. "
            f"How would you like to proceed?",
            ["Refund to original payment", "Replace with same item"],
            flow="return", step=2, data=data,
        )
    if step == 2:
        data["return_type"] = selection
        order_id = data.get("order_id", "")
        if "Refund" in selection:
            return _done(
                f"✅ Refund initiated for Order **{order_id}**! "
                f"You will receive your refund in 5–7 business days. "
                f"A pickup will be scheduled within 24 hours."
            )
        return _done(
            f"✅ Replacement requested for Order **{order_id}**! "
            f"Your replacement will be dispatched within 2–3 business days. "
            f"A pickup for the original item will be arranged."
        )
    return _done("Done!")


# ── COMPLAINT FLOW ─────────────────────────────────────────────────────────────
ISSUES = [
    "Size / Fit Problem",
    "Quality Issue",
    "Wrong Item Delivered",
    "Damaged Product",
    "Delivery Problem",
    "Fake / Counterfeit",
]

def _flow_complaint(step, selection, data, session_id):
    if step == 0:
        return _btn("😔 We're really sorry about this. Which shoe type was the issue with?",
                    SHOE_TYPES, flow="complaint", step=1, data=data)
    if step == 1:
        data["shoe_type"] = selection
        return _btn(f"What issue are you facing with your {selection}?",
                    ISSUES, flow="complaint", step=2, data=data)
    if step == 2:
        data["issue"] = selection
        return _image_upload(
            f"Got it — **{selection}**. Would you like to upload an image as evidence?",
            flow="complaint", step=3, data=data,
        )
    if step == 3:
        data["image_path"] = selection  # path returned from /upload-image or None
        # Perform sentiment on combined complaint text
        combined = f"{data.get('shoe_type','')} {data.get('issue','')}"
        sent     = analyze_sentiment(combined)
        cluster  = predict_cluster(combined)
        save_complaint(
            session_id,
            order_id=None,
            shoe_type=data.get("shoe_type"),
            issue=data.get("issue"),
            image_path=data.get("image_path"),
            sentiment=sent["label"],
            sentiment_score=sent["score"],
            cluster_id=cluster,
        )
        ticket_id = f"TKT-{uuid.uuid4().hex[:6].upper()}"
        return _done(
            f"✅ Your complaint has been registered (Ticket: **{ticket_id}**).\n\n"
            f"Our quality team will investigate your **{data.get('issue')}** issue and "
            f"respond within 48 hours. We sincerely apologise for the inconvenience."
        )
    return _done("Done!")


# ── FEEDBACK FLOW ─────────────────────────────────────────────────────────────
FEEDBACK_PRODUCTS = ["Sneakers", "Formal Shoes", "Sports Shoes", "Casual Shoes", "Boots"]

def _flow_feedback(step, selection, data, session_id):
    if step == 0:
        return _btn("💬 We'd love your feedback! Which product are you reviewing?",
                    FEEDBACK_PRODUCTS, flow="feedback", step=1, data=data)
    if step == 1:
        data["product"] = selection
        return {
            "type":    "rating",
            "message": f"How would you rate your **{selection}**?",
            "flow":    "feedback",
            "step":    2,
            "data":    data,
            "done":    False,
        }
    if step == 2:
        data["rating"] = int(selection)
        return _input(
            f"Thank you for the {selection}⭐ rating! Any additional comments? (optional — press Skip to continue)",
            flow="feedback", step=3, data=data, placeholder="Share your thoughts...", skippable=True,
        )
    if step == 3:
        data["comment"] = selection if selection and selection.lower() != "skip" else ""
        comment = data.get("comment", "")
        sent    = analyze_sentiment(comment) if comment else {"label": "neutral", "score": 0.5}
        save_feedback(session_id, data.get("product"), data.get("rating"), comment, sent["label"])
        stars = "⭐" * data.get("rating", 0)
        return _done(
            f"{stars} Thank you for your feedback on **{data.get('product')}**! "
            f"Your opinion helps us improve. We're glad to have you as a customer! 🙏"
        )
    return _done("Done!")


# ── CALL FLOW ─────────────────────────────────────────────────────────────────
TIME_SLOTS = ["10:00 AM – 12:00 PM", "12:00 PM – 3:00 PM", "3:00 PM – 6:00 PM", "6:00 PM – 9:00 PM"]

def _flow_call(step, selection, data, session_id):
    if step == 0:
        return _btn(
            "📞 Great! A customer service agent will call you. Please select a convenient time slot:",
            TIME_SLOTS, flow="call", step=1, data=data, icons=["🌅","☀️","🌤","🌆"],
        )
    if step == 1:
        data["time_slot"] = selection
        save_call_request(session_id, selection)
        return _done(
            f"✅ Your call has been scheduled!\n\n"
            f"📅 **Time Slot:** {selection}\n"
            f"📞 A customer service representative will call you during this window.\n\n"
            f"Please keep your registered phone number accessible. "
            f"Thank you for choosing NexaShoe!"
        )
    return _done("Done!")


# ═══════════════════════════════════════════════════════════════════════════════
# RESPONSE BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

def _btn(message, options, flow, step, data, icons=None):
    opts = [{"label": o, "icon": (icons[i] if icons and i < len(icons) else "")}
            for i, o in enumerate(options)]
    return {"type": "buttons", "message": message, "options": opts,
            "flow": flow, "step": step, "data": data, "done": False}


def _input(message, flow, step, data, placeholder="", skippable=False):
    return {"type": "text_input", "message": message, "placeholder": placeholder,
            "skippable": skippable, "flow": flow, "step": step, "data": data, "done": False}


def _confirm(message, flow, step, data):
    return {"type": "confirm", "message": message,
            "options": [{"label": "Yes, Cancel", "value": "yes", "icon": "❌"},
                        {"label": "No, Keep it", "value": "no",  "icon": "✅"}],
            "flow": flow, "step": step, "data": data, "done": False}


def _image_upload(message, flow, step, data):
    return {"type": "image_upload", "message": message,
            "flow": flow, "step": step, "data": data, "done": False}


def _done(message, error=False):
    return {"type": "done", "message": message, "error": error, "done": True}


# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n  ✦  NexaShoe Chat     → http://127.0.0.1:{port}")
    print(f"  ✦  Admin Dashboard   → http://127.0.0.1:{port}/admin-dashboard\n")
    app.run(host="0.0.0.0", port=10000)
