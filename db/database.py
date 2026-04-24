"""
db/database.py
--------------
SQLite database initialisation, CRUD helpers, and analytics queries.
"""

import sqlite3
import os
import uuid
from datetime import datetime, timedelta

DB_PATH = os.path.join(os.path.dirname(__file__), "database.db")


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# ── Schema ────────────────────────────────────────────────────────────────────
def init_db():
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS chats (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT    NOT NULL,
                message     TEXT    NOT NULL,
                intent      TEXT,
                confidence  REAL,
                source      TEXT    DEFAULT 'text',
                created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS complaints (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT,
                order_id    TEXT,
                shoe_type   TEXT,
                issue       TEXT    NOT NULL,
                image_path  TEXT,
                sentiment   TEXT,
                sentiment_score REAL,
                cluster_id  INTEGER DEFAULT -1,
                created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS orders (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id        TEXT UNIQUE NOT NULL,
                product         TEXT,
                status          TEXT DEFAULT 'processing',
                delivery_date   TEXT,
                location        TEXT,
                customer_name   TEXT,
                created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS feedback (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT,
                product     TEXT,
                rating      INTEGER,
                comment     TEXT,
                sentiment   TEXT,
                created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS call_requests (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT    NOT NULL,
                time_slot   TEXT    NOT NULL,
                status      TEXT    DEFAULT 'scheduled',
                created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS sessions (
                session_id              TEXT PRIMARY KEY,
                complaint_count         INTEGER DEFAULT 0,
                negative_sentiment_count INTEGER DEFAULT 0,
                return_requests         INTEGER DEFAULT 0,
                low_rating_count        INTEGER DEFAULT 0,
                churn_risk              TEXT    DEFAULT 'low',
                updated_at              DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        """)
        _seed_orders(conn)
    print(f"[DB] Ready: {DB_PATH}")


def _seed_orders(conn):
    """Seed realistic sample orders so track/cancel/return flows work."""
    today = datetime.utcnow()
    sample_orders = [
        ("ORD-1001", "Nike Air Max 270 - Black/White (Size 9)",
         "delivered",    (today - timedelta(days=8)).strftime("%Y-%m-%d"),  "Delivered to door"),
        ("ORD-1002", "Adidas Ultraboost 22 - Grey (Size 8)",
         "out_for_delivery", today.strftime("%Y-%m-%d"),                    "In your city"),
        ("ORD-1003", "Puma RS-X - White/Red (Size 10)",
         "shipped",      (today + timedelta(days=2)).strftime("%Y-%m-%d"),  "Dispatch hub, Mumbai"),
        ("ORD-1004", "Reebok Classic Leather - White (Size 7)",
         "processing",   (today + timedelta(days=5)).strftime("%Y-%m-%d"),  "Warehouse, Delhi"),
        ("ORD-1005", "New Balance 574 - Navy (Size 11)",
         "shipped",      (today + timedelta(days=3)).strftime("%Y-%m-%d"),  "In transit, Hyderabad"),
        ("ORD-1006", "Converse Chuck Taylor - Black (Size 9)",
         "delivered",    (today - timedelta(days=3)).strftime("%Y-%m-%d"),  "Delivered to door"),
        ("ORD-2001", "Skechers Go Walk 6 - Black (Size 8)",
         "processing",   (today + timedelta(days=4)).strftime("%Y-%m-%d"),  "Warehouse, Pune"),
        ("ORD-2002", "Woodland Sports - Brown (Size 10)",
         "shipped",      (today + timedelta(days=1)).strftime("%Y-%m-%d"),  "Local courier, Bengaluru"),
    ]
    for row in sample_orders:
        try:
            conn.execute(
                "INSERT OR IGNORE INTO orders (order_id,product,status,delivery_date,location) VALUES (?,?,?,?,?)",
                row,
            )
        except Exception:
            pass


STATUS_DISPLAY = {
    "processing":       ("⏳", "Processing",        "Your order is being prepared."),
    "shipped":          ("🚚", "Shipped",            "Your package is on its way!"),
    "out_for_delivery": ("📦", "Out for Delivery",   "Your package is out for delivery today!"),
    "delivered":        ("✅", "Delivered",          "Your order has been delivered."),
    "cancelled":        ("❌", "Cancelled",          "Your order has been cancelled."),
    "returned":         ("↩️", "Return Initiated",   "Your return request is being processed."),
}


# ── Chat ──────────────────────────────────────────────────────────────────────
def save_chat(session_id, message, intent=None, confidence=None, source="text"):
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO chats (session_id,message,intent,confidence,source) VALUES (?,?,?,?,?)",
            (session_id, message, intent, confidence, source),
        )
        return cur.lastrowid


# ── Complaint ─────────────────────────────────────────────────────────────────
def save_complaint(session_id, order_id, shoe_type, issue, image_path, sentiment, sentiment_score, cluster_id=-1):
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO complaints
               (session_id,order_id,shoe_type,issue,image_path,sentiment,sentiment_score,cluster_id)
               VALUES (?,?,?,?,?,?,?,?)""",
            (session_id, order_id, shoe_type, issue, image_path, sentiment, sentiment_score, cluster_id),
        )
        _update_session(conn, session_id,
                        complaint_delta=1,
                        neg_delta=(1 if sentiment == "negative" else 0))
        return cur.lastrowid


# ── Order ─────────────────────────────────────────────────────────────────────
def get_order(order_id: str):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM orders WHERE order_id=?", (order_id,)).fetchone()
        return dict(row) if row else None


def cancel_order(order_id: str) -> bool:
    with get_conn() as conn:
        conn.execute(
            "UPDATE orders SET status='cancelled' WHERE order_id=? AND status NOT IN ('delivered','cancelled')",
            (order_id,),
        )
        row = conn.execute("SELECT status FROM orders WHERE order_id=?", (order_id,)).fetchone()
        return row and row["status"] == "cancelled"


def initiate_return(order_id: str) -> dict:
    order = get_order(order_id)
    if not order:
        return {"success": False, "reason": "Order not found."}
    if order["status"] == "cancelled":
        return {"success": False, "reason": "Cancelled orders cannot be returned."}

    delivery = order.get("delivery_date")
    if delivery and order["status"] == "delivered":
        delivered_date = datetime.strptime(delivery, "%Y-%m-%d")
        days_since = (datetime.utcnow() - delivered_date).days
        if days_since > 7:
            return {
                "success": False,
                "reason":  f"Return window expired. Orders can only be returned within 7 days of delivery. "
                           f"Your order was delivered {days_since} days ago.",
            }
    with get_conn() as conn:
        conn.execute("UPDATE orders SET status='returned' WHERE order_id=?", (order_id,))
    return {"success": True}


# ── Feedback ──────────────────────────────────────────────────────────────────
def save_feedback(session_id, product, rating, comment, sentiment):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO feedback (session_id,product,rating,comment,sentiment) VALUES (?,?,?,?,?)",
            (session_id, product, rating, comment, sentiment),
        )
        low_rating = 1 if rating <= 2 else 0
        _update_session(conn, session_id, low_rating_delta=low_rating)


# ── Call request ──────────────────────────────────────────────────────────────
def save_call_request(session_id, time_slot):
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO call_requests (session_id, time_slot) VALUES (?,?)",
            (session_id, time_slot),
        )
        return cur.lastrowid


# ── Session ───────────────────────────────────────────────────────────────────
def _update_session(conn, session_id,
                    complaint_delta=0, neg_delta=0,
                    return_delta=0, low_rating_delta=0):
    conn.execute(
        "INSERT OR IGNORE INTO sessions (session_id) VALUES (?)", (session_id,)
    )
    conn.execute(
        """UPDATE sessions SET
            complaint_count          = complaint_count + ?,
            negative_sentiment_count = negative_sentiment_count + ?,
            return_requests          = return_requests + ?,
            low_rating_count         = low_rating_count + ?,
            updated_at               = CURRENT_TIMESTAMP
           WHERE session_id = ?""",
        (complaint_delta, neg_delta, return_delta, low_rating_delta, session_id),
    )


def get_all_sessions():
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM sessions").fetchall()
        return [dict(r) for r in rows]


# ── Analytics ─────────────────────────────────────────────────────────────────
def get_analytics():
    with get_conn() as conn:
        total_chats      = conn.execute("SELECT COUNT(*) FROM chats").fetchone()[0]
        total_complaints = conn.execute("SELECT COUNT(*) FROM complaints").fetchone()[0]
        total_calls      = conn.execute("SELECT COUNT(*) FROM call_requests").fetchone()[0]
        total_feedback   = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]

        intent_dist = conn.execute(
            "SELECT intent, COUNT(*) as cnt FROM chats WHERE intent IS NOT NULL GROUP BY intent ORDER BY cnt DESC"
        ).fetchall()

        sentiment_dist = conn.execute(
            "SELECT sentiment as label, COUNT(*) as cnt FROM complaints WHERE sentiment IS NOT NULL GROUP BY sentiment"
        ).fetchall()

        complaints_raw = conn.execute(
            "SELECT issue, shoe_type, sentiment, cluster_id, created_at FROM complaints ORDER BY created_at DESC"
        ).fetchall()

        recent_complaints = conn.execute(
            "SELECT * FROM complaints ORDER BY created_at DESC LIMIT 20"
        ).fetchall()

        recent_calls = conn.execute(
            "SELECT * FROM call_requests ORDER BY created_at DESC LIMIT 20"
        ).fetchall()

        recent_feedback = conn.execute(
            "SELECT * FROM feedback ORDER BY created_at DESC LIMIT 10"
        ).fetchall()

        avg_rating_row = conn.execute(
            "SELECT ROUND(AVG(rating),2) FROM feedback"
        ).fetchone()
        avg_rating = avg_rating_row[0] or 0

        sessions = conn.execute("SELECT * FROM sessions").fetchall()

        # Complaints per shoe type
        shoe_type_dist = conn.execute(
            "SELECT shoe_type, COUNT(*) as cnt FROM complaints WHERE shoe_type IS NOT NULL GROUP BY shoe_type ORDER BY cnt DESC"
        ).fetchall()

        # Cluster dist
        cluster_dist = conn.execute(
            "SELECT cluster_id, COUNT(*) as cnt FROM complaints WHERE cluster_id >= 0 GROUP BY cluster_id"
        ).fetchall()

        # Call slot dist
        slot_dist = conn.execute(
            "SELECT time_slot, COUNT(*) as cnt FROM call_requests GROUP BY time_slot ORDER BY cnt DESC"
        ).fetchall()

        # Churn distribution from sessions
        churn_dist = conn.execute(
            "SELECT churn_risk, COUNT(*) as cnt FROM sessions GROUP BY churn_risk"
        ).fetchall()

    return {
        "total_chats":        total_chats,
        "total_complaints":   total_complaints,
        "total_calls":        total_calls,
        "total_feedback":     total_feedback,
        "avg_feedback_rating": float(avg_rating),
        "intent_distribution": [dict(r) for r in intent_dist],
        "sentiment_distribution": [dict(r) for r in sentiment_dist],
        "complaints_raw":     [dict(r) for r in complaints_raw],
        "recent_complaints":  [dict(r) for r in recent_complaints],
        "recent_calls":       [dict(r) for r in recent_calls],
        "recent_feedback":    [dict(r) for r in recent_feedback],
        "shoe_type_distribution": [dict(r) for r in shoe_type_dist],
        "cluster_distribution": [dict(r) for r in cluster_dist],
        "slot_distribution":  [dict(r) for r in slot_dist],
        "churn_distribution": {r["churn_risk"]: r["cnt"] for r in churn_dist},
        "sessions":           [dict(r) for r in sessions],
    }