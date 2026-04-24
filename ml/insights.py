"""
ml/insights.py
--------------
AI-powered insights and recommendations generator.
Combines ML cluster analysis + churn stats + sentiment trends.
"""

from datetime import datetime, timedelta


def generate_insights(analytics: dict) -> dict:
    """
    Takes the full analytics dict and returns:
      - insights  : list of insight strings
      - recommendations: list of action recommendations
      - alerts    : list of urgent alerts
    """
    insights        = []
    recommendations = []
    alerts          = []

    total_chats      = analytics.get("total_chats", 0)
    total_complaints = analytics.get("total_complaints", 0)
    sentiment_dist   = analytics.get("sentiment_distribution", [])
    clusters         = analytics.get("complaint_clusters", {})
    churn_dist       = analytics.get("churn_distribution", {})
    total_calls      = analytics.get("total_calls", 0)
    feedback_avg     = analytics.get("avg_feedback_rating", 0)
    intent_dist      = analytics.get("intent_distribution", [])

    # ── Sentiment insights ────────────────────────────────────────────────
    neg_count = next((x["cnt"] for x in sentiment_dist if x["label"] == "negative"), 0)
    pos_count = next((x["cnt"] for x in sentiment_dist if x["label"] == "positive"), 0)
    total_sent = neg_count + pos_count or 1

    neg_pct = round((neg_count / total_sent) * 100, 1)
    pos_pct = round((pos_count / total_sent) * 100, 1)

    if neg_pct > 60:
        insights.append(f"⚠️ {neg_pct}% of complaints carry negative sentiment — customer satisfaction is critically low.")
        alerts.append(f"Negative sentiment at {neg_pct}% — immediate quality review needed.")
    elif neg_pct > 40:
        insights.append(f"📉 Negative sentiment accounts for {neg_pct}% of complaints — trending upwards.")
    else:
        insights.append(f"✅ Sentiment looks healthy — {pos_pct}% positive, {neg_pct}% negative.")

    # ── Complaint ratio insights ───────────────────────────────────────────
    if total_chats > 0:
        complaint_ratio = round((total_complaints / total_chats) * 100, 1)
        if complaint_ratio > 40:
            insights.append(f"🚨 {complaint_ratio}% of all chats are complaints — significantly above the 15% industry benchmark.")
            alerts.append(f"Complaint ratio ({complaint_ratio}%) is dangerously high.")
            recommendations.append("Conduct an urgent product quality audit across all shoe lines.")
        elif complaint_ratio > 20:
            insights.append(f"⚠️ Complaint rate is {complaint_ratio}%, above the 15% benchmark. Monitor closely.")
            recommendations.append("Review packaging and QC processes to reduce damage complaints.")
        else:
            insights.append(f"✅ Complaint rate is {complaint_ratio}% — within acceptable range.")

    # ── Cluster insights ──────────────────────────────────────────────────
    if clusters:
        top_cluster = max(clusters.items(), key=lambda x: x[1].get("count", 0))
        top_name  = top_cluster[1].get("name", "Unknown")
        top_count = top_cluster[1].get("count", 0)
        if top_count > 0:
            insights.append(f"📊 Most common complaint category: '{top_name}' ({top_count} complaints).")

            if "Size" in top_name:
                recommendations.append("Improve size chart accuracy — add a detailed size guide with measurements.")
                recommendations.append("Add a 'Size Guide' popup on all product pages to reduce size-related returns.")
            elif "Quality" in top_name or "Damage" in top_name:
                recommendations.append("Escalate product quality issues to manufacturing team for immediate review.")
                recommendations.append("Implement stricter QC checks before dispatch.")
            elif "Delivery" in top_name:
                recommendations.append("Review logistics partner SLA — negotiate better delivery timelines.")
                recommendations.append("Improve real-time tracking notifications for customers.")
            elif "Wrong" in top_name or "Order" in top_name:
                recommendations.append("Audit order fulfillment process — reduce wrong item dispatch errors.")
                recommendations.append("Add order verification step before shipping.")

    # ── Churn risk insights ────────────────────────────────────────────────
    high_risk  = churn_dist.get("high", 0)
    med_risk   = churn_dist.get("medium", 0)
    total_sess = sum(churn_dist.values()) or 1

    if high_risk > 0:
        high_pct = round((high_risk / total_sess) * 100, 1)
        insights.append(f"🔥 {high_risk} sessions ({high_pct}%) are classified as HIGH churn risk.")
        if high_pct > 20:
            alerts.append(f"{high_risk} high-risk customers detected — proactive outreach recommended.")
            recommendations.append("Launch a loyalty recovery campaign for high-risk customers (discount + personalised email).")
        else:
            recommendations.append("Monitor high-risk sessions and follow up proactively with support.")

    if med_risk > 0:
        insights.append(f"⚡ {med_risk} sessions are at MEDIUM churn risk — early intervention can retain them.")

    # ── Call scheduling insights ───────────────────────────────────────────
    if total_calls > 0:
        call_ratio = round((total_calls / max(total_chats, 1)) * 100, 1)
        if call_ratio > 30:
            insights.append(f"📞 {call_ratio}% of customers requested agent calls — chatbot resolution rate is low.")
            recommendations.append("Expand chatbot FAQ coverage to reduce escalation to human agents.")
        else:
            insights.append(f"📞 {total_calls} calls scheduled — chatbot is handling most queries independently.")

    # ── Feedback insights ──────────────────────────────────────────────────
    if feedback_avg > 0:
        if feedback_avg >= 4.5:
            insights.append(f"⭐ Average product rating is {feedback_avg}/5 — excellent customer satisfaction.")
        elif feedback_avg >= 3.5:
            insights.append(f"⭐ Average product rating is {feedback_avg}/5 — room for improvement.")
            recommendations.append("Identify specific product lines with low ratings and investigate root causes.")
        else:
            insights.append(f"⭐ Average product rating is {feedback_avg}/5 — critically low. Urgent action needed.")
            alerts.append(f"Average rating at {feedback_avg}/5 — product quality review is urgent.")
            recommendations.append("Hold a product line review and consider replacing low-rated inventory.")

    # ── Intent distribution insights ──────────────────────────────────────
    if intent_dist:
        top_intent = max(intent_dist, key=lambda x: x["cnt"])
        if top_intent["intent"] not in ("buy", "feedback"):
            insights.append(
                f"📈 Most common intent is '{top_intent['intent']}' — "
                f"indicating post-purchase issues dominate customer interactions."
            )

    # ── Default recommendations ────────────────────────────────────────────
    if not recommendations:
        recommendations.append("Maintain current quality standards and continue monitoring complaint trends.")
        recommendations.append("Consider expanding the chatbot with more automated resolution flows.")

    return {
        "insights":        insights,
        "recommendations": recommendations,
        "alerts":          alerts,
        "generated_at":    datetime.utcnow().isoformat() + "Z",
    }