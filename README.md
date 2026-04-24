# Customer_management

# 👟 NexaShoe – Customer Management & Hybrid Chatbot System

A production-grade customer support platform combining ML-based intent detection with a guided option-based chatbot UI.



```bash
pip install flask scikit-learn numpy
python app.py
```

Then open: **http://localhost:5000**  
Admin dashboard: **http://localhost:5000/admin**


---

## 🏗️ Architecture

```
shoecart/
├── app.py                  # Flask application (routes, flow logic)
├── requirements.txt
├── ml/
│   ├── intent_model.py     # ML: CountVectorizer + MultinomialNB + sentiment
│   ├── churn.py
│   ├── clustering.py
│   ├── sentiment.py
│   └── insights.py
├── db/
│   └── database.py               # SQLite ORM helpers + schema
└── templates/
    ├── index.html           # Chat UI
    └── admin.html           # Admin dashboard
```

---

## 🎯 Features

### Hybrid Chatbot
| Layer | Description |
|-------|-------------|
| ML Layer | CountVectorizer + MultinomialNB predicts intent from free text |
| UI Layer | Button-based guided flows, no typing required for most actions |

### Supported Flows
| Flow | Steps |
|------|-------|
| 🛍️ Buy | Shoe type → Brand → Size → Product cards with buy links |
| 📦 Track | Order ID → Live status + location |
| ❌ Cancel | Reason → Order ID → Confirmation |
| ↩️ Return | Order ID → Eligibility check → Refund/Replacement |
| ⚠️ Complaint | Shoe type → Issue → Order ID → Image upload → Sentiment analysis |
| ⭐ Feedback | Shoe type → Brand → Product → Star rating → Comment |
| 📞 Call | Time slot selection → Confirmation + reference ID |

### ML Model
- **Algorithm**: Multinomial Naive Bayes
- **Features**: Bag-of-Words (CountVectorizer, bigrams)
- **Intents**: buy, track, cancel, return, complaint, feedback, talk_to_agent
- **Sentiment**: Rule-based positive/negative/neutral for complaints
- **Persistence**: Saved as `intent_model.pkl`

### Admin Dashboard
- KPI cards (chats, complaints, feedback, calls, churn risk)
- Intent distribution bar chart
- Sentiment donut chart
- Complaint issue breakdown
- Call slot popularity
- Churn detection (sessions with 2+ complaints = high risk)
- Recent chats, complaints, call tables
- Auto-refresh every 30 seconds

---

## 🗄️ Database Schema

| Table | Purpose |
|-------|---------|
| `chats` | All messages with intent labels |
| `complaints` | Issues with sentiment + image path |
| `orders` | Seeded order data for tracking |
| `feedback` | Product ratings and comments |
| `call_requests` | Scheduled call time slots |
| `cancellations` | Cancelled orders with reason |

---

## 🔗 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/chat` | Free-text message → ML intent detection |
| POST | `/api/flow` | Button action → guided flow step |
| POST | `/api/upload` | Image upload for complaints |
| GET | `/api/admin` | Admin analytics JSON |

---

## 🧪 Sample Order IDs for Testing
- ORD-1001 → Delivered (8 days ago ❌ return not allowed)
- ORD-1002 → Out for delivery
- ORD-1003 → Shipped
- ORD-1004 → Processing
- ORD-1005 → Shipped
- ORD-1006 → Delivered (3 days ago ✅ return allowed)
- ORD-2001 → Processing
- ORD-2002 → Shipped
