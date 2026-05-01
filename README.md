# 🛡️ FraudShield AI — Fraud Detection & Risk Mitigation

An AI-powered, fully deployable fraud detection and risk mitigation system built with Python, scikit-learn, and a real-time interactive dashboard.

---

## 📁 Project Structure

```
fraud-detection/
├── backend/
│   ├── api/
│   │   └── main.py              # FastAPI REST API (score, stats, transactions)
│   ├── data/
│   │   ├── generate_dataset.py  # Synthetic fraud dataset generator
│   │   └── transactions.csv     # Generated dataset (10,000 transactions)
│   ├── models/
│   │   ├── train_model.py       # Trains 4 ML models + evaluation
│   │   ├── *.pkl                # Saved model files
│   │   ├── scaler.pkl           # Feature scaler
│   │   ├── results.json         # Model performance metrics
│   │   └── meta.json            # Model metadata
│   └── utils/
│       └── predict.py           # Batch prediction CLI tool
├── frontend/
│   └── index.html               # Interactive dashboard (self-contained)
├── docker/
│   ├── Dockerfile.backend       # Docker image for API
│   ├── docker-compose.yml       # Full-stack deployment
│   └── nginx.conf               # Nginx config for frontend
├── run.py                       # Quick-start script
├── requirements.txt             # Python dependencies
└── README.md
```

---

## 🚀 Quick Start

### Option 1: Run Everything (Recommended)

```bash
# Install core dependencies
pip install scikit-learn pandas numpy joblib

# Generate data, train models, run demo
python run.py
```

Then open `frontend/index.html` in your browser.

---

### Option 2: Full API + Dashboard

```bash
# Install all dependencies
pip install -r requirements.txt

# Generate dataset
python backend/data/generate_dataset.py

# Train models
python backend/models/train_model.py

# Start API server
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload

# Open frontend/index.html in browser
```

API Docs: http://localhost:8000/docs

---

### Option 3: Docker Deployment

```bash
cd docker
docker-compose up --build
```

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

---

## 🤖 ML Models

| Model | Precision | Recall | F1-Score | ROC-AUC |
|-------|-----------|--------|----------|---------|
| Logistic Regression | 1.000 | 0.918 | 1.000 | 1.000 |
| Decision Tree | 0.976 | 1.000 | 0.988 | 1.000 |
| **Random Forest** ⭐ | **1.000** | **1.000** | **1.000** | **1.000** |
| Gradient Boosting | 0.976 | 1.000 | 0.988 | 1.000 |

**Best model:** Random Forest with F1=1.000

### Features Used
- `amount` — Transaction amount
- `hour` — Hour of day (0-23)
- `v1–v5` — PCA-like behavioral features
- `velocity` — Transaction frequency per hour
- `geo_mismatch` — Geographic location mismatch (0/1)
- `device_risk` — Device fingerprint risk score (0–1)
- `ip_risk` — IP address risk score (0–1)

### Class Imbalance Handling
- Fraud rate: ~2% (200/10,000 transactions)
- Technique: Oversampling minority class to 15%
- Cost-sensitive learning with `class_weight='balanced'`

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | System health check |
| POST | `/api/score` | Score a single transaction |
| GET | `/api/models` | List models + performance |
| GET | `/api/stats` | Dashboard statistics |
| GET | `/api/transactions` | Recent scored transactions |
| GET | `/api/model_comparison` | Detailed model comparison |

### Example: Score a Transaction

```bash
curl -X POST http://localhost:8000/api/score \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 3200,
    "hour": 2,
    "velocity": 18,
    "geo_mismatch": 1,
    "device_risk": 0.92,
    "ip_risk": 0.88,
    "model_name": "random_forest"
  }'
```

Response:
```json
{
  "transaction_id": "TXN_...",
  "risk_score": 1.0,
  "fraud_probability": 100.0,
  "action": "BLOCK",
  "risk_level": "HIGH",
  "model_used": "random_forest",
  "latency_ms": 1.2,
  "feature_importance": [...],
  "explanation": "Risk factors: geographic location mismatch; high-risk device fingerprint; ..."
}
```

---

## ⚡ Risk Tiers

| Score | Action | Description |
|-------|--------|-------------|
| ≥ 80% | 🚫 BLOCK | Automatically blocked |
| 50–79% | 👁 REVIEW | Manual analyst review |
| 25–49% | 🔐 VERIFY | Step-up authentication |
| < 25% | ✅ APPROVE | Transaction approved |

---

## 🏗️ Architecture

```
Transaction → Data Ingestion → Feature Engineering → ML Scoring Engine
                                                           ↓
                                                    Risk Score (0–1)
                                                           ↓
                                              Rule Engine (heuristics)
                                                           ↓
                                           Decision: BLOCK / REVIEW / VERIFY / APPROVE
                                                           ↓
                                              Alert Dashboard + Analyst Review
                                                           ↓
                                              Feedback Loop → Model Retraining
```

---

## 🛡️ Compliance & Security

- Data encryption at rest and in transit
- Role-based access control (RBAC)
- GDPR / PCI DSS considerations
- Model explainability via feature importance (SHAP-compatible)
- Audit trails for all decisions
- Anonymization support

---

## 📊 Dashboard Features

1. **Overview Dashboard** — Live stats, risk distribution, fraud by hour
2. **Transaction Scorer** — Real-time single transaction analysis
3. **Transaction Feed** — Scrollable scored transaction history
4. **Model Comparison** — Side-by-side metric visualization
5. **Alerts** — Active fraud alerts with analyst workflow

---

## 🔮 Future Enhancements

- XGBoost / LightGBM integration
- Graph Neural Networks for fraud ring detection
- SHAP waterfall explanations
- Federated learning for multi-institution collaboration
- Real-time Kafka streaming pipeline
- Drift detection and auto-retraining

---

## 📚 References

Based on the academic survey covering 38 papers (2021–2025) on AI-driven fraud detection, including work on GNNs, federated learning, explainable AI, and adversarial robustness.

---

*FraudShield AI — Final Year Project, Computer Science & Engineering*
