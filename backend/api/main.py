"""
AI Fraud Detection & Risk Mitigation — FastAPI Backend
"""
import os
import json
import time
import random
import uuid
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# ─── Try to import FastAPI; fall back gracefully ───────────────────────────────
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    import uvicorn
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    print("FastAPI not available – running in CLI demo mode")

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')
DATA_DIR   = os.path.join(BASE_DIR, '..', 'data')

FEATURE_COLS = ['amount', 'hour', 'v1', 'v2', 'v3', 'v4', 'v5',
                'velocity', 'geo_mismatch', 'device_risk', 'ip_risk']

# ─── Load artefacts ───────────────────────────────────────────────────────────
def load_artefacts():
    models, scaler, meta, results = {}, None, {}, {}
    try:
        scaler  = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
        meta    = json.load(open(os.path.join(MODELS_DIR, 'meta.json')))
        results = json.load(open(os.path.join(MODELS_DIR, 'results.json')))
        for name in ['random_forest', 'gradient_boosting',
                     'logistic_regression', 'decision_tree']:
            path = os.path.join(MODELS_DIR, f'{name}.pkl')
            if os.path.exists(path):
                models[name] = joblib.load(path)
        print(f"✅ Loaded {len(models)} models")
    except Exception as e:
        print(f"⚠  Could not load models: {e}")
    return models, scaler, meta, results

MODELS, SCALER, META, RESULTS = load_artefacts()

# ─── Risk scoring logic ───────────────────────────────────────────────────────
def risk_action(score: float) -> dict:
    if score >= 0.80:
        return {"action": "BLOCK",   "level": "HIGH",   "color": "#ef4444"}
    if score >= 0.50:
        return {"action": "REVIEW",  "level": "MEDIUM", "color": "#f59e0b"}
    if score >= 0.25:
        return {"action": "VERIFY",  "level": "LOW",    "color": "#3b82f6"}
    return     {"action": "APPROVE", "level": "SAFE",   "color": "#22c55e"}

def feature_importance(model_name: str, feature_values: list) -> list:
    """Return simple feature importance from model."""
    model = MODELS.get(model_name)
    if model and hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        importances = np.abs(np.array(feature_values)) / (np.abs(np.array(feature_values)).sum() + 1e-9)
    
    return [
        {"feature": f, "importance": float(imp), "value": float(v)}
        for f, imp, v in sorted(
            zip(FEATURE_COLS, importances, feature_values),
            key=lambda x: x[1], reverse=True
        )
    ][:6]

def score_transaction(features: dict, model_name: str = 'random_forest') -> dict:
    t0 = time.time()
    feat_vec = [features.get(c, 0.0) for c in FEATURE_COLS]
    
    if SCALER and MODELS:
        X = np.array(feat_vec).reshape(1, -1)
        X_sc = SCALER.transform(X)
        model = MODELS.get(model_name, list(MODELS.values())[0])
        prob = float(model.predict_proba(X_sc)[0, 1])
    else:
        # Heuristic fallback
        risk = 0.0
        if features.get('geo_mismatch', 0) == 1: risk += 0.25
        risk += features.get('device_risk', 0) * 0.3
        risk += features.get('ip_risk', 0) * 0.2
        if features.get('velocity', 0) > 10: risk += 0.15
        if features.get('amount', 0) > 1000: risk += 0.1
        prob = min(1.0, risk)
    
    action = risk_action(prob)
    latency_ms = round((time.time() - t0) * 1000, 2)
    
    return {
        "transaction_id": features.get('transaction_id', 'TXN' + str(uuid.uuid4())[:8].upper()),
        "risk_score": round(prob, 4),
        "fraud_probability": round(prob * 100, 2),
        "action": action["action"],
        "risk_level": action["level"],
        "risk_color": action["color"],
        "model_used": model_name,
        "latency_ms": latency_ms,
        "timestamp": datetime.utcnow().isoformat(),
        "feature_importance": feature_importance(model_name, feat_vec),
        "explanation": _explain(prob, features),
    }

def _explain(score: float, features: dict) -> str:
    reasons = []
    if features.get('geo_mismatch') == 1:
        reasons.append("geographic location mismatch detected")
    if features.get('device_risk', 0) > 0.6:
        reasons.append("high-risk device fingerprint")
    if features.get('velocity', 0) > 10:
        reasons.append(f"unusual transaction velocity ({int(features['velocity'])} txns/hr)")
    if features.get('amount', 0) > 2000:
        reasons.append(f"large transaction amount (${features['amount']:.0f})")
    if features.get('ip_risk', 0) > 0.7:
        reasons.append("suspicious IP address")
    if not reasons:
        reasons = ["normal behavioral patterns observed"]
    return "Risk factors: " + "; ".join(reasons) + "."

# ─── FastAPI app ───────────────────────────────────────────────────────────────
if HAS_FASTAPI:
    app = FastAPI(
        title="AI Fraud Detection API",
        description="Real-time fraud detection and risk mitigation",
        version="1.0.0"
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    class TransactionInput(BaseModel):
        transaction_id: Optional[str] = None
        amount: float = Field(ge=0)
        hour: float = Field(ge=0, le=23)
        v1: float = 0.0
        v2: float = 0.0
        v3: float = 0.0
        v4: float = 0.0
        v5: float = 0.0
        velocity: float = Field(ge=0, default=1.0)
        geo_mismatch: int = Field(ge=0, le=1, default=0)
        device_risk: float = Field(ge=0.0, le=1.0, default=0.1)
        ip_risk: float = Field(ge=0.0, le=1.0, default=0.05)
        model_name: Optional[str] = "random_forest"

    @app.get("/")
    def root():
        return {"status": "ok", "service": "AI Fraud Detection API", "version": "1.0.0"}

    @app.get("/health")
    def health():
        return {
            "status": "healthy",
            "models_loaded": list(MODELS.keys()),
            "timestamp": datetime.utcnow().isoformat()
        }

    @app.post("/api/score")
    def score(txn: TransactionInput):
        features = txn.model_dump()
        model_name = features.pop('model_name', 'random_forest')
        return score_transaction(features, model_name or 'random_forest')

    @app.get("/api/models")
    def get_models():
        return {
            "models": list(MODELS.keys()),
            "best_model": META.get("best_model"),
            "results": RESULTS
        }

    @app.get("/api/stats")
    def get_stats():
        """Return dashboard statistics."""
        df = pd.read_csv(os.path.join(DATA_DIR, 'transactions.csv'))
        return {
            "total_transactions": len(df),
            "total_fraud": int(df['is_fraud'].sum()),
            "fraud_rate": round(df['is_fraud'].mean() * 100, 2),
            "avg_amount": round(df['amount'].mean(), 2),
            "max_amount": round(df['amount'].max(), 2),
            "fraud_avg_amount": round(df[df['is_fraud']==1]['amount'].mean(), 2),
        }

    @app.get("/api/transactions")
    def get_recent_transactions(limit: int = 50):
        """Return scored sample transactions."""
        df = pd.read_csv(os.path.join(DATA_DIR, 'transactions.csv'))
        sample = df.sample(min(limit, len(df)), random_state=int(time.time()) % 100)
        records = []
        for _, row in sample.iterrows():
            features = {c: row[c] for c in FEATURE_COLS}
            features['transaction_id'] = row['transaction_id']
            result = score_transaction(features)
            result['actual_fraud'] = int(row['is_fraud'])
            records.append(result)
        return {"transactions": records, "count": len(records)}

    @app.get("/api/model_comparison")
    def model_comparison():
        return {"results": RESULTS, "meta": META}

    if __name__ == "__main__":
        uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)

else:
    # CLI demo mode
    def demo():
        print("\n🔍 AI Fraud Detection — Demo Mode\n")
        test_cases = [
            {"transaction_id":"TXN_SAFE","amount":50.0,"hour":14,"v1":-0.3,"v2":0.2,"v3":0.1,"v4":0.0,"v5":-0.1,"velocity":2,"geo_mismatch":0,"device_risk":0.05,"ip_risk":0.03},
            {"transaction_id":"TXN_FRAUD","amount":3200.0,"hour":2,"v1":-6.0,"v2":4.0,"v3":-5.0,"v4":3.5,"v5":-4.0,"velocity":18,"geo_mismatch":1,"device_risk":0.92,"ip_risk":0.88},
        ]
        for t in test_cases:
            r = score_transaction(t)
            print(f"[{r['transaction_id']}] Score={r['risk_score']:.3f} | Action={r['action']} | {r['explanation']}")

    demo()
