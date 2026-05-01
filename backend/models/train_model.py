"""
Train fraud detection ML models:
- Random Forest
- Gradient Boosting (sklearn)
Handles class imbalance, evaluates with precision/recall/F1/AUC.
"""
import numpy as np
import pandas as pd
import joblib
import json
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, precision_recall_curve, f1_score,
    average_precision_score
)
from sklearn.utils.class_weight import compute_class_weight

FEATURE_COLS = ['amount', 'hour', 'v1', 'v2', 'v3', 'v4', 'v5',
                'velocity', 'geo_mismatch', 'device_risk', 'ip_risk']
TARGET_COL = 'is_fraud'

MODELS_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(MODELS_DIR, '..', 'data', 'transactions.csv')


def oversample_minority(X_train, y_train, ratio=0.1):
    """Simple oversampling to address class imbalance."""
    fraud_idx = np.where(y_train == 1)[0]
    legit_idx = np.where(y_train == 0)[0]
    
    target_fraud = int(len(legit_idx) * ratio)
    if target_fraud <= len(fraud_idx):
        return X_train, y_train
    
    extra = target_fraud - len(fraud_idx)
    chosen = np.random.choice(fraud_idx, size=extra, replace=True)
    X_over = np.vstack([X_train, X_train[chosen]])
    y_over = np.concatenate([y_train, y_train[chosen]])
    shuffle_idx = np.random.permutation(len(X_over))
    return X_over[shuffle_idx], y_over[shuffle_idx]


def train_and_evaluate():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    
    # Oversample minority class
    np.random.seed(42)
    X_train_bal, y_train_bal = oversample_minority(X_train_sc, y_train, ratio=0.15)
    print(f"Training: {len(X_train_bal)} samples, {y_train_bal.sum()} fraud")
    
    # Compute class weights
    classes = np.array([0, 1])
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    cw = {0: class_weights[0], 1: class_weights[1]}
    
    models = {
        'logistic_regression': LogisticRegression(
            class_weight='balanced', max_iter=1000, random_state=42
        ),
        'decision_tree': DecisionTreeClassifier(
            max_depth=8, class_weight='balanced', random_state=42
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=100, max_depth=12, class_weight='balanced',
            random_state=42, n_jobs=-1
        ),
        'gradient_boosting': GradientBoostingClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            subsample=0.8, random_state=42
        ),
    }
    
    results = {}
    best_f1 = 0
    best_model_name = None
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_bal, y_train_bal)
        
        y_pred = model.predict(X_test_sc)
        y_prob = model.predict_proba(X_test_sc)[:, 1]
        
        report = classification_report(y_test, y_pred, output_dict=True)
        auc = roc_auc_score(y_test, y_prob)
        avg_prec = average_precision_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        fraud_metrics = report.get('1', report.get('1.0', {}))
        f1 = fraud_metrics.get('f1-score', 0)
        
        results[name] = {
            'accuracy': report['accuracy'],
            'precision': fraud_metrics.get('precision', 0),
            'recall': fraud_metrics.get('recall', 0),
            'f1_score': f1,
            'roc_auc': auc,
            'avg_precision': avg_prec,
            'confusion_matrix': cm,
            'support': int(fraud_metrics.get('support', 0))
        }
        
        print(f"  F1={f1:.3f} | AUC={auc:.3f} | Precision={results[name]['precision']:.3f} | Recall={results[name]['recall']:.3f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
        
        # Save model
        joblib.dump(model, os.path.join(MODELS_DIR, f'{name}.pkl'))
    
    # Save scaler
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
    
    # Save results and feature cols
    with open(os.path.join(MODELS_DIR, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    meta = {
        'feature_cols': FEATURE_COLS,
        'best_model': best_model_name,
        'best_f1': best_f1,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'fraud_rate_train': float(y_train.mean()),
        'fraud_rate_test': float(y_test.mean()),
    }
    with open(os.path.join(MODELS_DIR, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f"\n✅ Best model: {best_model_name} (F1={best_f1:.3f})")
    print("All models saved.")
    return results, meta


if __name__ == '__main__':
    train_and_evaluate()
