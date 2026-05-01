"""
Standalone prediction utility for batch scoring.
Usage: python predict.py --input transactions.csv --output scored.csv
"""
import argparse
import json
import os
import sys
import pandas as pd
import numpy as np
import joblib

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')

FEATURE_COLS = ['amount', 'hour', 'v1', 'v2', 'v3', 'v4', 'v5',
                'velocity', 'geo_mismatch', 'device_risk', 'ip_risk']

def load_model(model_name='random_forest'):
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    model  = joblib.load(os.path.join(MODELS_DIR, f'{model_name}.pkl'))
    return scaler, model

def predict_batch(df: pd.DataFrame, model_name='random_forest') -> pd.DataFrame:
    scaler, model = load_model(model_name)
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    X = df[FEATURE_COLS].fillna(0).values
    X_sc = scaler.transform(X)
    probs = model.predict_proba(X_sc)[:, 1]
    
    df = df.copy()
    df['risk_score']   = probs.round(4)
    df['fraud_pct']    = (probs * 100).round(2)
    df['risk_action']  = np.where(probs>=0.8,'BLOCK',
                         np.where(probs>=0.5,'REVIEW',
                         np.where(probs>=0.25,'VERIFY','APPROVE')))
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch fraud scoring')
    parser.add_argument('--input',  default='backend/data/transactions.csv')
    parser.add_argument('--output', default='scored_output.csv')
    parser.add_argument('--model',  default='random_forest',
                        choices=['random_forest','gradient_boosting','logistic_regression','decision_tree'])
    args = parser.parse_args()
    
    print(f"Loading {args.input}...")
    df = pd.read_csv(args.input)
    print(f"Scoring {len(df)} transactions with {args.model}...")
    result = predict_batch(df, args.model)
    result.to_csv(args.output, index=False)
    
    print(f"\n✅ Saved to {args.output}")
    print(f"   BLOCK:   {(result.risk_action=='BLOCK').sum()}")
    print(f"   REVIEW:  {(result.risk_action=='REVIEW').sum()}")
    print(f"   VERIFY:  {(result.risk_action=='VERIFY').sum()}")
    print(f"   APPROVE: {(result.risk_action=='APPROVE').sum()}")
