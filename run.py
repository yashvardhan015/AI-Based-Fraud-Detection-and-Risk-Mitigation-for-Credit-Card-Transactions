#!/usr/bin/env python3
"""
Quick-start script for FraudShield AI.
Generates data, trains models, then serves the API.
"""
import os
import sys
import subprocess

BASE = os.path.dirname(os.path.abspath(__file__))

def run(cmd, **kwargs):
    print(f"\n⟶  {' '.join(cmd)}")
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        print(f"❌ Command failed with exit code {result.returncode}")
        sys.exit(1)
    return result

def main():
    print("=" * 60)
    print("  🛡️  FraudShield AI — Fraud Detection & Risk Mitigation")
    print("=" * 60)

    # Step 1: Generate dataset
    print("\n[1/3] Generating synthetic fraud dataset...")
    run([sys.executable, os.path.join(BASE, 'backend', 'data', 'generate_dataset.py')])

    # Step 2: Train models
    print("\n[2/3] Training ML models...")
    run([sys.executable, os.path.join(BASE, 'backend', 'models', 'train_model.py')])

    # Step 3: Batch predict demo
    print("\n[3/3] Running batch prediction demo...")
    run([sys.executable, os.path.join(BASE, 'backend', 'utils', 'predict.py'),
         '--model', 'random_forest'])

    print("\n" + "=" * 60)
    print("  ✅ Setup complete!")
    print()
    print("  📂 Project outputs:")
    print("     backend/data/transactions.csv  — Synthetic dataset")
    print("     backend/models/*.pkl           — Trained models")
    print("     backend/models/results.json    — Performance metrics")
    print("     scored_output.csv              — Batch predictions")
    print()
    print("  🌐 To run the dashboard:")
    print("     Open frontend/index.html in your browser")
    print()
    print("  🐳 To deploy with Docker:")
    print("     cd docker && docker-compose up --build")
    print()
    print("  🔗 API (requires FastAPI):")
    print("     pip install fastapi uvicorn")
    print("     uvicorn backend.api.main:app --reload")
    print("     http://localhost:8000/docs")
    print("=" * 60)

if __name__ == '__main__':
    main()
