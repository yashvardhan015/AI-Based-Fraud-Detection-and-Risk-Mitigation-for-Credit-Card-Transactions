"""
Generate synthetic credit card fraud dataset for training.
"""
import numpy as np
import pandas as pd
import os

np.random.seed(42)

def generate_fraud_dataset(n_samples=10000, fraud_ratio=0.02):
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    legit_records = []
    for _ in range(n_legit):
        hour = np.random.randint(0, 24)
        amount = np.random.lognormal(mean=3.5, sigma=1.2)
        v1 = np.random.normal(-0.5, 1.2)
        v2 = np.random.normal(0.2, 0.8)
        v3 = np.random.normal(0.1, 1.5)
        v4 = np.random.normal(0.0, 0.9)
        v5 = np.random.normal(-0.2, 1.1)
        velocity = np.random.randint(1, 8)
        geo_mismatch = np.random.choice([0, 1], p=[0.95, 0.05])
        device_risk = np.random.beta(1, 9)
        ip_risk = np.random.beta(1, 10)
        legit_records.append([amount, hour, v1, v2, v3, v4, v5,
                               velocity, geo_mismatch, device_risk, ip_risk, 0])

    fraud_records = []
    for _ in range(n_fraud):
        hour = np.random.choice([0, 1, 2, 3, 22, 23], p=[0.2, 0.2, 0.2, 0.1, 0.15, 0.15])
        amount = np.random.lognormal(mean=4.5, sigma=1.8)
        v1 = np.random.normal(-5.0, 2.5)
        v2 = np.random.normal(3.0, 2.0)
        v3 = np.random.normal(-4.0, 2.5)
        v4 = np.random.normal(2.5, 1.5)
        v5 = np.random.normal(-3.5, 2.0)
        velocity = np.random.randint(5, 25)
        geo_mismatch = np.random.choice([0, 1], p=[0.2, 0.8])
        device_risk = np.random.beta(7, 3)
        ip_risk = np.random.beta(8, 2)
        fraud_records.append([amount, hour, v1, v2, v3, v4, v5,
                               velocity, geo_mismatch, device_risk, ip_risk, 1])

    columns = ['amount', 'hour', 'v1', 'v2', 'v3', 'v4', 'v5',
               'velocity', 'geo_mismatch', 'device_risk', 'ip_risk', 'is_fraud']
    
    all_records = legit_records + fraud_records
    df = pd.DataFrame(all_records, columns=columns)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df['transaction_id'] = ['TXN' + str(100000 + i) for i in range(len(df))]
    return df

if __name__ == '__main__':
    df = generate_fraud_dataset()
    out_path = os.path.join(os.path.dirname(__file__), 'transactions.csv')
    df.to_csv(out_path, index=False)
    print(f"Dataset: {len(df)} rows, {df['is_fraud'].sum()} fraud cases")
    print(df.head())
