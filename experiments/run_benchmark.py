import time
import joblib
import psutil
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix

print("=== DDOSHIELD PERFORMANCE BENCHMARK ===")

# Paths
DATA_PATH = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
OUTPUT_DIR = Path("output")
MODEL_DIR = Path("DDoShield/models")

# Load models
print("Loading models...")
imputer = joblib.load(MODEL_DIR / "imputer.pkl")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")
selector = joblib.load(MODEL_DIR / "selector_kopt.pkl")
model = joblib.load(MODEL_DIR / "xgb_model_kopt.pkl")
print("Models loaded successfully.")

# Get initial CPU & RAM
process = psutil.Process(os.getpid())
# Warm up system metrics
psutil.cpu_percent(interval=0.1)

mem_idle_mb = process.memory_info().rss / (1024 * 1024)
cpu_idle_pct = psutil.cpu_percent(interval=0.5)
print(f"System Idle -> RAM: {mem_idle_mb:.2f} MB, CPU: {cpu_idle_pct:.2f}%")

# Load dataset
print("Loading dataset...")
df = pd.read_csv(DATA_PATH, low_memory=False)
TARGET_COL = " Label"
df = df[df[TARGET_COL].isin(["BENIGN", "DDoS"])].copy()
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Separate benign and ddos
df_benign = df[df[TARGET_COL] == "BENIGN"].copy()
df_ddos = df[df[TARGET_COL] == "DDoS"].copy()

print(f"Dataset loaded: {len(df)} total rows. Benign: {len(df_benign)}, DDoS: {len(df_ddos)}")

feature_cols = [c for c in df.columns if c != TARGET_COL]

# Helper function to run the prediction pipeline and measure time
def run_pipeline(data_slice):
    # Extract features
    X = data_slice[feature_cols].values
    
    # 1. Impute
    X_imputed = imputer.transform(X)
    # 2. Scale
    X_scaled = scaler.transform(X_imputed)
    # 3. Select K=30
    X_k30 = selector.transform(X_scaled)
    # 4. Predict
    preds = model.predict(X_k30)
    return preds

# Benchmark Latency & Throughput
results = {}

for size in [100, 1000, 10000]:
    print(f"\nBenchmarking {size} flows...")
    # Sample a mixture of Benign and DDoS
    sample_df = df.sample(n=size, random_state=42)
    
    # Warmup
    _ = run_pipeline(df.sample(n=10, random_state=123))
    
    # Measure latency
    t0 = time.perf_counter()
    preds = run_pipeline(sample_df)
    t1 = time.perf_counter()
    
    elapsed_ms = (t1 - t0) * 1000
    throughput = size / (t1 - t0)
    
    # Measure CPU and memory during active execution
    mem_active_mb = process.memory_info().rss / (1024 * 1024)
    # Get CPU usage over a short window
    cpu_active_pct = psutil.cpu_percent(interval=0.1)
    
    print(f"Size {size} -> Latency: {elapsed_ms:.4f} ms, Throughput: {throughput:.2f} flow/s")
    print(f"Size {size} -> Active RAM: {mem_active_mb:.2f} MB, CPU: {cpu_active_pct:.2f}%")
    
    results[size] = {
        "latency_ms": elapsed_ms,
        "throughput": throughput,
        "mem_mb": mem_active_mb,
        "cpu_pct": cpu_active_pct
    }

# FPR Benchmark (BENIGN only)
print("\nBenchmarking FPR on Benign-only flows...")
benign_sample = df_benign.sample(n=10000, random_state=42) if len(df_benign) >= 10000 else df_benign
preds_benign = run_pipeline(benign_sample)
# Map predictions (BENIGN=0, DDoS=1)
# Note: check target encoding in model. Let's see if BENIGN=0 and DDoS=1.
# In quick_metrics.py: df[TARGET_COL] = le.fit_transform(df[TARGET_COL])
# Since "BENIGN" comes alphabetically before "DDoS", BENIGN=0 and DDoS=1.
# Let's check how many were predicted as DDoS (1).
fp = np.sum(preds_benign == 1)
tn = np.sum(preds_benign == 0)
fpr_pct = (fp / (fp + tn)) * 100
print(f"FP: {fp}, TN: {tn}, FPR: {fpr_pct:.6f}%")

# Save summary
print("\n=== LATEX ROWS ===")
print(f"Latensi (100 flow)   & {results[100]['latency_ms']:.2f} & ms  & .pcap kecil \\\\")
print(f"Latensi (1000 flow)  & {results[1000]['latency_ms']:.2f} & ms  & .pcap sedang \\\\")
print(f"Latensi (10000 flow) & {results[10000]['latency_ms']:.2f} & ms  & .pcap besar \\\\")
print(f"Throughput           & {results[1000]['throughput']:.2f} & flow/s & .pcap sedang \\\\")
print(f"CPU idle             & {cpu_idle_pct:.1f} & % & Server idle \\\\")
print(f"CPU aktif            & {results[1000]['cpu_pct']:.1f} & % & 1000 flow \\\\")
print(f"Memory idle          & {mem_idle_mb:.1f} & MB & Server idle \\\\")
print(f"Memory aktif         & {results[1000]['mem_mb']:.1f} & MB & 1000 flow \\\\")
print(f"FPR (BENIGN only)    & {fpr_pct:.4f} & % & 100% normal \\\\")

print("\nSaving results...")
import json
with open(OUTPUT_DIR / "benchmark_results.json", "w") as f:
    json.dump({
        "cpu_idle": cpu_idle_pct,
        "mem_idle_mb": mem_idle_mb,
        "results": results,
        "fpr_pct": fpr_pct
    }, f, indent=2)
print("Benchmark completed.")
