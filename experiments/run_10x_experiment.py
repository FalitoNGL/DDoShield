import time
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
import xgboost as xgb

warnings.filterwarnings("ignore")

DATA_PATH = r"C:\Users\lenovo\Documents\SEMESTER 4\Keamanan Jaringan\Proyek Kelompok\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
TARGET_COL = " Label"
TEST_SIZE = 0.20
K_OPT = 30
SEEDS = [42, 7, 123, 99, 256, 1024, 2048, 777, 888, 999]
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("  10x REPEATED RANDOM SUB-SAMPLING VALIDATION")
print("=" * 60)

# Load data once
print(f"Loading dataset: {DATA_PATH}")
df = pd.read_csv(DATA_PATH, low_memory=False)
df = df[df[TARGET_COL].isin(["BENIGN", "DDoS"])].copy()
df.replace([np.inf, -np.inf], np.nan, inplace=True)

thresh = len(df) * 0.50
df.dropna(axis=1, thresh=thresh, inplace=True)

for col in df.select_dtypes(include=[np.number]).columns:
    if df[col].isna().any():
        df[col].fillna(df[col].median(), inplace=True)

le = LabelEncoder()
df[TARGET_COL] = le.fit_transform(df[TARGET_COL])

feature_cols = [c for c in df.columns if c != TARGET_COL]
X = df[feature_cols].values
y = df[TARGET_COL].values

print(f"Data ready. Total samples: {X.shape[0]}, Features: {X.shape[1]}")

results_xgb = []
results_rf = []

for idx, seed in enumerate(SEEDS):
    print(f"\n--- Iteration {idx+1}/10 (Seed: {seed}) ---")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=seed, stratify=y
    )
    
    # Impute
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_test  = imputer.transform(X_test)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    
    # Feature Selection (K=30)
    ig_scores = mutual_info_classif(X_train_scaled, y_train, random_state=seed)
    selector = SelectKBest(score_func=lambda X, y: ig_scores, k=K_OPT)
    X_train_k = selector.fit_transform(X_train_scaled, y_train)
    X_test_k  = selector.transform(X_test_scaled)
    
    # ---------------- XGBoost ----------------
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    clf_xgb = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.2,
        subsample=0.7,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        n_jobs=-1
    )
    
    t0 = time.time()
    clf_xgb.fit(X_train_k, y_train)
    train_time_xgb = time.time() - t0
    
    t0 = time.time()
    preds_xgb = clf_xgb.predict(X_test_k)
    inf_time_xgb = (time.time() - t0) / len(X_test_k) * 1000
    
    res_xgb = {
        "acc": accuracy_score(y_test, preds_xgb),
        "prec": precision_score(y_test, preds_xgb, average='macro'),
        "rec": recall_score(y_test, preds_xgb, average='macro'),
        "f1": f1_score(y_test, preds_xgb, average='macro'),
        "auc": roc_auc_score(y_test, clf_xgb.predict_proba(X_test_k)[:, 1]),
        "train_time": train_time_xgb,
        "inf_time": inf_time_xgb
    }
    results_xgb.append(res_xgb)
    print(f"  XGBoost -> F1: {res_xgb['f1']:.6f}, Train Time: {train_time_xgb:.2f}s")
    
    # ---------------- Random Forest ----------------
    clf_rf = RandomForestClassifier(
        n_estimators=200,
        max_features='sqrt',
        class_weight='balanced',
        random_state=seed,
        n_jobs=-1
    )
    
    t0 = time.time()
    clf_rf.fit(X_train_k, y_train)
    train_time_rf = time.time() - t0
    
    t0 = time.time()
    preds_rf = clf_rf.predict(X_test_k)
    inf_time_rf = (time.time() - t0) / len(X_test_k) * 1000
    
    res_rf = {
        "acc": accuracy_score(y_test, preds_rf),
        "prec": precision_score(y_test, preds_rf, average='macro'),
        "rec": recall_score(y_test, preds_rf, average='macro'),
        "f1": f1_score(y_test, preds_rf, average='macro'),
        "auc": roc_auc_score(y_test, clf_rf.predict_proba(X_test_k)[:, 1]),
        "train_time": train_time_rf,
        "inf_time": inf_time_rf
    }
    results_rf.append(res_rf)
    print(f"  RF      -> F1: {res_rf['f1']:.6f}, Train Time: {train_time_rf:.2f}s")

print("\n" + "=" * 60)
print("  FINAL RESULTS (Mean ± Std Dev over 10 runs)")
print("=" * 60)

def print_stats(model_name, results_list):
    print(f"\n{model_name}:")
    metrics = ["acc", "prec", "rec", "f1", "auc", "train_time", "inf_time"]
    for m in metrics:
        vals = [r[m] for r in results_list]
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        
        if 'time' in m:
            print(f"  {m:10}: {mean_val:.4f} ± {std_val:.4f}")
        else:
            print(f"  {m:10}: {mean_val*100:.4f}% ± {std_val*100:.4f}%")

print_stats("XGBoost", results_xgb)
print_stats("Random Forest", results_rf)

with open(OUTPUT_DIR / "10x_results.json", "w") as f:
    json.dump({"xgb": results_xgb, "rf": results_rf}, f, indent=4)
print("\n[Done] Results saved to output/10x_results.json")
