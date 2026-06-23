"""
get_precision_recall.py
Jalankan setelah experiment.py selesai.
Menghasilkan Precision & Recall per kelas untuk semua K + FPR.
"""
import time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, roc_auc_score, confusion_matrix,
    classification_report
)
from scipy.stats import chi2
import xgboost as xgb
import joblib

warnings.filterwarnings("ignore")

DATA_PATH    = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
TARGET_COL   = " Label"
K_VALUES     = [5, 10, 15, 20, 30, "all"]
RANDOM_STATE = 42
TEST_SIZE    = 0.20
OUTPUT_DIR   = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── 1. LOAD & PREPROCESS ─────────────────────────────────────
print("[1/4] Load & preprocess...")
df = pd.read_csv(DATA_PATH, low_memory=False)
df = df[df[TARGET_COL].isin(["BENIGN", "DDoS"])].copy()
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(axis=1, thresh=len(df)*0.50, inplace=True)

le = LabelEncoder()
df[TARGET_COL] = le.fit_transform(df[TARGET_COL])

feature_cols = [c for c in df.columns if c != TARGET_COL]
X = df[feature_cols].values
y = df[TARGET_COL].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train)
X_test  = imputer.transform(X_test)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── 2. IG SCORES ─────────────────────────────────────────────
print("[2/4] Computing IG scores...")
ig_scores = mutual_info_classif(X_train_s, y_train, random_state=RANDOM_STATE)
ig_series = pd.Series(ig_scores, index=feature_cols).sort_values(ascending=False)

# Best XGBoost params (dari experiment sebelumnya)
best_params = {
    'subsample': 0.7, 'n_estimators': 300,
    'max_depth': 4, 'learning_rate': 0.2, 'colsample_bytree': 0.8
}
spw = (y_train == 0).sum() / (y_train == 1).sum()

# ── 3. EVALUASI PER K ────────────────────────────────────────
print("[3/4] Evaluating all K values...")

rows_xgb = []
for k in K_VALUES:
    if k == "all":
        X_tr, X_te = X_train_s, X_test_s
        k_label = f"all (78)"
    else:
        sel = SelectKBest(mutual_info_classif, k=k)
        X_tr = sel.fit_transform(X_train_s, y_train)
        X_te = sel.transform(X_test_s)
        k_label = str(k)

    model = xgb.XGBClassifier(
        **best_params,
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=spw,
        random_state=RANDOM_STATE,
        tree_method="hist",
    )
    t0 = time.perf_counter()
    model.fit(X_tr, y_train)
    train_t = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = model.predict(X_te)
    infer_t = (time.perf_counter() - t0) / len(X_te) * 1000
    y_prob = model.predict_proba(X_te)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) * 100 if (fp+tn) > 0 else 0

    # per-class precision & recall
    prec_ddos   = precision_score(y_test, y_pred, pos_label=1, zero_division=0) * 100
    prec_benign = precision_score(y_test, y_pred, pos_label=0, zero_division=0) * 100
    rec_ddos    = recall_score(y_test, y_pred, pos_label=1, zero_division=0) * 100
    rec_benign  = recall_score(y_test, y_pred, pos_label=0, zero_division=0) * 100
    # macro
    prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=0) * 100
    rec_macro  = recall_score(y_test, y_pred, average='macro', zero_division=0) * 100

    rows_xgb.append({
        "K": k_label,
        "Accuracy": round(accuracy_score(y_test, y_pred)*100, 4),
        "Prec_macro": round(prec_macro, 4),
        "Rec_macro":  round(rec_macro, 4),
        "Prec_DDoS":  round(prec_ddos, 4),
        "Rec_DDoS":   round(rec_ddos, 4),
        "Prec_BENIGN":round(prec_benign, 4),
        "Rec_BENIGN": round(rec_benign, 4),
        "F1_macro":   round(f1_score(y_test, y_pred, average='macro')*100, 4),
        "AUC":        round(roc_auc_score(y_test, y_prob), 6),
        "FPR":        round(fpr, 4),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "Train_s":    round(train_t, 4),
        "Infer_ms":   round(infer_t, 4),
    })
    print(f"  K={k_label:<8} | Acc={rows_xgb[-1]['Accuracy']}% | "
          f"Prec(DDoS)={prec_ddos:.2f}% | Rec(DDoS)={rec_ddos:.2f}% | "
          f"FPR={fpr:.4f}% | TP={tp} FP={fp} FN={fn} TN={tn}")

# ── 4. RF DETAIL ─────────────────────────────────────────────
print("\n[4/4] Random Forest detail (K=30)...")
k_opt = 30
sel_opt = SelectKBest(mutual_info_classif, k=k_opt)
X_tr_opt = sel_opt.fit_transform(X_train_s, y_train)
X_te_opt = sel_opt.transform(X_test_s)

rf = RandomForestClassifier(n_estimators=200, max_features="sqrt",
                            class_weight="balanced",
                            random_state=RANDOM_STATE, n_jobs=-1)
t0 = time.perf_counter()
rf.fit(X_tr_opt, y_train)
rf_train_t = time.perf_counter() - t0

t0 = time.perf_counter()
rf_pred = rf.predict(X_te_opt)
rf_infer_t = (time.perf_counter() - t0) / len(X_te_opt) * 1000
rf_prob = rf.predict_proba(X_te_opt)[:, 1]

rf_cm = confusion_matrix(y_test, rf_pred)
tn_r, fp_r, fn_r, tp_r = rf_cm.ravel()
rf_fpr = fp_r / (fp_r + tn_r) * 100 if (fp_r + tn_r) > 0 else 0

rf_row = {
    "K": "30",
    "Accuracy":   round(accuracy_score(y_test, rf_pred)*100, 4),
    "Prec_macro": round(precision_score(y_test, rf_pred, average='macro')*100, 4),
    "Rec_macro":  round(recall_score(y_test, rf_pred, average='macro')*100, 4),
    "Prec_DDoS":  round(precision_score(y_test, rf_pred, pos_label=1)*100, 4),
    "Rec_DDoS":   round(recall_score(y_test, rf_pred, pos_label=1)*100, 4),
    "Prec_BENIGN":round(precision_score(y_test, rf_pred, pos_label=0)*100, 4),
    "Rec_BENIGN": round(recall_score(y_test, rf_pred, pos_label=0)*100, 4),
    "F1_macro":   round(f1_score(y_test, rf_pred, average='macro')*100, 4),
    "AUC":        round(roc_auc_score(y_test, rf_prob), 6),
    "FPR":        round(rf_fpr, 4),
    "TP": int(tp_r), "FP": int(fp_r), "TN": int(tn_r), "FN": int(fn_r),
    "Train_s":  round(rf_train_t, 4),
    "Infer_ms": round(rf_infer_t, 4),
}
print(f"  RF K=30 | Acc={rf_row['Accuracy']}% | "
      f"Prec(DDoS)={rf_row['Prec_DDoS']:.2f}% | Rec(DDoS)={rf_row['Rec_DDoS']:.2f}% | "
      f"FPR={rf_row['FPR']:.4f}%")

# ── PRINT TABEL LENGKAP ──────────────────────────────────────
print("\n" + "="*90)
print("  TABEL III LENGKAP (XGBoost)")
print("="*90)
hdr = f"{'K':>8} | {'Acc':>7} | {'Prec(M)':>8} | {'Rec(M)':>8} | {'F1(M)':>7} | {'AUC':>7} | {'FPR':>6} | {'Train':>7} | {'Infer':>7}"
print(hdr)
print("-"*90)
for r in rows_xgb:
    print(f"{r['K']:>8} | {r['Accuracy']:>7.2f}% | {r['Prec_macro']:>7.2f}% | "
          f"{r['Rec_macro']:>7.2f}% | {r['F1_macro']:>6.2f}% | {r['AUC']:>7.4f} | "
          f"{r['FPR']:>5.4f}% | {r['Train_s']:>6.2f}s | {r['Infer_ms']:>6.4f}ms")

print("\n" + "="*90)
print("  TABEL IV LENGKAP (XGBoost vs RF, K=30)")
print("="*90)
xgb30 = [r for r in rows_xgb if r['K']=='30'][0]
print(f"{'Metrik':25s} | {'XGBoost':>10} | {'RF':>10} | {'Selisih':>10}")
print("-"*60)
fields = [
    ("Accuracy (%)",    "Accuracy"),
    ("Prec DDoS (%)",   "Prec_DDoS"),
    ("Rec DDoS (%)",    "Rec_DDoS"),
    ("Prec BENIGN (%)", "Prec_BENIGN"),
    ("Rec BENIGN (%)",  "Rec_BENIGN"),
    ("F1-Macro (%)",    "F1_macro"),
    ("AUC-ROC",         "AUC"),
    ("FPR (%)",         "FPR"),
    ("Train Time (s)",  "Train_s"),
    ("Infer Time (ms)", "Infer_ms"),
]
for label, key in fields:
    d = xgb30[key] - rf_row[key]
    print(f"{label:25s} | {xgb30[key]:>10.4f} | {rf_row[key]:>10.4f} | {d:>+10.4f}")

print("\n  Confusion Matrix XGBoost K=30:")
xgb30_cm = [r for r in rows_xgb if r['K']=='30'][0]
print(f"    TP={xgb30_cm['TP']}  FP={xgb30_cm['FP']}  FN={xgb30_cm['FN']}  TN={xgb30_cm['TN']}")
print("\n  Confusion Matrix RF K=30:")
print(f"    TP={rf_row['TP']}  FP={rf_row['FP']}  FN={rf_row['FN']}  TN={rf_row['TN']}")

# ── SIMPAN ───────────────────────────────────────────────────
out = {
    "xgboost_detailed": rows_xgb,
    "rf_detailed": rf_row,
    "ig_top10": ig_series.head(10).to_dict(),
    "ig_top30": ig_series.head(30).to_dict(),
}
with open(OUTPUT_DIR / "results_detailed.json", "w") as f:
    json.dump(out, f, indent=2, default=str)
print("\nSimpan ke output/results_detailed.json")
print("SELESAI!")
