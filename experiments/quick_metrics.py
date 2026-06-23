"""
quick_metrics.py — hanya K=30, tanpa tuning ulang.
Selesai dalam ~3-5 menit.
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
import xgboost as xgb

warnings.filterwarnings("ignore")

DATA_PATH    = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
TARGET_COL   = " Label"
RANDOM_STATE = 42
OUTPUT_DIR   = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Best params dari eksperimen sebelumnya
BEST_PARAMS = {
    'subsample': 0.7, 'n_estimators': 300,
    'max_depth': 4, 'learning_rate': 0.2, 'colsample_bytree': 0.8
}

print("[1/4] Load & preprocess...")
df = pd.read_csv(DATA_PATH, low_memory=False)
df = df[df[TARGET_COL].isin(["BENIGN","DDoS"])].copy()
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(axis=1, thresh=len(df)*0.50, inplace=True)

le = LabelEncoder()
df[TARGET_COL] = le.fit_transform(df[TARGET_COL])
feature_cols = [c for c in df.columns if c != TARGET_COL]
X = df[feature_cols].values
y = df[TARGET_COL].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)

imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train)
X_test  = imputer.transform(X_test)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
spw = (y_train == 0).sum() / (y_train == 1).sum()

print("[2/4] Information Gain & SelectKBest K=30...")
t0 = time.perf_counter()
sel = SelectKBest(mutual_info_classif, k=30)
X_tr30 = sel.fit_transform(X_train_s, y_train)
X_te30 = sel.transform(X_test_s)
ig_scores = mutual_info_classif(X_train_s, y_train, random_state=RANDOM_STATE)
ig_series = pd.Series(ig_scores, index=feature_cols).sort_values(ascending=False)
print(f"  IG done in {time.perf_counter()-t0:.1f}s")

# Fitur terpilih
mask = sel.get_support()
selected_feats = [feature_cols[i] for i, m in enumerate(mask) if m]

print("[3/4] XGBoost K=30...")
xgb_model = xgb.XGBClassifier(
    **BEST_PARAMS,
    objective="binary:logistic",
    eval_metric="auc",
    scale_pos_weight=spw,
    random_state=RANDOM_STATE,
    tree_method="hist",
)
t0 = time.perf_counter()
xgb_model.fit(X_tr30, y_train)
xgb_train_t = time.perf_counter() - t0

t0 = time.perf_counter()
xgb_pred = xgb_model.predict(X_te30)
xgb_infer_t = (time.perf_counter() - t0) / len(X_te30) * 1000
xgb_prob = xgb_model.predict_proba(X_te30)[:, 1]

xgb_cm = confusion_matrix(y_test, xgb_pred)
tn_x, fp_x, fn_x, tp_x = xgb_cm.ravel()

xgb_metrics = {
    "accuracy":    round(accuracy_score(y_test, xgb_pred)*100, 4),
    "prec_macro":  round(precision_score(y_test, xgb_pred, average='macro')*100, 4),
    "rec_macro":   round(recall_score(y_test, xgb_pred, average='macro')*100, 4),
    "prec_ddos":   round(precision_score(y_test, xgb_pred, pos_label=1)*100, 4),
    "rec_ddos":    round(recall_score(y_test, xgb_pred, pos_label=1)*100, 4),
    "prec_benign": round(precision_score(y_test, xgb_pred, pos_label=0)*100, 4),
    "rec_benign":  round(recall_score(y_test, xgb_pred, pos_label=0)*100, 4),
    "f1_macro":    round(f1_score(y_test, xgb_pred, average='macro')*100, 4),
    "auc":         round(roc_auc_score(y_test, xgb_prob), 6),
    "fpr":         round(fp_x/(fp_x+tn_x)*100, 6),
    "tp": int(tp_x), "fp": int(fp_x), "fn": int(fn_x), "tn": int(tn_x),
    "train_s":     round(xgb_train_t, 3),
    "infer_ms":    round(xgb_infer_t, 4),
}

print("[4/4] Random Forest K=30...")
rf = RandomForestClassifier(n_estimators=200, max_features="sqrt",
                            class_weight="balanced",
                            random_state=RANDOM_STATE, n_jobs=-1)
t0 = time.perf_counter()
rf.fit(X_tr30, y_train)
rf_train_t = time.perf_counter() - t0

t0 = time.perf_counter()
rf_pred = rf.predict(X_te30)
rf_infer_t = (time.perf_counter() - t0) / len(X_te30) * 1000
rf_prob = rf.predict_proba(X_te30)[:, 1]

rf_cm = confusion_matrix(y_test, rf_pred)
tn_r, fp_r, fn_r, tp_r = rf_cm.ravel()

rf_metrics = {
    "accuracy":    round(accuracy_score(y_test, rf_pred)*100, 4),
    "prec_macro":  round(precision_score(y_test, rf_pred, average='macro')*100, 4),
    "rec_macro":   round(recall_score(y_test, rf_pred, average='macro')*100, 4),
    "prec_ddos":   round(precision_score(y_test, rf_pred, pos_label=1)*100, 4),
    "rec_ddos":    round(recall_score(y_test, rf_pred, pos_label=1)*100, 4),
    "prec_benign": round(precision_score(y_test, rf_pred, pos_label=0)*100, 4),
    "rec_benign":  round(recall_score(y_test, rf_pred, pos_label=0)*100, 4),
    "f1_macro":    round(f1_score(y_test, rf_pred, average='macro')*100, 4),
    "auc":         round(roc_auc_score(y_test, rf_prob), 6),
    "fpr":         round(fp_r/(fp_r+tn_r)*100, 6),
    "tp": int(tp_r), "fp": int(fp_r), "fn": int(fn_r), "tn": int(tn_r),
    "train_s":     round(rf_train_t, 3),
    "infer_ms":    round(rf_infer_t, 4),
}

# ── PRINT RINGKASAN ──────────────────────────────────────────
print("\n" + "="*65)
print("  HASIL DETAIL K=30 — SALIN KE PAPER")
print("="*65)
print(f"\n[XGBoost K=30]")
print(f"  Accuracy    : {xgb_metrics['accuracy']}%")
print(f"  Prec (DDoS) : {xgb_metrics['prec_ddos']}%")
print(f"  Rec  (DDoS) : {xgb_metrics['rec_ddos']}%")
print(f"  Prec(BENIGN): {xgb_metrics['prec_benign']}%")
print(f"  Rec (BENIGN): {xgb_metrics['rec_benign']}%")
print(f"  Prec (macro): {xgb_metrics['prec_macro']}%")
print(f"  Rec  (macro): {xgb_metrics['rec_macro']}%")
print(f"  F1   (macro): {xgb_metrics['f1_macro']}%")
print(f"  AUC-ROC     : {xgb_metrics['auc']}")
print(f"  FPR         : {xgb_metrics['fpr']}%")
print(f"  Train Time  : {xgb_metrics['train_s']}s")
print(f"  Infer Time  : {xgb_metrics['infer_ms']}ms/flow")
print(f"  TP={xgb_metrics['tp']}  FP={xgb_metrics['fp']}  FN={xgb_metrics['fn']}  TN={xgb_metrics['tn']}")

print(f"\n[Random Forest K=30]")
print(f"  Accuracy    : {rf_metrics['accuracy']}%")
print(f"  Prec (DDoS) : {rf_metrics['prec_ddos']}%")
print(f"  Rec  (DDoS) : {rf_metrics['rec_ddos']}%")
print(f"  Prec(BENIGN): {rf_metrics['prec_benign']}%")
print(f"  Rec (BENIGN): {rf_metrics['rec_benign']}%")
print(f"  Prec (macro): {rf_metrics['prec_macro']}%")
print(f"  Rec  (macro): {rf_metrics['rec_macro']}%")
print(f"  F1   (macro): {rf_metrics['f1_macro']}%")
print(f"  AUC-ROC     : {rf_metrics['auc']}")
print(f"  FPR         : {rf_metrics['fpr']}%")
print(f"  Train Time  : {rf_metrics['train_s']}s")
print(f"  Infer Time  : {rf_metrics['infer_ms']}ms/flow")
print(f"  TP={rf_metrics['tp']}  FP={rf_metrics['fp']}  FN={rf_metrics['fn']}  TN={rf_metrics['tn']}")

print(f"\n[Top-10 Fitur IG terpilih K=30]")
for i, f in enumerate(ig_series.index[:10], 1):
    sel_mark = "*" if f in selected_feats else " "
    print(f"  {sel_mark}{i:2d}. {f:<40} IG={ig_series[f]:.6f}")

# ── SIMPAN JSON ───────────────────────────────────────────────
result = {
    "xgb_k30": xgb_metrics,
    "rf_k30": rf_metrics,
    "selected_features_k30": selected_feats,
    "ig_top10": ig_series.head(10).to_dict(),
}
with open(OUTPUT_DIR / "results_k30_detail.json", "w") as f:
    json.dump(result, f, indent=2, default=str)
print(f"\nSimpan ke output/results_k30_detail.json")
print("SELESAI!")
