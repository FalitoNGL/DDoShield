"""
Eksperimen ML: Deteksi DDoS dengan XGBoost + Information Gain
Dataset: CIC-IDS2017 (Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv)

CARA PAKAI:
  1. Letakkan file CSV CIC-IDS2017 di folder yang sama atau ubah DATA_PATH
  2. pip install xgboost scikit-learn pandas numpy matplotlib seaborn
  3. python experiment.py
  4. Hasil otomatis dicetak ke konsol dan disimpan ke results.json
"""

import time
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
from sklearn.inspection import permutation_importance
from scipy.stats import chi2
import xgboost as xgb
import joblib

def mcnemar_test(b, c):
    """McNemar's test (mid-p correction). Returns (statistic, pvalue)."""
    if (b + c) == 0:
        return 0.0, 1.0
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    pval = chi2.sf(stat, df=1)
    return stat, pval

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────
DATA_PATH = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
TARGET_COL = " Label"          # perhatikan spasi — nama kolom di CIC-IDS2017
K_VALUES   = [5, 10, 15, 20, 30, "all"]
RANDOM_STATE = 42
TEST_SIZE    = 0.20
CV_FOLDS     = 5
OUTPUT_DIR   = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("  DDOS DETECTION EXPERIMENT — CIC-IDS2017")
print("=" * 60)
print(f"\n[1/7] Loading dataset: {DATA_PATH}")

df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"  Shape awal        : {df.shape}")
print(f"  Kolom target      : '{TARGET_COL}'")
print(f"  Distribusi kelas  :\n{df[TARGET_COL].value_counts()}")

# ─────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────
print("\n[2/7] Preprocessing...")

# 2a. Ambil hanya kelas DDoS dan BENIGN
df = df[df[TARGET_COL].isin(["BENIGN", "DDoS"])].copy()
print(f"  Setelah filter DDoS+BENIGN : {df.shape}")

# 2b. Ganti Inf/-Inf dengan NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# 2c. Drop kolom dengan >50% NaN
thresh = len(df) * 0.50
df.dropna(axis=1, thresh=thresh, inplace=True)

# 2d. Median imputation untuk sisa NaN
for col in df.select_dtypes(include=[np.number]).columns:
    if df[col].isna().any():
        df[col].fillna(df[col].median(), inplace=True)

# 2e. Label encoding
le = LabelEncoder()
df[TARGET_COL] = le.fit_transform(df[TARGET_COL])
# BENIGN=0, DDoS=1
label_map = dict(zip(le.classes_, le.transform(le.classes_)))
print(f"  Label mapping     : {label_map}")
print(f"  Distribusi kelas  :\n{df[TARGET_COL].value_counts()}")

# 2f. Pisahkan fitur dan target
feature_cols = [c for c in df.columns if c != TARGET_COL]
X = df[feature_cols].values
y = df[TARGET_COL].values
print(f"  Jumlah fitur      : {X.shape[1]}")

# 2g. Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"  Train size        : {X_train.shape[0]}")
print(f"  Test size         : {X_test.shape[0]}")

# 2h. Imputer (median) untuk sisa NaN
imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train)
X_test  = imputer.transform(X_test)

# 2i. Normalisasi
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
joblib.dump(scaler, OUTPUT_DIR / "scaler.pkl")
joblib.dump(imputer, OUTPUT_DIR / "imputer.pkl")

# ─────────────────────────────────────────────
# 3. INFORMATION GAIN FEATURE SELECTION
# ─────────────────────────────────────────────
print("\n[3/7] Menghitung Information Gain untuk semua fitur...")

ig_scores = mutual_info_classif(
    X_train_scaled, y_train,
    random_state=RANDOM_STATE
)
ig_series = pd.Series(ig_scores, index=feature_cols).sort_values(ascending=False)
ig_series.to_csv(OUTPUT_DIR / "ig_scores.csv")
print(f"  Top-10 fitur IG:\n{ig_series.head(10).to_string()}")

# Plot IG scores
plt.figure(figsize=(14, 6))
ig_series.head(30).plot(kind="bar", color="steelblue")
plt.title("Top-30 Fitur Berdasarkan Skor Information Gain")
plt.xlabel("Nama Fitur")
plt.ylabel("Skor IG")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ig_scores_bar.png", dpi=150)
plt.close()
print("  [Gambar disimpan: output/ig_scores_bar.png]")

# ─────────────────────────────────────────────
# 4. HYPERPARAMETER TUNING XGBoost (pada K=all)
# ─────────────────────────────────────────────
print("\n[4/7] Hyperparameter tuning XGBoost (RandomizedSearchCV)...")

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"  scale_pos_weight  : {scale_pos_weight:.4f}")

xgb_param_dist = {
    "n_estimators"    : [100, 200, 300],
    "max_depth"       : [3, 4, 5, 6],
    "learning_rate"   : [0.01, 0.05, 0.1, 0.2],
    "subsample"       : [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
}

xgb_base = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",
    scale_pos_weight=scale_pos_weight,
    random_state=RANDOM_STATE,

    tree_method="hist",
)

rscv = RandomizedSearchCV(
    xgb_base, xgb_param_dist,
    n_iter=20, cv=StratifiedKFold(CV_FOLDS),
    scoring="f1", n_jobs=-1,
    random_state=RANDOM_STATE, verbose=0
)
rscv.fit(X_train_scaled, y_train)
best_params = rscv.best_params_
print(f"  Best params       : {best_params}")

# ─────────────────────────────────────────────
# 5. EKSPERIMEN VARIASI K — XGBoost
# ─────────────────────────────────────────────
print("\n[5/7] Eksperimen variasi K (XGBoost)...")

results_xgb = {}

for k in K_VALUES:
    if k == "all":
        k_label = f"all ({X_train_scaled.shape[1]})"
        X_tr = X_train_scaled
        X_te = X_test_scaled
        selected_features = feature_cols
    else:
        selector = SelectKBest(mutual_info_classif, k=k)
        X_tr = selector.fit_transform(X_train_scaled, y_train)
        X_te = selector.transform(X_test_scaled)
        mask = selector.get_support()
        selected_features = [feature_cols[i] for i, m in enumerate(mask) if m]
        k_label = str(k)

    model = xgb.XGBClassifier(
        **best_params,
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
    
        tree_method="hist",
    )

    t_start = time.perf_counter()
    model.fit(X_tr, y_train)
    train_time = time.perf_counter() - t_start

    t_start = time.perf_counter()
    y_pred = model.predict(X_te)
    infer_time_total = time.perf_counter() - t_start
    infer_time_per = (infer_time_total / len(X_te)) * 1000  # ms per flow

    y_prob = model.predict_proba(X_te)[:, 1]

    acc   = accuracy_score(y_test, y_pred)
    prec  = precision_score(y_test, y_pred, zero_division=0)
    rec   = recall_score(y_test, y_pred, zero_division=0)
    f1    = f1_score(y_test, y_pred, zero_division=0)
    auc   = roc_auc_score(y_test, y_prob)
    cm    = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr   = fp / (fp + tn) if (fp + tn) > 0 else 0

    results_xgb[k_label] = {
        "k": k_label,
        "accuracy"  : round(acc * 100, 4),
        "precision" : round(prec * 100, 4),
        "recall"    : round(rec * 100, 4),
        "f1_score"  : round(f1 * 100, 4),
        "auc_roc"   : round(auc, 6),
        "fpr"       : round(fpr * 100, 4),
        "train_time_s"     : round(train_time, 4),
        "infer_time_ms_flow": round(infer_time_per, 4),
        "selected_features": selected_features,
        "confusion_matrix" : cm.tolist(),
    }

    print(f"  K={k_label:10s} | Acc={acc*100:.2f}% | F1={f1*100:.2f}% | "
          f"AUC={auc:.4f} | TrainT={train_time:.2f}s | InferT={infer_time_per:.3f}ms")

    # Simpan model untuk K optimal (akan di-update)
    if k == "all":
        joblib.dump(model, OUTPUT_DIR / "xgb_model_all.pkl")

# ─────────────────────────────────────────────
# 6. EKSPERIMEN Random Forest (K optimal)
# ─────────────────────────────────────────────
print("\n[6/7] Training Random Forest (baseline)...")

# Temukan K optimal dari XGBoost
best_k_label = max(
    {k: v for k, v in results_xgb.items() if k != f"all ({X_train_scaled.shape[1]})"},
    key=lambda k: results_xgb[k]["f1_score"]
)
print(f"  K optimal XGBoost : {best_k_label}")

# Ambil subset fitur K optimal
k_opt_int = int(best_k_label)
selector_opt = SelectKBest(mutual_info_classif, k=k_opt_int)
X_tr_opt = selector_opt.fit_transform(X_train_scaled, y_train)
X_te_opt = selector_opt.transform(X_test_scaled)
joblib.dump(selector_opt, OUTPUT_DIR / "selector_kopt.pkl")

# RF
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_features="sqrt",
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

t_start = time.perf_counter()
rf_model.fit(X_tr_opt, y_train)
rf_train_time = time.perf_counter() - t_start

t_start = time.perf_counter()
rf_pred = rf_model.predict(X_te_opt)
rf_infer_total = time.perf_counter() - t_start
rf_infer_per = (rf_infer_total / len(X_te_opt)) * 1000

rf_prob = rf_model.predict_proba(X_te_opt)[:, 1]

rf_acc  = accuracy_score(y_test, rf_pred)
rf_prec = precision_score(y_test, rf_pred, zero_division=0)
rf_rec  = recall_score(y_test, rf_pred, zero_division=0)
rf_f1   = f1_score(y_test, rf_pred, zero_division=0)
rf_auc  = roc_auc_score(y_test, rf_prob)
rf_cm   = confusion_matrix(y_test, rf_pred)
tn_r, fp_r, fn_r, tp_r = rf_cm.ravel()
rf_fpr  = fp_r / (fp_r + tn_r) if (fp_r + tn_r) > 0 else 0

results_rf = {
    "accuracy"  : round(rf_acc * 100, 4),
    "precision" : round(rf_prec * 100, 4),
    "recall"    : round(rf_rec * 100, 4),
    "f1_score"  : round(rf_f1 * 100, 4),
    "auc_roc"   : round(rf_auc, 6),
    "fpr"       : round(rf_fpr * 100, 4),
    "train_time_s"      : round(rf_train_time, 4),
    "infer_time_ms_flow": round(rf_infer_per, 4),
    "confusion_matrix"  : rf_cm.tolist(),
}
print(f"  RF (K={best_k_label}) | Acc={rf_acc*100:.2f}% | F1={rf_f1*100:.2f}% | "
      f"AUC={rf_auc:.4f} | TrainT={rf_train_time:.2f}s | InferT={rf_infer_per:.3f}ms")

# McNemar's Test
xgb_opt_pred = np.array(results_xgb[best_k_label]["confusion_matrix"])
# Hitung langsung dari prediksi
xgb_opt_pred_arr = results_xgb[best_k_label]
# Re-predict untuk McNemar
xgb_opt_model = xgb.XGBClassifier(
    **best_params,
    objective="binary:logistic",
    eval_metric="auc",
    scale_pos_weight=scale_pos_weight,
    random_state=RANDOM_STATE,

    tree_method="hist",
)
xgb_opt_model.fit(X_tr_opt, y_train)
xgb_opt_pred_labels = xgb_opt_model.predict(X_te_opt)
joblib.dump(xgb_opt_model, OUTPUT_DIR / "xgb_model_kopt.pkl")

# Tabel kontingensi McNemar
n00 = np.sum((xgb_opt_pred_labels == y_test) & (rf_pred == y_test))
n01 = np.sum((xgb_opt_pred_labels == y_test) & (rf_pred != y_test))
n10 = np.sum((xgb_opt_pred_labels != y_test) & (rf_pred == y_test))
n11 = np.sum((xgb_opt_pred_labels != y_test) & (rf_pred != y_test))
mcnemar_stat, mcnemar_pval = mcnemar_test(n01, n10)
print(f"\n  McNemar's test: chi2={mcnemar_stat:.4f}, p-value={mcnemar_pval:.6f}")

# ─────────────────────────────────────────────
# 7. VISUALISASI
# ─────────────────────────────────────────────
print("\n[7/7] Membuat visualisasi...")

# 7a. Line chart F1 vs K
k_labels_plot = [k for k in results_xgb.keys()]
f1_values     = [results_xgb[k]["f1_score"] for k in k_labels_plot]

plt.figure(figsize=(9, 5))
plt.plot(k_labels_plot, f1_values, marker="o", linewidth=2,
         color="royalblue", markersize=8)
plt.title("F1-Score XGBoost vs. Jumlah Fitur (K)")
plt.xlabel("Nilai K")
plt.ylabel("F1-Score (%)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "f1_vs_k.png", dpi=150)
plt.close()

# 7b. Bar chart Training Time & Inference Time: K_all vs K_opt
all_label = f"all ({X_train_scaled.shape[1]})"
cats = [f"K={best_k_label} (optimal)", f"K=all (78 fitur)"]
train_times = [results_xgb[best_k_label]["train_time_s"],
               results_xgb[all_label]["train_time_s"]]
infer_times = [results_xgb[best_k_label]["infer_time_ms_flow"],
               results_xgb[all_label]["infer_time_ms_flow"]]

x = np.arange(len(cats))
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
axes[0].bar(cats, train_times, color=["steelblue", "salmon"])
axes[0].set_title("Training Time (detik)")
axes[0].set_ylabel("Waktu (s)")
axes[1].bar(cats, infer_times, color=["steelblue", "salmon"])
axes[1].set_title("Inference Time per Flow (ms)")
axes[1].set_ylabel("Waktu (ms)")
plt.suptitle("Perbandingan Efisiensi: K Optimal vs. K=All")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "efficiency_comparison.png", dpi=150)
plt.close()

# 7c. Confusion Matrix berdampingan
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, cm_data, title in zip(
    axes,
    [confusion_matrix(y_test, xgb_opt_pred_labels), rf_cm],
    [f"XGBoost (K={best_k_label})", f"Random Forest (K={best_k_label})"]
):
    sns.heatmap(cm_data, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["BENIGN", "DDoS"], yticklabels=["BENIGN", "DDoS"])
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrices.png", dpi=150)
plt.close()

# 7d. ROC Curve overlay
from sklearn.metrics import roc_curve
xgb_opt_prob = xgb_opt_model.predict_proba(X_te_opt)[:, 1]
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_opt_prob)
fpr_rf,  tpr_rf,  _ = roc_curve(y_test, rf_prob)

plt.figure(figsize=(7, 6))
plt.plot(fpr_xgb, tpr_xgb, label=f"XGBoost (AUC={roc_auc_score(y_test, xgb_opt_prob):.4f})",
         linewidth=2, color="royalblue")
plt.plot(fpr_rf,  tpr_rf,  label=f"Random Forest (AUC={rf_auc:.4f})",
         linewidth=2, color="darkorange", linestyle="--")
plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — XGBoost vs. Random Forest")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "roc_curve.png", dpi=150)
plt.close()

# ─────────────────────────────────────────────
# RINGKASAN HASIL (untuk paper)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  RINGKASAN HASIL — ISI KE PAPER")
print("=" * 60)

print("\n▶ TABEL III — PERFORMA XGBOOST PER VARIASI K:")
print(f"{'K':>10} | {'Acc':>8} | {'Prec':>8} | {'Recall':>8} | "
      f"{'F1':>8} | {'AUC':>8} | {'Train(s)':>10} | {'Infer(ms)':>10}")
print("-" * 85)
for k, v in results_xgb.items():
    print(f"{k:>10} | {v['accuracy']:>7.2f}% | {v['precision']:>7.2f}% | "
          f"{v['recall']:>7.2f}% | {v['f1_score']:>7.2f}% | {v['auc_roc']:>8.4f} | "
          f"{v['train_time_s']:>9.2f}s | {v['infer_time_ms_flow']:>8.4f}ms")

print(f"\n▶ K OPTIMAL: {best_k_label}")
print(f"  Pengurangan Train Time vs K=all: "
      f"{(1 - results_xgb[best_k_label]['train_time_s']/results_xgb[all_label]['train_time_s'])*100:.1f}%")
print(f"  Pengurangan Infer Time vs K=all: "
      f"{(1 - results_xgb[best_k_label]['infer_time_ms_flow']/results_xgb[all_label]['infer_time_ms_flow'])*100:.1f}%")

print(f"\n▶ TABEL IV — XGBOOST vs. RANDOM FOREST (K={best_k_label}):")
xgb_opt = results_xgb[best_k_label]
print(f"  {'Metrik':20s} | {'XGBoost':>10} | {'RF':>10} | {'Selisih':>10}")
print(f"  {'-'*55}")
for m in ["accuracy","precision","recall","f1_score","auc_roc","fpr","train_time_s","infer_time_ms_flow"]:
    diff = xgb_opt[m] - results_rf[m]
    print(f"  {m:20s} | {xgb_opt[m]:>10.4f} | {results_rf[m]:>10.4f} | {diff:>+10.4f}")

print(f"\n[McNemar] p-value = {mcnemar_pval:.6f} ({'SIGNIFIKAN (p<0.05)' if mcnemar_pval < 0.05 else 'TIDAK SIGNIFIKAN (p>=0.05)'})")  # fixed

print(f"\n▶ TOP-5 FITUR (K={best_k_label}):")
mask_opt = selector_opt.get_support()
top5 = [feature_cols[i] for i, m in enumerate(mask_opt) if m][:5]
for i, f in enumerate(top5, 1):
    print(f"  {i}. {f} (IG={ig_series[f]:.6f})")

# ─────────────────────────────────────────────
# SIMPAN HASIL KE JSON
# ─────────────────────────────────────────────
output_summary = {
    "dataset"    : DATA_PATH,
    "n_features_original": X_train_scaled.shape[1],
    "best_k"     : best_k_label,
    "best_params_xgb" : best_params,
    "scale_pos_weight": round(float(scale_pos_weight), 4),
    "mcnemar"    : {
        "statistic": round(float(mcnemar_stat), 4),
        "pvalue"   : round(float(mcnemar_pval), 6),
        "significant": bool(mcnemar_pval < 0.05)
    },
    "xgboost_results" : results_xgb,
    "rf_results"      : results_rf,
    "top5_features"   : top5,
    "ig_top10"        : ig_series.head(10).to_dict(),
}

with open(OUTPUT_DIR / "results.json", "w") as f:
    json.dump(output_summary, f, indent=2, default=str)

print(f"\n✅ Semua hasil disimpan ke folder: {OUTPUT_DIR}/")
print("   - results.json          (semua angka untuk paper)")
print("   - ig_scores.csv         (skor IG semua fitur)")
print("   - ig_scores_bar.png     (Gambar 3 paper)")
print("   - f1_vs_k.png           (Gambar 5 paper)")
print("   - efficiency_comparison.png (Gambar 6 paper)")
print("   - confusion_matrices.png    (Gambar 7 paper)")
print("   - roc_curve.png             (Gambar 8 paper)")
print("   - xgb_model_kopt.pkl    (model terlatih)")
print("   - scaler.pkl            (untuk DDoShield)")
print("   - selector_kopt.pkl     (untuk DDoShield)")
print("\n🎯 Salin angka dari output di atas ke [FILL] di paper!")
