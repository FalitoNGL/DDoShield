import json

with open('output/results.json', encoding='utf-8') as f:
    r = json.load(f)

print('=== HASIL EKSPERIMEN ===')
print('Dataset:', r['dataset'])
print('K Optimal:', r['best_k'])
print('Best XGBoost params:', r['best_params_xgb'])
print()
print('=== TABEL III: XGBoost per K ===')
for k, v in r['xgboost_results'].items():
    acc = v["accuracy"]
    f1  = v["f1_score"]
    auc = v["auc_roc"]
    tr  = v["train_time_s"]
    inf = v["infer_time_ms_flow"]
    print(f'K={k:<12} Acc={acc}%  F1={f1}%  AUC={auc}  Train={tr}s  Infer={inf}ms')

print()
print('=== TABEL IV: XGBoost vs RF (K optimal) ===')
xgb = r['xgboost_results'][r['best_k']]
rf  = r['rf_results']
for m in ['accuracy','precision','recall','f1_score','auc_roc','fpr','train_time_s','infer_time_ms_flow']:
    diff = xgb[m] - rf[m]
    print(f'{m:<25}  XGB={xgb[m]:>10.4f}  RF={rf[m]:>10.4f}  diff={diff:>+.4f}')

print()
sig = r['mcnemar']['significant']
pv  = r['mcnemar']['pvalue']
print(f'McNemar p-value: {pv}  ->  {"SIGNIFIKAN" if sig else "TIDAK SIGNIFIKAN"}')

print()
print('Top-5 fitur IG:')
for i, fname in enumerate(r['top5_features'], 1):
    print(f'  {i}. {fname}')

print()
print('IG Top-10:')
for fname, score in r['ig_top10'].items():
    print(f'  {fname}: {score:.6f}')
