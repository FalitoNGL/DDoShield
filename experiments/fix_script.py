"""Fix script: perbaiki baris bermasalah di experiment.py"""
with open('experiment.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

fixed = []
skip_next = False
for i, line in enumerate(lines):
    if 'mcnemar_result.pvalue' in line:
        fixed.append(f"print(f\"\\n[McNemar] p-value = {{mcnemar_pval:.6f}} ({{'SIGNIFIKAN (p<0.05)' if mcnemar_pval < 0.05 else 'TIDAK SIGNIFIKAN (p>=0.05)'}})\")  # fixed\n")
        skip_next = True
    elif skip_next and line.strip().startswith('f"('):
        skip_next = False  # skip continuation line
    else:
        skip_next = False
        fixed.append(line)

with open('experiment.py', 'w', encoding='utf-8') as f:
    f.writelines(fixed)

print("Done! Lines fixed:")
with open('experiment.py', 'r', encoding='utf-8') as f:
    for i, l in enumerate(f.readlines(), 1):
        if 'McNemar' in l or 'mcnemar' in l.lower():
            print(f"  {i}: {l.rstrip()}")
