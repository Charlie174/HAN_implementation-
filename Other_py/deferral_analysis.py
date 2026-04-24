"""
deferral_analysis.py — Day 3 of 13-day plan
============================================
Links MC Dropout uncertainty (σ) to actual prediction errors.
For each σ threshold: compute deferral rate, deferral precision, and
F1-Macro on retained (non-deferred) patients.

Inputs (from evaluate_v6.py):
  output/careai_march/eval_outputs/test_probs.npy    [N_test, 9]
  output/careai_march/eval_outputs/test_labels.npy   [N_test, 9]
  output/careai_march/eval_outputs/mc_sigma.npy      [N_test, 9]
  output/careai_march/inductive_schema.json           (opt_thresholds)

Outputs:
  output/careai_march/deferral_analysis.json
  output/careai_march/deferral_curve.png

Usage:
    python Other_py/deferral_analysis.py
"""

import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

EVAL_DIR    = 'output/careai_march/eval_outputs'
SCHEMA_PATH = 'output/careai_march/inductive_schema.json'
OUT_DIR     = 'output/careai_march'

DISEASE_NAMES = [
    'Anemia', 'CKD', 'Diabetes', 'Dyslipidemia',
    'Electrolyte_Imbalance', 'Hematology_Disorder',
    'Infection_Inflammation', 'Liver_Disease', 'Thyroid_Disorder'
]
PAPER_DISEASES = [
    'Anemia', 'CKD', 'Diabetes', 'Dyslipidemia',
    'Electrolyte_Imbalance', 'Hematology_Disorder', 'Liver_Disease', 'Thyroid_Disorder'
]
PAPER_IDX = [i for i, d in enumerate(DISEASE_NAMES) if d in PAPER_DISEASES]

# ── Load ──────────────────────────────────────────────────────────────────────
print("[Load] Reading eval_outputs ...")
test_probs  = np.load(os.path.join(EVAL_DIR, 'test_probs.npy'))
test_labels = np.load(os.path.join(EVAL_DIR, 'test_labels.npy'))
mc_sigma    = np.load(os.path.join(EVAL_DIR, 'mc_sigma.npy'))

with open(SCHEMA_PATH) as f:
    schema = json.load(f)
opt_thresholds = schema['opt_thresholds']

thresholds = np.array([opt_thresholds.get(d, 0.5) for d in DISEASE_NAMES])
preds_all  = (test_probs >= thresholds).astype(int)

N = test_probs.shape[0]
print(f"  N_test={N:,}  Loaded sigma shape={mc_sigma.shape}")

# Per-patient max σ across the 8 paper diseases
sigma_max = mc_sigma[:, PAPER_IDX].max(axis=1)   # [N_test]

# Per-patient "any error" flag (at least one disease mispredicted)
errors_paper = (preds_all[:, PAPER_IDX] != test_labels[:, PAPER_IDX]).any(axis=1)
error_rate_overall = errors_paper.mean()
print(f"  Overall error rate (any disease wrong): {error_rate_overall:.4f}")

# Baseline F1-Macro (no deferral)
f1_baseline = f1_score(
    test_labels[:, PAPER_IDX], preds_all[:, PAPER_IDX],
    average='macro', zero_division=0
)
print(f"  Baseline F1-Macro (8 diseases, no deferral): {f1_baseline:.4f}")


# ── Deferral analysis ─────────────────────────────────────────────────────────
sigma_thresholds = [0.05, 0.10, 0.15, 0.20, 0.25]
results = []

print(f"\n{'σ thresh':>10} {'Defer%':>8} {'Defer prec':>12} {'Retained F1':>13} {'N retained':>12}")
print("-" * 60)

for sigma_t in sigma_thresholds:
    deferred  = sigma_max > sigma_t          # [N] bool
    retained  = ~deferred

    n_deferred = deferred.sum()
    n_retained = retained.sum()
    defer_rate = n_deferred / N

    # Precision of deferral: fraction of deferred patients that are actual errors
    if n_deferred > 0:
        defer_precision = errors_paper[deferred].mean()
    else:
        defer_precision = float('nan')

    # F1-Macro on retained patients only
    if n_retained > 10:
        f1_retained = f1_score(
            test_labels[retained][:, PAPER_IDX],
            preds_all[retained][:, PAPER_IDX],
            average='macro', zero_division=0
        )
    else:
        f1_retained = float('nan')

    print(f"{sigma_t:>10.2f} {defer_rate*100:>7.2f}% {defer_precision:>12.4f} "
          f"{f1_retained:>13.4f} {n_retained:>12,}")

    results.append({
        'sigma_threshold':  float(sigma_t),
        'deferral_rate':    float(defer_rate),
        'n_deferred':       int(n_deferred),
        'n_retained':       int(n_retained),
        'deferral_precision': float(defer_precision),
        'f1_macro_retained': float(f1_retained),
    })

print("-" * 60)


# ── Plot ──────────────────────────────────────────────────────────────────────
print("\n[Plot] Deferral curve ...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sigma_vals     = [r['sigma_threshold']    for r in results]
defer_rates    = [r['deferral_rate'] * 100 for r in results]
defer_precs    = [r['deferral_precision'] for r in results]
f1_retained_v  = [r['f1_macro_retained']  for r in results]

# Left: deferral rate vs precision
ax = axes[0]
ax.plot(defer_rates, defer_precs, 'o-', color='tomato', lw=2, ms=8)
for i, st in enumerate(sigma_vals):
    ax.annotate(f'σ>{st}', (defer_rates[i], defer_precs[i]),
                textcoords='offset points', xytext=(5, 5), fontsize=8)
ax.set_xlabel('Deferral Rate (%)', fontsize=11)
ax.set_ylabel('Deferral Precision\n(fraction deferred = actual errors)', fontsize=10)
ax.set_title('Deferral Rate vs Precision', fontsize=11)
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)

# Right: deferral rate vs retained F1
ax = axes[1]
ax.axhline(f1_baseline, color='gray', lw=1.5, ls='--', label=f'No deferral F1={f1_baseline:.4f}')
ax.plot(defer_rates, f1_retained_v, 's-', color='steelblue', lw=2, ms=8)
for i, st in enumerate(sigma_vals):
    ax.annotate(f'σ>{st}', (defer_rates[i], f1_retained_v[i]),
                textcoords='offset points', xytext=(5, -12), fontsize=8)
ax.set_xlabel('Deferral Rate (%)', fontsize=11)
ax.set_ylabel('F1-Macro (retained patients)', fontsize=11)
ax.set_title('Deferral Rate vs Retained F1-Macro', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

fig.suptitle('MC Dropout Deferral Analysis — CareAI HAN++', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'deferral_curve.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved deferral_curve.png")


# ── Save JSON ─────────────────────────────────────────────────────────────────
output = {
    'baseline_f1_macro':    float(f1_baseline),
    'overall_error_rate':   float(error_rate_overall),
    'n_test':               int(N),
    'deferral_results':     results,
}
with open(os.path.join(OUT_DIR, 'deferral_analysis.json'), 'w') as f:
    json.dump(output, f, indent=2)

print(f"  Saved deferral_analysis.json")
print(f"\n[Done]")
print(f"  Baseline F1-Macro      : {f1_baseline:.4f}")
print(f"  Overall error rate     : {error_rate_overall:.4f}")
print(f"\nNext: run Other_py/faithfulness_tests.py")
