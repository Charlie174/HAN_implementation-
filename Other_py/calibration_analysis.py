"""
calibration_analysis.py — Day 2 of 13-day plan
================================================
Calibration metrics for HANPP_Disease v6 (CareAI March 2026).

Inputs (from evaluate_v6.py):
  output/careai_march/eval_outputs/test_probs.npy   [N_test, 9]
  output/careai_march/eval_outputs/test_labels.npy  [N_test, 9]
  output/careai_march/eval_outputs/val_probs.npy    [N_val, 9]
  output/careai_march/eval_outputs/val_labels.npy   [N_val, 9]

Outputs:
  output/careai_march/calibration/calibration_curves.png  — reliability diagrams (8 diseases)
  output/careai_march/calibration/decision_curve.png      — net benefit DCA
  output/careai_march/calibration/calibration_metrics.json

Usage:
    python Other_py/calibration_analysis.py
"""

import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

EVAL_DIR = 'output/careai_march/eval_outputs'
OUT_DIR  = 'output/careai_march/calibration'
os.makedirs(OUT_DIR, exist_ok=True)

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

# ── Load outputs ──────────────────────────────────────────────────────────────
print("[Load] Reading eval_outputs ...")
test_probs  = np.load(os.path.join(EVAL_DIR, 'test_probs.npy'))   # [N_test, 9]
test_labels = np.load(os.path.join(EVAL_DIR, 'test_labels.npy'))  # [N_test, 9]
val_probs   = np.load(os.path.join(EVAL_DIR, 'val_probs.npy'))    # [N_val, 9]
val_labels  = np.load(os.path.join(EVAL_DIR, 'val_labels.npy'))   # [N_val, 9]

N_test = test_probs.shape[0]
N_val  = val_probs.shape[0]
print(f"  Test: {N_test:,}  Val: {N_val:,}")


# ── 1. Brier Scores (before temperature scaling) ─────────────────────────────
print("\n[1] Computing Brier scores (raw) ...")
brier_raw = {}
for i, d in zip(PAPER_IDX, PAPER_DISEASES):
    bs = brier_score_loss(test_labels[:, i], test_probs[:, i])
    brier_raw[d] = float(bs)
    print(f"  {d:25s}  Brier = {bs:.4f}")
brier_raw_mean = float(np.mean(list(brier_raw.values())))
print(f"  {'Mean':25s}  Brier = {brier_raw_mean:.4f}")


# ── 2. Temperature Scaling ────────────────────────────────────────────────────
print("\n[2] Temperature scaling (optimise on val set) ...")

def temperature_scale(probs, T):
    """Apply temperature T to probabilities via logit space."""
    eps = 1e-7
    logits = np.log(np.clip(probs, eps, 1 - eps) / (1 - np.clip(probs, eps, 1 - eps)))
    scaled_logits = logits / T
    return 1.0 / (1.0 + np.exp(-scaled_logits))

def val_brier_loss(T):
    scaled = temperature_scale(val_probs[:, PAPER_IDX], T)
    return np.mean([brier_score_loss(val_labels[:, i], scaled[:, k])
                    for k, i in enumerate(PAPER_IDX)])

result = minimize_scalar(val_brier_loss, bounds=(0.1, 10.0), method='bounded')
T_opt = float(result.x)
print(f"  Optimal T = {T_opt:.4f}  (val Brier before={val_brier_loss(1.0):.4f}, after={result.fun:.4f})")

# Apply temperature scaling to test probs
test_probs_scaled = temperature_scale(test_probs, T_opt)

brier_scaled = {}
for i, d in zip(PAPER_IDX, PAPER_DISEASES):
    bs = brier_score_loss(test_labels[:, i], test_probs_scaled[:, i])
    brier_scaled[d] = float(bs)
    print(f"  {d:25s}  Brier (scaled) = {bs:.4f}")
brier_scaled_mean = float(np.mean(list(brier_scaled.values())))
print(f"  {'Mean':25s}  Brier (scaled) = {brier_scaled_mean:.4f}")


# ── 3. Calibration Curves ─────────────────────────────────────────────────────
print("\n[3] Plotting calibration curves ...")
n_diseases = len(PAPER_DISEASES)
ncols = 4
nrows = (n_diseases + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(16, 8))
axes = axes.flatten()

cal_data = {}
for k, (i, d) in enumerate(zip(PAPER_IDX, PAPER_DISEASES)):
    ax = axes[k]

    # Raw
    frac_pos_raw, mean_pred_raw = calibration_curve(
        test_labels[:, i], test_probs[:, i], n_bins=10, strategy='uniform')
    # Scaled
    frac_pos_sc, mean_pred_sc = calibration_curve(
        test_labels[:, i], test_probs_scaled[:, i], n_bins=10, strategy='uniform')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect')
    ax.plot(mean_pred_raw, frac_pos_raw, 'o-', color='steelblue', ms=4,
            label=f'Raw (B={brier_raw[d]:.3f})')
    ax.plot(mean_pred_sc,  frac_pos_sc,  's--', color='tomato',    ms=4,
            label=f'T-scaled (B={brier_scaled[d]:.3f})')

    ax.set_title(d.replace('_', ' '), fontsize=9, fontweight='bold')
    ax.set_xlabel('Mean Predicted Prob', fontsize=7)
    ax.set_ylabel('Fraction Positives', fontsize=7)
    ax.legend(fontsize=6)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.tick_params(labelsize=7)

    cal_data[d] = {
        'mean_pred_raw': mean_pred_raw.tolist(),
        'frac_pos_raw':  frac_pos_raw.tolist(),
        'mean_pred_scaled': mean_pred_sc.tolist(),
        'frac_pos_scaled':  frac_pos_sc.tolist(),
    }

# Hide unused axes
for ax in axes[n_diseases:]:
    ax.set_visible(False)

fig.suptitle('Calibration Curves — CareAI HAN++ (8 Diseases)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'calibration_curves.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved calibration_curves.png")


# ── 4. Decision Curve Analysis ────────────────────────────────────────────────
print("\n[4] Decision Curve Analysis ...")
thresholds_dca = np.linspace(0.01, 0.99, 100)

def net_benefit(labels, probs, pt):
    """Net benefit at threshold pt."""
    preds = (probs >= pt).astype(int)
    TP = ((preds == 1) & (labels == 1)).sum()
    FP = ((preds == 1) & (labels == 0)).sum()
    n  = len(labels)
    return (TP / n) - (FP / n) * (pt / (1 - pt))

fig, ax = plt.subplots(figsize=(9, 5))

# Aggregate net benefit across 8 diseases (mean)
nb_model = []
nb_treat_all = []
for pt in thresholds_dca:
    nbs = [net_benefit(test_labels[:, i], test_probs[:, i], pt) for i in PAPER_IDX]
    nb_model.append(np.mean(nbs))
    prevalences = [test_labels[:, i].mean() for i in PAPER_IDX]
    nb_all = [prev - (1 - prev) * (pt / (1 - pt)) for prev in prevalences]
    nb_treat_all.append(np.mean(nb_all))

nb_model    = np.array(nb_model)
nb_treat_all = np.array(nb_treat_all)

ax.plot(thresholds_dca, nb_model,    color='steelblue', lw=2, label='HAN++ Model')
ax.plot(thresholds_dca, nb_treat_all, color='tomato',   lw=1.5, ls='--', label='Treat All')
ax.axhline(0, color='gray', lw=1, ls=':',  label='Treat None')
ax.set_xlabel('Threshold Probability', fontsize=11)
ax.set_ylabel('Net Benefit', fontsize=11)
ax.set_title('Decision Curve Analysis — HAN++ (mean over 8 diseases)', fontsize=11)
ax.legend(fontsize=10)
ax.set_xlim(0, 1)
ax.set_ylim(-0.05, max(nb_model.max(), nb_treat_all.max()) + 0.05)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'decision_curve.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved decision_curve.png")


# ── 5. Save metrics JSON ──────────────────────────────────────────────────────
print("\n[5] Saving calibration_metrics.json ...")
metrics = {
    'temperature_optimal': T_opt,
    'brier_raw':    brier_raw,
    'brier_raw_mean': brier_raw_mean,
    'brier_scaled': brier_scaled,
    'brier_scaled_mean': brier_scaled_mean,
    'calibration_curves': cal_data,
    'dca': {
        'thresholds': thresholds_dca.tolist(),
        'nb_model':   nb_model.tolist(),
        'nb_treat_all': nb_treat_all.tolist(),
    },
    'test_size': N_test,
    'val_size':  N_val,
}
with open(os.path.join(OUT_DIR, 'calibration_metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"\n[Done] Outputs saved to {OUT_DIR}/")
print(f"  calibration_curves.png, decision_curve.png, calibration_metrics.json")
print(f"\n  Summary:")
print(f"    Optimal temperature T     = {T_opt:.4f}")
print(f"    Mean Brier (raw)          = {brier_raw_mean:.4f}")
print(f"    Mean Brier (T-scaled)     = {brier_scaled_mean:.4f}")
print(f"\nNext: run Other_py/deferral_analysis.py")
