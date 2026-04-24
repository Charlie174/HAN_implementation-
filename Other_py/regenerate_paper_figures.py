"""
Regenerate all paper figures for IEEE AIDM 2026 submission.
Saves directly to paper/figures/.

Run from project root:
    python Other_py/regenerate_paper_figures.py
"""

import os
import shutil
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT        = os.path.join(BASE, "output", "careai_march")
FIG_DIR    = os.path.join(BASE, "paper", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Colour palette (IEEE-friendly, colour-blind safe) ─────────────────────────
C_TRAD  = "#9ecae1"   # light blue  – traditional ML
C_BAS   = "#fdae6b"   # orange      – GNN/DL baselines
C_HGT   = "#a1d99b"   # green       – HGT-HAN hybrid
C_HAN   = "#e6550d"   # dark orange – HAN++ (ours)
C_BEST  = "#d62728"   # red         – best HAN++

plt.rcParams.update({
    "font.family":  "serif",
    "font.size":    9,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "figure.dpi":   150,
})


# ══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Model Comparison Bar Chart (updated with RETAIN)
# ══════════════════════════════════════════════════════════════════════════════
def make_fig4():
    models = [
        # (label,         f1_macro, colour,  group)
        ("Naïve Bayes",   0.3223,   C_TRAD,  "Traditional\n(GridSearch)"),
        ("KNN",           0.8315,   C_TRAD,  "Traditional\n(GridSearch)"),
        ("SVM (Linear)",  0.9322,   C_TRAD,  "Traditional\n(GridSearch)"),
        ("XGBoost",       0.9355,   C_TRAD,  "Traditional\n(GridSearch)"),
        ("Logistic Reg.", 0.9430,   C_TRAD,  "Traditional\n(GridSearch)"),
        ("Random Forest", 0.9458,   C_TRAD,  "Traditional\n(GridSearch)"),
        ("Decision Tree", 0.9458,   C_TRAD,  "Traditional\n(GridSearch)"),
        ("SeHGNN",        0.6300,   C_BAS,   "GNN / DL\nBaselines"),
        ("pyHGT",         0.7550,   C_BAS,   "GNN / DL\nBaselines"),
        ("RETAIN",        0.8111,   C_BAS,   "GNN / DL\nBaselines"),
        ("HGT-HAN\n(P-D-P)", 0.8401, C_HGT, "HGT-HAN\nHybrid"),
        ("HAN++\n(P-D-P)", 0.8319,  C_HAN,  "HAN++\n(Ours)"),
        ("HAN++\n(P-O-P)", 0.4176,  C_HAN,  "HAN++\n(Ours)"),
        ("HAN++\n(P-D-P\n+P-O-P)", 0.9378, C_BEST, "HAN++\n(Ours)"),
    ]

    labels = [m[0] for m in models]
    values = [m[1] for m in models]
    colours = [m[2] for m in models]

    fig, ax = plt.subplots(figsize=(7.0, 3.4))

    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colours, width=0.65, edgecolor="white", linewidth=0.4)

    # Highlight best bar
    bars[-1].set_edgecolor("#333")
    bars[-1].set_linewidth(1.2)

    # Value labels on top of bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"{val:.4f}", ha="center", va="bottom", fontsize=6.5,
                rotation=90, color="#333")

    # Dashed separator lines between groups
    separators = [6.5, 9.5, 10.5]   # after DT, after RETAIN, after HGT-HAN
    for sep in separators:
        ax.axvline(x=sep, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.8)
    ax.set_ylabel("F1-Macro")
    ax.set_ylim(0, 1.05)
    ax.set_title("F1-Macro Comparison Across All Model Families (n = 48,546)")
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    # Legend
    legend_items = [
        mpatches.Patch(color=C_TRAD, label="Traditional ML (GridSearch-tuned)"),
        mpatches.Patch(color=C_BAS,  label="GNN / DL Baselines"),
        mpatches.Patch(color=C_HGT,  label="HGT-HAN Hybrid"),
        mpatches.Patch(color=C_HAN,  label="HAN++ (single meta-path)"),
        mpatches.Patch(color=C_BEST, label="HAN++ Best (P-D-P + P-O-P) ← Ours"),
    ]
    ax.legend(handles=legend_items, loc="upper left", fontsize=6.8,
              framealpha=0.85, edgecolor="gray")

    fig.tight_layout()
    out_path = os.path.join(FIG_DIR, "fig4_model_comparison.pdf")
    fig.savefig(out_path, bbox_inches="tight", format="pdf")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 6 — Training Convergence (copy from v6 output)
# ══════════════════════════════════════════════════════════════════════════════
def make_fig6():
    src = os.path.join(OUT, "training_curves_PDP_POP.png")
    dst = os.path.join(FIG_DIR, "fig6_training_convergence.png")
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"  Copied: {dst}")
    else:
        print(f"  WARNING: source not found: {src}")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 7 — Calibration Curves + Decision Curve (side-by-side)
# ══════════════════════════════════════════════════════════════════════════════
def make_fig7():
    """Composite: calibration reliability diagram + decision curve."""
    import json

    cal_metrics_path = os.path.join(OUT, "calibration", "calibration_metrics.json")
    if not os.path.exists(cal_metrics_path):
        print(f"  WARNING: {cal_metrics_path} not found, skipping fig7")
        return

    with open(cal_metrics_path) as f:
        cal = json.load(f)

    diseases     = list(cal["brier_raw"].keys())
    brier_raw    = [cal["brier_raw"][d]    for d in diseases]
    brier_scaled = [cal["brier_scaled"][d] for d in diseases]

    short = {
        "Anemia": "Anemia",
        "CKD": "CKD",
        "Diabetes": "Diabetes",
        "Dyslipidemia": "Dyslip.",
        "Electrolyte_Imbalance": "Elec.Imb.",
        "Hematology_Disorder": "Hematol.",
        "Infection_Inflammation": "Infect.",
        "Liver_Disease": "Liver",
        "Thyroid_Disorder": "Thyroid",
    }
    labels = [short.get(d, d) for d in diseases]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))

    # -- Left: Brier score comparison --
    ax = axes[0]
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w/2, brier_raw,    width=w, color="#fdae6b", label="Raw",          edgecolor="white")
    ax.bar(x + w/2, brier_scaled, width=w, color="#4292c6", label="T-scaled (T*=0.276)", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Brier Score (lower = better)")
    ax.set_title("Per-Disease Calibration (Brier Score)")
    ax.set_ylim(0, 0.08)
    ax.legend(fontsize=7)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    # Mean lines
    ax.axhline(np.mean(brier_raw),    color="#fdae6b", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.axhline(np.mean(brier_scaled), color="#4292c6", linestyle="--", linewidth=1.0, alpha=0.8)

    # -- Right: copy decision curve image --
    dc_path = os.path.join(OUT, "calibration", "decision_curve.png")
    if os.path.exists(dc_path):
        img = plt.imread(dc_path)
        axes[1].imshow(img)
        axes[1].axis("off")
        axes[1].set_title("Decision Curve Analysis")
    else:
        axes[1].text(0.5, 0.5, "decision_curve.png\nnot found",
                     ha="center", va="center", transform=axes[1].transAxes)
        axes[1].axis("off")

    fig.tight_layout()
    out_path = os.path.join(FIG_DIR, "fig7_calibration.pdf")
    fig.savefig(out_path, bbox_inches="tight", format="pdf")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 8 — MC Dropout Deferral Curve (copy from output)
# ══════════════════════════════════════════════════════════════════════════════
def make_fig8():
    src = os.path.join(OUT, "deferral_curve.png")
    dst = os.path.join(FIG_DIR, "fig8_deferral_curve.png")
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"  Copied: {dst}")
    else:
        print(f"  WARNING: source not found: {src}")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 9 — Per-Disease F1 with Uncertainty Bars (from multiseed results)
# ══════════════════════════════════════════════════════════════════════════════
def make_fig9():
    ms_path = os.path.join(OUT, "multiseed_results.json")
    if not os.path.exists(ms_path):
        print(f"  WARNING: {ms_path} not found, skipping fig9")
        return

    with open(ms_path) as f:
        ms = json.load(f)

    per_d = ms["per_disease_summary"]
    diseases = list(per_d.keys())
    means = [per_d[d]["mean"] for d in diseases]
    stds  = [per_d[d]["std"]  for d in diseases]

    short = {
        "Anemia": "Anemia", "CKD": "CKD", "Diabetes": "Diabetes",
        "Dyslipidemia": "Dyslip.", "Electrolyte_Imbalance": "Elec.Imb.",
        "Hematology_Disorder": "Hematol.", "Liver_Disease": "Liver",
        "Thyroid_Disorder": "Thyroid",
    }
    labels = [short.get(d, d) for d in diseases]

    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, color="#4292c6", edgecolor="white",
                  capsize=4, error_kw={"elinewidth": 1.2, "ecolor": "#d62728"})

    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + s + 0.003,
                f"{m:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=7.5)
    ax.set_ylabel("F1-Macro (mean ± std, 3 seeds)")
    ax.set_ylim(0.85, 1.01)
    ax.set_title("Per-Disease F1-Macro Stability (3 Random Seeds)")
    ax.axhline(np.mean(means), color="#e6550d", linestyle="--",
               linewidth=1.0, label=f"Overall mean = {np.mean(means):.4f}")
    ax.legend(fontsize=7.5)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    fig.tight_layout()
    out_path = os.path.join(FIG_DIR, "fig9_per_disease_f1.pdf")
    fig.savefig(out_path, bbox_inches="tight", format="pdf")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Regenerating paper figures...")
    print("\n[fig4] Model comparison bar chart (with RETAIN)")
    make_fig4()
    print("\n[fig6] Training convergence (v6 curves)")
    make_fig6()
    print("\n[fig7] Calibration + Decision Curve")
    make_fig7()
    print("\n[fig8] MC Dropout deferral curve")
    make_fig8()
    print("\n[fig9] Per-disease F1 with stability bars")
    make_fig9()
    print("\nDone. All figures saved to paper/figures/")
