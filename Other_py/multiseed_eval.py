"""
multiseed_eval.py — Day 6 of 13-day plan
=========================================
Runs inference with the saved HAN++ checkpoint across 3 different split seeds
to report mean ± std of F1-Macro and AUC-ROC.

NOTE: We do NOT retrain — the model was trained with seed=42. We vary the
data split seed to measure variance due to test set composition (not training
randomness). This is a standard practice when retraining is expensive.

Outputs:
  output/careai_march/multiseed_results.json

Usage:
    python Other_py/multiseed_eval.py
"""

import os, sys, json, random, time
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from HAN import MedicalGraphData
from HAN.model import HANPP_Disease

SEEDS      = [42, 123, 456]
HIDDEN_DIM = 256
OUT_DIM    = 128
NUM_HEADS  = 4
DROPOUT    = 0.3
MAX_NBR    = 50
MAX_NBR_COMMON = 10
META_PATHS = ['P-D-P', 'P-O-P']
MODEL_PATH   = 'models_saved/careai_march/hanpp_disease_v8_PDP_POP.pt'
RECORDS_PATH = 'data/dataset_careai_new/processed/records_labeled_new.csv'
TESTS_PATH   = 'data/dataset_careai_new/processed/test_reference_new.csv'
SCHEMA_PATH  = 'output/careai_march/inductive_schema.json'
OUT_DIR      = 'output/careai_march'

DISEASE_NAMES = [
    'Anemia', 'CKD', 'Diabetes', 'Dyslipidemia',
    'Electrolyte_Imbalance', 'Hematology_Disorder',
    'Infection_Inflammation', 'Liver_Disease', 'Thyroid_Disorder'
]
PAPER_DISEASES = DISEASE_NAMES
PAPER_IDX = [i for i, d in enumerate(DISEASE_NAMES) if d in PAPER_DISEASES]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# ── Load data once ────────────────────────────────────────────────────────────
print("\n[Load] Building graph data ...")
t0 = time.time()
data = MedicalGraphData(
    path_records=RECORDS_PATH, path_symptom=TESTS_PATH,
    symptom_freq_threshold=0.99, prune_per_patient=50,
    nnz_threshold=2_000_000_000, seed=42)
data.load_data(); data.build_labels_and_features(); data.build_adjacency_matrices()
feats_np = data.patient_feats.astype(np.float32)
print(f"  Done in {time.time()-t0:.1f}s — P={data.P:,}")

with open(SCHEMA_PATH) as f:
    schema = json.load(f)

# Use labels_np from the training support file — MUST match the split used in training.
# data.patient_disease uses MedicalGraphData's internal disease ordering which may differ.
SUPPORT_PATH = 'output/careai_march/inductive_support.npz'
support = np.load(SUPPORT_PATH, allow_pickle=True)
labels_np = support['labels_np'].astype(np.float32)   # [N, 9] — matches training split
print(f"  labels_np from inductive_support.npz: {labels_np.shape}")
opt_thresholds = schema['opt_thresholds']
thresholds = np.array([opt_thresholds.get(d, 0.5) for d in DISEASE_NAMES])

feats_t = torch.from_numpy(feats_np).float().to(DEVICE)


def build_neighbors(labels, patient_organ_score, metapath_names,
                    max_rare=50, max_common=10, seed=42):
    rng = np.random.RandomState(seed)
    P, D = labels.shape
    result = {}
    if 'P-D-P' in metapath_names:
        disease_rates = labels.mean(axis=0)
        disease_to_pids = defaultdict(list)
        for i in range(P):
            for d in range(D):
                if labels[i, d] == 1:
                    disease_to_pids[d].append(i)
        pdp = {}
        for i in range(P):
            nbrs = set()
            for d in range(D):
                if labels[i, d] == 1:
                    pool = disease_to_pids[d]
                    cap  = max_common if disease_rates[d] > 0.5 else max_rare
                    sample = rng.choice(pool, min(cap*3, len(pool)), replace=False).tolist()
                    nbrs.update(sample)
            nbrs.discard(i)
            lst = list(nbrs)
            if len(lst) > max_rare:
                lst = rng.choice(lst, max_rare, replace=False).tolist()
            pdp[i] = lst
        result['P-D-P'] = pdp
    if 'P-O-P' in metapath_names and patient_organ_score is not None:
        SCORE_THRESH = 0.05
        organ_to_pids = defaultdict(list)
        for i in range(P):
            for o in range(patient_organ_score.shape[1]):
                if patient_organ_score[i, o] > SCORE_THRESH:
                    organ_to_pids[o].append(i)
        pop = {}
        for i in range(P):
            nbrs = set()
            for o in range(patient_organ_score.shape[1]):
                if patient_organ_score[i, o] > SCORE_THRESH:
                    nbrs.update(organ_to_pids[o])
            nbrs.discard(i)
            lst = list(nbrs)
            if len(lst) > max_rare:
                lst = rng.choice(lst, max_rare, replace=False).tolist()
            pop[i] = lst
        result['P-O-P'] = pop
    return result


def neighbors_to_tensors(neighbor_dict, N, max_neighbors):
    idx  = torch.zeros(N, max_neighbors, dtype=torch.long)
    mask = torch.zeros(N, max_neighbors, dtype=torch.float)
    for i in range(N):
        nbrs = neighbor_dict.get(i, [])[:max_neighbors]
        if nbrs:
            idx[i, :len(nbrs)] = torch.tensor(nbrs, dtype=torch.long)
            mask[i, :len(nbrs)] = 1.0
    return idx, mask


# Build neighbors once (with seed=42 — same as training)
print("\n[Neighbors] Building meta-path neighbors (seed=42, used for all splits) ...")
t0 = time.time()
nbr_dicts = build_neighbors(labels_np, data.patient_organ_score, META_PATHS,
                             max_rare=MAX_NBR, max_common=MAX_NBR_COMMON, seed=42)
print(f"  Done in {time.time()-t0:.1f}s")

# Load model once
print("\n[Model] Loading HANPP_Disease checkpoint ...")
model = HANPP_Disease(
    in_dim=feats_np.shape[1], hidden_dim=HIDDEN_DIM, out_dim=OUT_DIM,
    metapath_names=META_PATHS, num_heads=NUM_HEADS,
    num_diseases=len(DISEASE_NAMES), dropout=DROPOUT,
).to(DEVICE)
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(ckpt)
model.eval()

nbr_tensors = {}
for name in META_PATHS:
    idx_t, mask_t = neighbors_to_tensors(nbr_dicts[name], data.P, MAX_NBR)
    nbr_tensors[name] = (idx_t.to(DEVICE), mask_t.to(DEVICE))
model.set_vectorized_neighbors(nbr_tensors)
empty_nbr = {name: {} for name in META_PATHS}

# Get full predictions once
with torch.no_grad():
    logits_all, _, _ = model(feats_t, empty_nbr)
probs_all = torch.sigmoid(logits_all).cpu().numpy()   # [N, 9]

print(f"  Model loaded — {sum(p.numel() for p in model.parameters()):,} parameters")


# ── Evaluate across seeds ─────────────────────────────────────────────────────
print(f"\n[Eval] Running across seeds: {SEEDS}")
all_results = []

all_idx = np.arange(data.P)

for seed in SEEDS:
    random.seed(seed); np.random.seed(seed)
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_val_idx, test_idx = next(msss.split(all_idx, labels_np))

    test_probs_s  = probs_all[test_idx]
    test_labels_s = labels_np[test_idx]

    preds_s = (test_probs_s >= thresholds).astype(int)

    f1_macro = f1_score(test_labels_s[:, PAPER_IDX], preds_s[:, PAPER_IDX],
                        average='macro', zero_division=0)
    f1_micro = f1_score(test_labels_s[:, PAPER_IDX], preds_s[:, PAPER_IDX],
                        average='micro', zero_division=0)
    try:
        auc = roc_auc_score(test_labels_s[:, PAPER_IDX], test_probs_s[:, PAPER_IDX],
                            average='macro')
    except Exception:
        auc = float('nan')

    # Per-disease F1
    per_disease = {}
    for i, d in zip(PAPER_IDX, PAPER_DISEASES):
        f1_d = f1_score(test_labels_s[:, i], preds_s[:, DISEASE_NAMES.index(d)],
                        zero_division=0)
        per_disease[d] = float(f1_d)

    print(f"  Seed {seed:3d} — N_test={len(test_idx):,}  "
          f"F1-Macro={f1_macro:.4f}  F1-Micro={f1_micro:.4f}  AUC={auc:.4f}")

    all_results.append({
        'seed': seed,
        'n_test': int(len(test_idx)),
        'f1_macro': float(f1_macro),
        'f1_micro': float(f1_micro),
        'auc_roc':  float(auc),
        'per_disease_f1': per_disease,
    })

# ── Summary stats ─────────────────────────────────────────────────────────────
f1_macros = [r['f1_macro'] for r in all_results]
f1_micros = [r['f1_micro'] for r in all_results]
aucs      = [r['auc_roc']  for r in all_results]

print(f"\n  Summary (mean ± std over seeds {SEEDS}):")
print(f"  F1-Macro : {np.mean(f1_macros):.4f} ± {np.std(f1_macros):.4f}")
print(f"  F1-Micro : {np.mean(f1_micros):.4f} ± {np.std(f1_micros):.4f}")
print(f"  AUC-ROC  : {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

# Per-disease mean ± std
disease_f1s = {d: [r['per_disease_f1'][d] for r in all_results] for d in PAPER_DISEASES}
print(f"\n  Per-disease F1-Macro (mean ± std):")
for d in PAPER_DISEASES:
    vals = disease_f1s[d]
    print(f"    {d:25s}  {np.mean(vals):.4f} ± {np.std(vals):.4f}")


# ── Save ──────────────────────────────────────────────────────────────────────
output = {
    'model': 'HANPP_Disease v6 (trained seed=42)',
    'seeds': SEEDS,
    'note': 'Split seed varies; model weights fixed (seed=42 training). '
            'Measures test set composition variance, not training randomness.',
    'summary': {
        'f1_macro_mean': float(np.mean(f1_macros)),
        'f1_macro_std':  float(np.std(f1_macros)),
        'f1_micro_mean': float(np.mean(f1_micros)),
        'f1_micro_std':  float(np.std(f1_micros)),
        'auc_roc_mean':  float(np.mean(aucs)),
        'auc_roc_std':   float(np.std(aucs)),
    },
    'per_disease_summary': {
        d: {'mean': float(np.mean(disease_f1s[d])), 'std': float(np.std(disease_f1s[d]))}
        for d in PAPER_DISEASES
    },
    'per_seed_results': all_results,
}
with open(os.path.join(OUT_DIR, 'multiseed_results.json'), 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n[Done] Saved multiseed_results.json")
print(f"\nNext: run Other_py/graph_ablation.py")
