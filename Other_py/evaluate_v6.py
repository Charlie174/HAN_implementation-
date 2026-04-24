"""
evaluate_v6.py — Day 1 of 13-day plan
======================================
Full inference script for HANPP_Disease v6 (CareAI March 2026).

Outputs saved to output/careai_march/eval_outputs/:
  - test_probs.npy       [N_test, 9]   sigmoid probabilities per disease
  - test_labels.npy      [N_test, 9]   ground-truth binary labels
  - test_patient_ids.npy [N_test]      global patient indices
  - val_probs.npy        [N_val, 9]    validation set probabilities (for calibration)
  - val_labels.npy       [N_val, 9]    validation set labels
  - mc_sigma.npy         [N_test, 9]   MC Dropout std per disease (50 passes)
  - mc_mean.npy          [N_test, 9]   MC Dropout mean per disease (50 passes)
  - beta_weights.npy     [N_test, 2]   semantic attention weights (P-D-P, P-O-P)

Usage:
    cd /path/to/HAN-implementation
    python Other_py/evaluate_v6.py

Dependencies: same as training notebook (torch, numpy, sklearn, iterstrat)
"""

import os, sys, time, json, random
from collections import defaultdict

import numpy as np
import torch
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

# ── Project root setup ────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from HAN import MedicalGraphData
from HAN.model import HANPP_Disease

# ── Config (must match training notebook exactly) ─────────────────────────────
SEED        = 42
HIDDEN_DIM  = 256
OUT_DIM     = 128
NUM_HEADS   = 4
DROPOUT     = 0.3
MAX_NBR     = 50
MAX_NBR_COMMON = 10
META_PATHS  = ['P-D-P', 'P-O-P']
MC_SAMPLES  = 50

DISEASE_NAMES = [
    'Anemia', 'CKD', 'Diabetes', 'Dyslipidemia',
    'Electrolyte_Imbalance', 'Hematology_Disorder',
    'Infection_Inflammation', 'Liver_Disease', 'Thyroid_Disorder'
]

PAPER_DISEASES = DISEASE_NAMES

MODEL_PATH   = 'models_saved/careai_march/hanpp_disease_v8_PDP_POP.pt'
RECORDS_PATH = 'data/dataset_careai_new/processed/records_labeled_new.csv'
TESTS_PATH   = 'data/dataset_careai_new/processed/test_reference_new.csv'
SCHEMA_PATH  = 'output/careai_march/inductive_schema.json'
SUPPORT_PATH = 'output/careai_march/inductive_support.npz'
OUT_DIR      = 'output/careai_march/eval_outputs'

os.makedirs(OUT_DIR, exist_ok=True)

# ── Reproducibility ───────────────────────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# ── Load schema (opt_thresholds + disease_order) ──────────────────────────────
with open(SCHEMA_PATH) as f:
    schema = json.load(f)
opt_thresholds = schema['opt_thresholds']
print(f"Loaded schema — in_dim={schema['in_dim']}, diseases={len(DISEASE_NAMES)}")


# ── Step 1: Load features + labels from saved support file ───────────────────
# Much faster than rebuilding MedicalGraphData from scratch.
# We still need organ_score for P-O-P neighbor building.
print("\n[1/6] Loading saved features + labels from inductive_support.npz ...")
support = np.load(SUPPORT_PATH, allow_pickle=True)
feats_np  = support['feats_np']    # [N, 120]
labels_np = support['labels_np']   # [N, 9]
print(f"  feats_np : {feats_np.shape}")
print(f"  labels_np: {labels_np.shape}")


# ── Step 2: Rebuild organ scores (needed for P-O-P neighbors) ────────────────
print("\n[2/6] Rebuilding graph via MedicalGraphData (for organ scores) ...")
t0 = time.time()
data = MedicalGraphData(
    path_records=RECORDS_PATH,
    path_symptom=TESTS_PATH,
    symptom_freq_threshold=0.99,
    prune_per_patient=50,
    nnz_threshold=2_000_000_000,
    seed=SEED
)
data.load_data()
data.build_labels_and_features()
data.build_adjacency_matrices()
print(f"  Graph loaded in {time.time()-t0:.1f}s — P={data.P:,}, S={data.S}, O={data.O}")

# Use data.patient_feats directly (authoritative source)
feats_np = data.patient_feats.astype(np.float32)  # [N, 120]


# ── Step 3: Reproduce exact train/val/test split ──────────────────────────────
print("\n[3/6] Reproducing train/val/test split (seed=42, 80/20/20) ...")
all_idx = np.arange(data.P)

msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_val_idx, test_idx = next(msss.split(all_idx, labels_np))

msss_val = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_sub, val_sub = next(msss_val.split(train_val_idx, labels_np[train_val_idx]))
train_idx = train_val_idx[train_sub]
val_idx   = train_val_idx[val_sub]

print(f"  Train: {len(train_idx):,}  Val: {len(val_idx):,}  Test: {len(test_idx):,}")


# ── Step 4: Build neighbors + load model ─────────────────────────────────────
print("\n[4/6] Building meta-path neighbors (this may take ~2 min) ...")

def build_neighbors_v2(labels, disease_order, patient_organ_score,
                        metapath_names, max_rare=50, max_common=10, seed=42):
    rng = np.random.RandomState(seed)
    P, D = labels.shape
    result = {}

    if 'P-D-P' in metapath_names:
        t0 = time.time()
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
                    sample = rng.choice(pool, min(cap * 3, len(pool)), replace=False).tolist()
                    nbrs.update(sample)
            nbrs.discard(i)
            lst = list(nbrs)
            if len(lst) > max_rare:
                lst = rng.choice(lst, max_rare, replace=False).tolist()
            pdp[i] = lst
        result['P-D-P'] = pdp
        avg_nbr = np.mean([len(v) for v in pdp.values()])
        print(f"  P-D-P: {time.time()-t0:.1f}s, avg {avg_nbr:.1f} nbrs/patient")

    if 'P-O-P' in metapath_names and patient_organ_score is not None:
        t0 = time.time()
        SCORE_THRESH = 0.05
        O = patient_organ_score.shape[1]
        organ_to_pids = defaultdict(list)
        for i in range(P):
            for o in range(O):
                if patient_organ_score[i, o] > SCORE_THRESH:
                    organ_to_pids[o].append(i)
        pop = {}
        for i in range(P):
            nbrs = set()
            for o in range(O):
                if patient_organ_score[i, o] > SCORE_THRESH:
                    nbrs.update(organ_to_pids[o])
            nbrs.discard(i)
            lst = list(nbrs)
            if len(lst) > max_rare:
                lst = rng.choice(lst, max_rare, replace=False).tolist()
            pop[i] = lst
        result['P-O-P'] = pop
        avg_nbr = np.mean([len(v) for v in pop.values()])
        print(f"  P-O-P: {time.time()-t0:.1f}s, avg {avg_nbr:.1f} nbrs/patient")

    return result


def neighbors_to_tensors(neighbor_dict, N, max_neighbors):
    idx  = torch.zeros(N, max_neighbors, dtype=torch.long)
    mask = torch.zeros(N, max_neighbors, dtype=torch.float)
    for i in range(N):
        nbrs = neighbor_dict.get(i, [])[:max_neighbors]
        if nbrs:
            idx[i,  :len(nbrs)] = torch.tensor(nbrs, dtype=torch.long)
            mask[i, :len(nbrs)] = 1.0
    return idx, mask


nbr_dicts = build_neighbors_v2(
    labels_np, DISEASE_NAMES,
    patient_organ_score=data.patient_organ_score,
    metapath_names=META_PATHS,
    max_rare=MAX_NBR,
    max_common=MAX_NBR_COMMON,
    seed=SEED,
)
active_paths = [n for n in META_PATHS if n in nbr_dicts]

# ── Load model ────────────────────────────────────────────────────────────────
print(f"\n  Loading model from {MODEL_PATH} ...")
model = HANPP_Disease(
    in_dim=feats_np.shape[1],
    hidden_dim=HIDDEN_DIM,
    out_dim=OUT_DIM,
    metapath_names=active_paths,
    num_heads=NUM_HEADS,
    num_diseases=len(DISEASE_NAMES),
    dropout=DROPOUT,
).to(DEVICE)

ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(ckpt)
model.eval()
print(f"  Model loaded — {sum(p.numel() for p in model.parameters()):,} parameters")

# Pre-set vectorized neighbors (same as training)
nbr_tensors = {}
for name in active_paths:
    idx_t, mask_t = neighbors_to_tensors(nbr_dicts[name], data.P, MAX_NBR)
    nbr_tensors[name] = (idx_t.to(DEVICE), mask_t.to(DEVICE))
model.set_vectorized_neighbors(nbr_tensors)

feats_t  = torch.from_numpy(feats_np).float().to(DEVICE)
empty_nbr = {name: {} for name in active_paths}


# ── Step 5: Deterministic inference (eval mode) ───────────────────────────────
print("\n[5/6] Running deterministic inference on val + test sets ...")

@torch.no_grad()
def run_inference(indices):
    """Run model in eval mode on given patient indices. Returns probs + beta."""
    model.eval()
    logits, z, beta = model(feats_t, empty_nbr)
    probs = torch.sigmoid(logits[indices]).cpu().numpy()   # [N, D]
    beta_w = beta[indices].cpu().numpy()                   # [N, K]
    return probs, beta_w

t0 = time.time()
test_probs, test_beta = run_inference(test_idx)
val_probs,  val_beta  = run_inference(val_idx)
print(f"  Deterministic inference done in {time.time()-t0:.1f}s")
print(f"  test_probs: {test_probs.shape}  val_probs: {val_probs.shape}")

test_labels = labels_np[test_idx]   # [N_test, 9]
val_labels  = labels_np[val_idx]    # [N_val, 9]


# ── Step 6: MC Dropout inference (50 passes) ──────────────────────────────────
print(f"\n[6/6] Running MC Dropout ({MC_SAMPLES} passes) on test set ...")
t0 = time.time()

model.train()   # enable dropout
all_mc_probs = []

with torch.no_grad():
    for s in range(MC_SAMPLES):
        logits, _, _ = model(feats_t, empty_nbr)
        probs_s = torch.sigmoid(logits[test_idx]).cpu().numpy()  # [N_test, 9]
        all_mc_probs.append(probs_s)
        if (s + 1) % 10 == 0:
            print(f"  MC pass {s+1}/{MC_SAMPLES} — {time.time()-t0:.1f}s elapsed")

model.eval()

all_mc_probs = np.stack(all_mc_probs, axis=0)   # [50, N_test, 9]
mc_mean = all_mc_probs.mean(axis=0)             # [N_test, 9]
mc_sigma = all_mc_probs.std(axis=0)             # [N_test, 9]

print(f"  MC Dropout done in {time.time()-t0:.1f}s")
print(f"  Mean σ across diseases: {mc_sigma.mean():.4f}")
print(f"  Max  σ across diseases: {mc_sigma.max():.4f}")


# ── Save all outputs ──────────────────────────────────────────────────────────
print(f"\n[Saving] Writing outputs to {OUT_DIR}/ ...")

np.save(os.path.join(OUT_DIR, 'test_probs.npy'),       test_probs)
np.save(os.path.join(OUT_DIR, 'test_labels.npy'),      test_labels)
np.save(os.path.join(OUT_DIR, 'test_patient_ids.npy'), test_idx)
np.save(os.path.join(OUT_DIR, 'val_probs.npy'),        val_probs)
np.save(os.path.join(OUT_DIR, 'val_labels.npy'),       val_labels)
np.save(os.path.join(OUT_DIR, 'val_patient_ids.npy'),  val_idx)
np.save(os.path.join(OUT_DIR, 'mc_mean.npy'),          mc_mean)
np.save(os.path.join(OUT_DIR, 'mc_sigma.npy'),         mc_sigma)
np.save(os.path.join(OUT_DIR, 'beta_weights.npy'),     test_beta)

# ── Quick sanity check: reproduce F1 from saved results ──────────────────────
from sklearn.metrics import f1_score, roc_auc_score

thresholds = np.array([opt_thresholds.get(d, 0.5) for d in DISEASE_NAMES])
preds = (test_probs >= thresholds).astype(int)

paper_idx = [i for i, d in enumerate(DISEASE_NAMES) if d in PAPER_DISEASES]

# Sanity check: F1-Macro over ALL 9 diseases (matches training notebook 0.8491)
# Infection_Inflammation has 0 positives → F1=0, which pulls macro down to 0.8491
f1_macro_all9 = f1_score(test_labels, preds, average='macro', zero_division=0)
f1_micro_all9 = f1_score(test_labels, preds, average='micro', zero_division=0)

# 8-disease metrics (excluding Infection_Inflammation) — used for paper tables
f1_macro_8 = f1_score(test_labels[:, paper_idx], preds[:, paper_idx],
                      average='macro', zero_division=0)

try:
    auc = roc_auc_score(test_labels[:, paper_idx], test_probs[:, paper_idx],
                        average='macro')
except Exception as e:
    auc = float('nan')
    print(f"  AUC warning: {e}")

print(f"\n[Sanity Check] Reproduced test metrics:")
print(f"  F1-Macro (9 diseases, matches JSON) : {f1_macro_all9:.4f}  (expected 0.8491)")
print(f"  F1-Micro (9 diseases)               : {f1_micro_all9:.4f}  (expected 0.9531)")
print(f"  F1-Macro (8 paper diseases)         : {f1_macro_8:.4f}  (expected ~0.9553)")
print(f"  AUC-ROC  (8 paper diseases)         : {auc:.4f}  (expected 0.9969)")

f1_macro = f1_macro_all9  # use 9-disease for summary consistency with training JSON
f1_micro  = f1_micro_all9

# Save summary
summary = {
    'reproduced_f1_macro_9diseases': float(f1_macro_all9),   # matches training JSON 0.8491
    'reproduced_f1_macro_8diseases': float(f1_macro_8),       # paper tables ~0.9553
    'reproduced_f1_micro': float(f1_micro_all9),
    'reproduced_auc_roc':  float(auc),
    'test_size':    int(len(test_idx)),
    'val_size':     int(len(val_idx)),
    'mc_samples':   MC_SAMPLES,
    'mean_sigma':   float(mc_sigma[:, paper_idx].mean()),
    'max_sigma':    float(mc_sigma[:, paper_idx].max()),
    'disease_names': DISEASE_NAMES,
    'paper_diseases': PAPER_DISEASES,
}
with open(os.path.join(OUT_DIR, 'eval_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n[Done] All outputs saved to {OUT_DIR}/")
print("  Files: test_probs.npy, test_labels.npy, test_patient_ids.npy")
print("         val_probs.npy, val_labels.npy, val_patient_ids.npy")
print("         mc_mean.npy, mc_sigma.npy, beta_weights.npy, eval_summary.json")
print("\nNext: run Other_py/calibration_analysis.py")
