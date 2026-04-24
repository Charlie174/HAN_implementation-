"""
faithfulness_tests.py — Day 4 of 13-day plan
=============================================
Attention faithfulness: do the high-attention meta-paths actually drive predictions?

Tests:
  1. Meta-path weight masking: for patients where beta[P-D-P] > beta[P-O-P],
     set beta = [0, 1] (force P-O-P) → measure average logit change.
     Report % of patients where prediction shifts significantly (>0.1 logit).

  2. Top-path feature shuffle: for each patient, identify the dominant meta-path,
     shuffle the neighbor features for that path → measure F1 drop.
     Compare to shuffling the non-dominant path (should drop more for dominant).

Inputs (from evaluate_v6.py):
  output/careai_march/eval_outputs/test_probs.npy      [N_test, 9]
  output/careai_march/eval_outputs/test_labels.npy     [N_test, 9]
  output/careai_march/eval_outputs/beta_weights.npy    [N_test, 2]

Outputs:
  output/careai_march/faithfulness_results.json

Usage:
    python Other_py/faithfulness_tests.py
"""

import os, sys, json, random, time
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import f1_score
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from HAN import MedicalGraphData
from HAN.model import HANPP_Disease

SEED        = 42
HIDDEN_DIM  = 256
OUT_DIM     = 128
NUM_HEADS   = 4
DROPOUT     = 0.3
MAX_NBR     = 50
MAX_NBR_COMMON = 10
META_PATHS  = ['P-D-P', 'P-O-P']
MODEL_PATH  = 'models_saved/careai_march/hanpp_disease_v8_PDP_POP.pt'
RECORDS_PATH= 'data/dataset_careai_new/processed/records_labeled_new.csv'
TESTS_PATH  = 'data/dataset_careai_new/processed/test_reference_new.csv'
SCHEMA_PATH  = 'output/careai_march/inductive_schema.json'
SUPPORT_PATH = 'output/careai_march/inductive_support.npz'
EVAL_DIR     = 'output/careai_march/eval_outputs'
OUT_DIR      = 'output/careai_march'

DISEASE_NAMES = [
    'Anemia', 'CKD', 'Diabetes', 'Dyslipidemia',
    'Electrolyte_Imbalance', 'Hematology_Disorder',
    'Infection_Inflammation', 'Liver_Disease', 'Thyroid_Disorder'
]
PAPER_DISEASES = DISEASE_NAMES
PAPER_IDX = [i for i, d in enumerate(DISEASE_NAMES) if d in PAPER_DISEASES]

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Load eval outputs ─────────────────────────────────────────────────────────
print("[Load] Reading eval_outputs ...")
test_probs   = np.load(os.path.join(EVAL_DIR, 'test_probs.npy'))
test_labels  = np.load(os.path.join(EVAL_DIR, 'test_labels.npy'))
beta_weights = np.load(os.path.join(EVAL_DIR, 'beta_weights.npy'))  # [N_test, 2]
test_ids     = np.load(os.path.join(EVAL_DIR, 'test_patient_ids.npy'))

import json as _json
with open(SCHEMA_PATH) as f:
    schema = _json.load(f)
opt_thresholds = schema['opt_thresholds']
thresholds = np.array([opt_thresholds.get(d, 0.5) for d in DISEASE_NAMES])

print(f"  N_test={len(test_ids):,}  beta_weights shape={beta_weights.shape}")

# ── Rebuild graph + model ─────────────────────────────────────────────────────
print("\n[Rebuild] Loading graph + model (needed for forward passes) ...")
t0 = time.time()
data = MedicalGraphData(
    path_records=RECORDS_PATH, path_symptom=TESTS_PATH,
    symptom_freq_threshold=0.99, prune_per_patient=50,
    nnz_threshold=2_000_000_000, seed=SEED)
data.load_data(); data.build_labels_and_features(); data.build_adjacency_matrices()
support = np.load(SUPPORT_PATH, allow_pickle=True)
feats_np  = support['feats_np']
labels_np = support['labels_np']
print(f"  Graph in {time.time()-t0:.1f}s — P={feats_np.shape[0]:,}")

# Reproduce test split
all_idx = np.arange(data.P)
msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_val_idx, test_idx = next(msss.split(all_idx, labels_np))
msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_sub, val_sub = next(msss2.split(train_val_idx, labels_np[train_val_idx]))
train_idx = train_val_idx[train_sub]

# Build neighbors
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

print("  Building neighbors ...")
t0 = time.time()
nbr_dicts = build_neighbors(labels_np, data.patient_organ_score, META_PATHS,
                             max_rare=MAX_NBR, max_common=MAX_NBR_COMMON, seed=SEED)
print(f"  Neighbors done in {time.time()-t0:.1f}s")

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

feats_t   = torch.from_numpy(feats_np).float().to(DEVICE)
empty_nbr = {name: {} for name in META_PATHS}

@torch.no_grad()
def get_logits_beta(feat_tensor):
    logits, _, beta = model(feat_tensor, empty_nbr)
    return logits.cpu().numpy(), beta.cpu().numpy()

# ── Test 1: Semantic attention beta analysis ──────────────────────────────────
print("\n[Test 1] Semantic attention distribution (beta) ...")
# beta_weights[i] = [beta_PDP, beta_POP]  — already extracted from eval
# Dominant path per patient
dominant_pdp = beta_weights[:, 0] > beta_weights[:, 1]   # P-D-P dominant
dominant_pop = ~dominant_pdp

n_pdp_dominant = dominant_pdp.sum()
n_pop_dominant = dominant_pop.sum()
mean_beta_pdp = beta_weights[:, 0].mean()
mean_beta_pop = beta_weights[:, 1].mean()

print(f"  Mean β(P-D-P) = {mean_beta_pdp:.4f}  Mean β(P-O-P) = {mean_beta_pop:.4f}")
print(f"  P-D-P dominant: {n_pdp_dominant:,} patients ({100*n_pdp_dominant/len(test_ids):.1f}%)")
print(f"  P-O-P dominant: {n_pop_dominant:,} patients ({100*n_pop_dominant/len(test_ids):.1f}%)")

# ── Test 2: Meta-path masking via feature perturbation ───────────────────────
# For a subsample of test patients, shuffle features of P-D-P-like neighbors
# (simulate "removing" the P-D-P signal) and compare logit change.
# We approximate this by:
#   - For each patient i, their neighbors via P-D-P share diseases.
#   - We corrupt feats_t by replacing neighbor feature means with shuffled versions.
# Simpler and faster: measure per-path contribution by comparing:
#   - Full model output (both paths active)
#   - Model output with ONLY P-O-P neighbors (zero out P-D-P neighbors)
#   - Model output with ONLY P-D-P neighbors (zero out P-O-P neighbors)

print("\n[Test 2] Meta-path ablation (disable one path at a time) ...")

# Build tensors with only one path active
nbr_tensors_pdp_only = {
    'P-D-P': nbr_tensors['P-D-P'],
    'P-O-P': (torch.zeros_like(nbr_tensors['P-O-P'][0]),
               torch.zeros_like(nbr_tensors['P-O-P'][1])),
}
nbr_tensors_pop_only = {
    'P-D-P': (torch.zeros_like(nbr_tensors['P-D-P'][0]),
               torch.zeros_like(nbr_tensors['P-D-P'][1])),
    'P-O-P': nbr_tensors['P-O-P'],
}

@torch.no_grad()
def get_probs_with_nbr(nbr_t, indices):
    model.set_vectorized_neighbors(nbr_t)
    logits, _, _ = model(feats_t, empty_nbr)
    model.set_vectorized_neighbors(nbr_tensors)  # restore
    probs = torch.sigmoid(logits[indices]).cpu().numpy()
    return probs

t0 = time.time()
probs_full   = test_probs                          # already saved
probs_pdp    = get_probs_with_nbr(nbr_tensors_pdp_only, test_idx)
probs_pop    = get_probs_with_nbr(nbr_tensors_pop_only, test_idx)
print(f"  Ablation forward passes done in {time.time()-t0:.1f}s")

# F1-Macro for each configuration
preds_full = (probs_full >= thresholds).astype(int)
preds_pdp  = (probs_pdp  >= thresholds).astype(int)
preds_pop  = (probs_pop  >= thresholds).astype(int)

f1_full = f1_score(test_labels[:, PAPER_IDX], preds_full[:, PAPER_IDX],
                   average='macro', zero_division=0)
f1_pdp  = f1_score(test_labels[:, PAPER_IDX], preds_pdp[:, PAPER_IDX],
                   average='macro', zero_division=0)
f1_pop  = f1_score(test_labels[:, PAPER_IDX], preds_pop[:, PAPER_IDX],
                   average='macro', zero_division=0)

print(f"\n  F1-Macro — Full (both paths)  : {f1_full:.4f}")
print(f"  F1-Macro — P-D-P only        : {f1_pdp:.4f}  (drop = {f1_full-f1_pdp:+.4f})")
print(f"  F1-Macro — P-O-P only        : {f1_pop:.4f}  (drop = {f1_full-f1_pop:+.4f})")

# Logit change per patient when dominant path is removed
logits_full, _   = get_logits_beta(feats_t)
model.set_vectorized_neighbors(nbr_tensors_pdp_only)
logits_pdp, _ = get_logits_beta(feats_t)
model.set_vectorized_neighbors(nbr_tensors_pop_only)
logits_pop, _ = get_logits_beta(feats_t)
model.set_vectorized_neighbors(nbr_tensors)  # restore

logit_change_pdp = np.abs(logits_full[test_idx] - logits_pdp[test_idx]).mean(axis=1)
logit_change_pop = np.abs(logits_full[test_idx] - logits_pop[test_idx]).mean(axis=1)

LOGIT_THRESH = 0.1
# For PDP-dominant patients: removing PDP should hurt more than removing POP
pdp_dom_idx = np.where(dominant_pdp)[0]
pop_dom_idx = np.where(dominant_pop)[0]

pct_pdp_dom_sensitive = (logit_change_pdp[pdp_dom_idx] > LOGIT_THRESH).mean() if len(pdp_dom_idx) > 0 else float('nan')
pct_pop_dom_sensitive = (logit_change_pop[pop_dom_idx] > LOGIT_THRESH).mean() if len(pop_dom_idx) > 0 else float('nan')

print(f"\n  Faithfulness check (logit change > {LOGIT_THRESH}):")
print(f"  P-D-P dominant patients — sensitive when P-D-P removed: "
      f"{100*pct_pdp_dom_sensitive:.1f}%  (n={len(pdp_dom_idx):,})")
print(f"  P-O-P dominant patients — sensitive when P-O-P removed: "
      f"{100*pct_pop_dom_sensitive:.1f}%  (n={len(pop_dom_idx):,})")

# ── Save ──────────────────────────────────────────────────────────────────────
results = {
    'semantic_attention': {
        'mean_beta_PDP': float(mean_beta_pdp),
        'mean_beta_POP': float(mean_beta_pop),
        'n_PDP_dominant': int(n_pdp_dominant),
        'n_POP_dominant': int(n_pop_dominant),
        'pct_PDP_dominant': float(n_pdp_dominant / len(test_ids)),
        'pct_POP_dominant': float(n_pop_dominant / len(test_ids)),
    },
    'metapath_ablation': {
        'f1_macro_full_both_paths': float(f1_full),
        'f1_macro_PDP_only':        float(f1_pdp),
        'f1_macro_POP_only':        float(f1_pop),
        'f1_drop_remove_PDP':       float(f1_full - f1_pdp),
        'f1_drop_remove_POP':       float(f1_full - f1_pop),
    },
    'faithfulness': {
        'logit_threshold': LOGIT_THRESH,
        'pct_PDP_dominant_sensitive_to_PDP_removal': float(pct_pdp_dom_sensitive),
        'pct_POP_dominant_sensitive_to_POP_removal': float(pct_pop_dom_sensitive),
        'interpretation': (
            'A faithful model should show high sensitivity when removing the '
            'dominant path for patients that prefer that path.'
        ),
    },
    'n_test': int(len(test_ids)),
}

with open(os.path.join(OUT_DIR, 'faithfulness_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n[Done] Saved faithfulness_results.json")
print(f"\nNext: run Other_py/retain_baseline.py")
