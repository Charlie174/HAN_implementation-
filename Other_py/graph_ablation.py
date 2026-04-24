"""
graph_ablation.py — Day 7 of 13-day plan
=========================================
Ablation study: how sensitive is HAN++ to graph construction choices?

Tests:
  1. symptom_freq_threshold sensitivity: [0.50, 0.70, 0.90, 0.99] (0.99 = used in training)
     - Higher threshold = fewer symptoms kept (sparser graph)
     - Reports F1-Macro for each, using fixed model weights

  2. Meta-path combination ablation:
     - P-D-P only (zeroed P-O-P neighbors)
     - P-O-P only (zeroed P-D-P neighbors)
     - P-D-P + P-O-P (both active) — baseline

NOTE: Graph is rebuilt for each threshold; model weights are NOT retrained.
This measures how robust the trained model is to graph construction changes.

Outputs:
  output/careai_march/graph_ablation_results.json

Usage:
    python Other_py/graph_ablation.py
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

SEED       = 42
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
SUPPORT_PATH = 'output/careai_march/inductive_support.npz'
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
print(f"Device: {DEVICE}")

# ── Load labels from support (for consistent split) ───────────────────────────
support  = np.load(SUPPORT_PATH, allow_pickle=True)
labels_np = support['labels_np'].astype(np.float32)   # [N, 9]

with open(SCHEMA_PATH) as f:
    schema = json.load(f)
opt_thresholds = schema['opt_thresholds']
thresholds = np.array([opt_thresholds.get(d, 0.5) for d in DISEASE_NAMES])


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


def evaluate(model, feats_t, test_idx, labels_np, nbr_tensors, thresholds):
    model.set_vectorized_neighbors(nbr_tensors)
    empty_nbr = {name: {} for name in META_PATHS}
    with torch.no_grad():
        logits, _, _ = model(feats_t, empty_nbr)
    probs = torch.sigmoid(logits[test_idx]).cpu().numpy()
    preds = (probs >= thresholds).astype(int)
    test_labels = labels_np[test_idx]

    f1_macro = f1_score(test_labels[:, PAPER_IDX], preds[:, PAPER_IDX],
                        average='macro', zero_division=0)
    try:
        auc = roc_auc_score(test_labels[:, PAPER_IDX], probs[:, PAPER_IDX], average='macro')
    except Exception:
        auc = float('nan')
    return float(f1_macro), float(auc)


# ── Reproduce test split ──────────────────────────────────────────────────────
all_idx = np.arange(labels_np.shape[0])
msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_val_idx, test_idx = next(msss.split(all_idx, labels_np))
print(f"Test split: {len(test_idx):,} patients")


# ── Part 1: symptom_freq_threshold ablation ───────────────────────────────────
THRESHOLDS_TO_TEST = [0.50, 0.70, 0.90, 0.99]
print(f"\n{'='*60}")
print(f"Part 1: symptom_freq_threshold ablation")
print(f"{'='*60}")
print(f"\n{'Threshold':>12} {'N_symptoms':>12} {'F1-Macro':>10} {'AUC-ROC':>10}")
print("-" * 50)

threshold_results = []
for freq_thresh in THRESHOLDS_TO_TEST:
    t0 = time.time()
    print(f"  Building graph (threshold={freq_thresh}) ...", flush=True)

    data_t = MedicalGraphData(
        path_records=RECORDS_PATH, path_symptom=TESTS_PATH,
        symptom_freq_threshold=freq_thresh, prune_per_patient=50,
        nnz_threshold=2_000_000_000, seed=SEED)
    data_t.load_data()
    data_t.build_labels_and_features()
    data_t.build_adjacency_matrices()

    feats_np_t = data_t.patient_feats.astype(np.float32)
    feats_t_t  = torch.from_numpy(feats_np_t).float().to(DEVICE)
    n_symptoms = data_t.S

    # Load model (same weights every time)
    model_t = HANPP_Disease(
        in_dim=feats_np_t.shape[1], hidden_dim=HIDDEN_DIM, out_dim=OUT_DIM,
        metapath_names=META_PATHS, num_heads=NUM_HEADS,
        num_diseases=len(DISEASE_NAMES), dropout=DROPOUT,
    ).to(DEVICE)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)

    in_dim_actual = feats_np_t.shape[1]
    in_dim_model = list(ckpt.values())[0].shape[1]  # project.weight: [hidden, in_dim]
    if in_dim_actual != in_dim_model:
        f1_macro, auc = float('nan'), float('nan')
        print(f"  Skipping threshold={freq_thresh}: in_dim={in_dim_actual} != {in_dim_model} (model mismatch)")
    else:
        model_t.load_state_dict(ckpt)
        model_t.eval()

        nbr_dicts_t = build_neighbors(labels_np, data_t.patient_organ_score, META_PATHS,
                                       max_rare=MAX_NBR, max_common=MAX_NBR_COMMON, seed=SEED)
        nbr_tensors_t = {}
        for name in META_PATHS:
            if name in nbr_dicts_t:
                idx_t, mask_t = neighbors_to_tensors(nbr_dicts_t[name], data_t.P, MAX_NBR)
                nbr_tensors_t[name] = (idx_t.to(DEVICE), mask_t.to(DEVICE))
            else:
                zero_idx  = torch.zeros(data_t.P, MAX_NBR, dtype=torch.long)
                zero_mask = torch.zeros(data_t.P, MAX_NBR, dtype=torch.float)
                nbr_tensors_t[name] = (zero_idx.to(DEVICE), zero_mask.to(DEVICE))

        f1_macro, auc = evaluate(model_t, feats_t_t, test_idx, labels_np,
                                  nbr_tensors_t, thresholds)
        elapsed = time.time() - t0
        print(f"  threshold={freq_thresh}: n_symptoms={n_symptoms}  "
              f"F1-Macro={f1_macro:.4f}  AUC={auc:.4f}  [{elapsed:.1f}s]")

    threshold_results.append({
        'symptom_freq_threshold': float(freq_thresh),
        'n_symptoms_kept': int(n_symptoms) if not isinstance(n_symptoms, float) else None,
        'f1_macro': f1_macro,
        'auc_roc': auc,
    })

# ── Part 2: Meta-path combination ablation ───────────────────────────────────
print(f"\n{'='*60}")
print(f"Part 2: Meta-path combination ablation (threshold=0.99)")
print(f"{'='*60}")

# Load the standard graph (threshold=0.99)
print("\n  Loading standard graph (threshold=0.99) ...")
t0 = time.time()
data_std = MedicalGraphData(
    path_records=RECORDS_PATH, path_symptom=TESTS_PATH,
    symptom_freq_threshold=0.99, prune_per_patient=50,
    nnz_threshold=2_000_000_000, seed=SEED)
data_std.load_data()
data_std.build_labels_and_features()
data_std.build_adjacency_matrices()
feats_np_std = data_std.patient_feats.astype(np.float32)
feats_t_std  = torch.from_numpy(feats_np_std).float().to(DEVICE)
print(f"  Graph loaded in {time.time()-t0:.1f}s")

model_std = HANPP_Disease(
    in_dim=feats_np_std.shape[1], hidden_dim=HIDDEN_DIM, out_dim=OUT_DIM,
    metapath_names=META_PATHS, num_heads=NUM_HEADS,
    num_diseases=len(DISEASE_NAMES), dropout=DROPOUT,
).to(DEVICE)
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
model_std.load_state_dict(ckpt)
model_std.eval()

t0 = time.time()
nbr_dicts_std = build_neighbors(labels_np, data_std.patient_organ_score, META_PATHS,
                                  max_rare=MAX_NBR, max_common=MAX_NBR_COMMON, seed=SEED)
print(f"  Neighbors built in {time.time()-t0:.1f}s")

def make_nbr_tensors(pdp_dict, pop_dict, N, max_nbr, active_pdp=True, active_pop=True):
    result = {}
    for name, d in [('P-D-P', pdp_dict), ('P-O-P', pop_dict)]:
        active = active_pdp if name == 'P-D-P' else active_pop
        if active and d:
            idx_t, mask_t = neighbors_to_tensors(d, N, max_nbr)
        else:
            idx_t  = torch.zeros(N, max_nbr, dtype=torch.long)
            mask_t = torch.zeros(N, max_nbr, dtype=torch.float)
        result[name] = (idx_t.to(DEVICE), mask_t.to(DEVICE))
    return result

configs = [
    ('P-D-P + P-O-P (baseline)', True,  True),
    ('P-D-P only',               True,  False),
    ('P-O-P only',               False, True),
]

metapath_results = []
print(f"\n{'Configuration':30s} {'F1-Macro':>10} {'AUC-ROC':>10}")
print("-" * 55)

for label, use_pdp, use_pop in configs:
    nbr_t = make_nbr_tensors(
        nbr_dicts_std.get('P-D-P', {}),
        nbr_dicts_std.get('P-O-P', {}),
        data_std.P, MAX_NBR,
        active_pdp=use_pdp, active_pop=use_pop
    )
    f1_macro, auc = evaluate(model_std, feats_t_std, test_idx, labels_np, nbr_t, thresholds)
    print(f"  {label:30s}  F1-Macro={f1_macro:.4f}  AUC={auc:.4f}")
    metapath_results.append({
        'config': label,
        'use_PDP': use_pdp,
        'use_POP': use_pop,
        'f1_macro': f1_macro,
        'auc_roc': auc,
    })

# ── Save ──────────────────────────────────────────────────────────────────────
output = {
    'model': 'HANPP_Disease v6 (weights fixed, trained with threshold=0.99, P-D-P+P-O-P)',
    'note': 'No retraining — model weights are fixed. Tests graph construction sensitivity.',
    'threshold_ablation': threshold_results,
    'metapath_ablation': metapath_results,
}
with open(os.path.join(OUT_DIR, 'graph_ablation_results.json'), 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n[Done] Saved graph_ablation_results.json")
print(f"\nNext: update paper/main.tex (Days 9-10)")
