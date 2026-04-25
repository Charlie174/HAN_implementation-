"""
patient_conditioning_ablation.py
================================
Ablation: Does patient-conditioned semantic attention (q_i = W_q * h_i)
improve over HAN's original global query (q = learnable parameter)?

Trains two models from scratch with identical hyperparameters:
  1. HAN++ (patient-conditioned) — use_global_query=False (default)
  2. HAN  (global query)        — use_global_query=True

Reports F1-Macro, AUC-ROC, per-disease F1, and beta variance.

Usage:
    python Other_py/patient_conditioning_ablation.py
"""

import os, sys, json, random, time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from HAN import MedicalGraphData
from HAN.model import HANPP_Disease

# ── Config ────────────────────────────────────────────────────────────────────
SEED       = 42
HIDDEN_DIM = 256
OUT_DIM    = 128
NUM_HEADS  = 4
DROPOUT    = 0.3
LR         = 3e-4
WD         = 1e-4
EPOCHS     = 100
PATIENCE   = 20
MAX_NBR    = 50
MAX_NBR_COMMON = 10
META_PATHS = ['P-D-P', 'P-O-P']

RECORDS_PATH = 'data/dataset_careai_new/processed/records_labeled_new.csv'
TESTS_PATH   = 'data/dataset_careai_new/processed/test_reference_new.csv'
SUPPORT_PATH = 'output/careai_march/inductive_support.npz'
SCHEMA_PATH  = 'output/careai_march/inductive_schema.json'
OUT_DIR      = 'output/careai_march'

DISEASE_NAMES = [
    'Anemia', 'CKD', 'Diabetes', 'Dyslipidemia',
    'Electrolyte_Imbalance', 'Hematology_Disorder',
    'Infection_Inflammation', 'Liver_Disease', 'Thyroid_Disorder'
]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Utilities ─────────────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
                    cap = max_common if disease_rates[d] > 0.5 else max_rare
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


def focal_bce(logits, targets, pos_weight, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction='none')
    prob = torch.sigmoid(logits)
    p_t = targets * prob + (1 - targets) * (1 - prob)
    return ((1 - p_t) ** gamma * bce).mean()


@torch.no_grad()
def evaluate(model, feats, labels_np, idx_t, threshold=0.5):
    model.eval()
    empty_nbr = {name: {} for name in model.metapath_names}
    logits, _, beta = model(feats, empty_nbr)
    probs = torch.sigmoid(logits[idx_t]).cpu().numpy()
    beta_np = beta[idx_t].cpu().numpy()
    preds = (probs >= threshold).astype(int)
    labels = labels_np[idx_t.cpu().numpy()]

    f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
    f1_micro = f1_score(labels, preds, average='micro', zero_division=0)

    aucs = []
    for d in range(labels.shape[1]):
        pos = labels[:, d].sum()
        if 0 < pos < len(labels):
            aucs.append(roc_auc_score(labels[:, d], probs[:, d]))
    mean_auc = float(np.mean(aucs)) if aucs else 0.0

    per_disease_f1 = {}
    for j, dname in enumerate(DISEASE_NAMES):
        per_disease_f1[dname] = float(f1_score(labels[:, j], preds[:, j], zero_division=0))

    return {
        'f1_macro': f1_macro, 'f1_micro': f1_micro,
        'auc_roc': mean_auc, 'per_disease_f1': per_disease_f1,
        'beta_mean': beta_np.mean(axis=0).tolist(),
        'beta_std': beta_np.std(axis=0).tolist(),
    }


def find_optimal_thresholds(model, feats, labels_np, val_idx):
    model.eval()
    empty_nbr = {name: {} for name in model.metapath_names}
    with torch.no_grad():
        logits, _, _ = model(feats, empty_nbr)
        probs = torch.sigmoid(logits[val_idx]).cpu().numpy()
    labels_val = labels_np[val_idx.cpu().numpy()]

    thresholds = np.zeros(len(DISEASE_NAMES))
    for j in range(len(DISEASE_NAMES)):
        best_f1, best_t = 0.0, 0.5
        for t in np.arange(0.01, 0.96, 0.01):
            preds_j = (probs[:, j] >= t).astype(int)
            f1 = f1_score(labels_val[:, j], preds_j, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[j] = best_t
    return thresholds


def train_model(feats_t, labels_t, labels_np, train_idx_t, val_idx_t,
                pos_weight, in_dim, use_global_query, label):
    set_seed(SEED)

    model = HANPP_Disease(
        in_dim=in_dim, hidden_dim=HIDDEN_DIM, out_dim=OUT_DIM,
        metapath_names=META_PATHS, num_heads=NUM_HEADS,
        num_diseases=len(DISEASE_NAMES), dropout=DROPOUT,
        use_global_query=use_global_query,
    ).to(DEVICE)
    model.set_vectorized_neighbors(nbr_tensors_global)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'='*60}")
    print(f"Training: {label}")
    print(f"  use_global_query={use_global_query}  params={params:,}")
    print(f"{'='*60}")
    print(f"{'Epoch':>6}  {'Loss':>8}  {'Val F1':>8}  {'Val AUC':>8}  {'Time':>6}")
    print("-" * 44)

    best_val_auc = 0.0
    best_state = None
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        optimizer.zero_grad()
        empty_nbr = {name: {} for name in META_PATHS}
        logits, _, _ = model(feats_t, empty_nbr)
        loss = focal_bce(logits[train_idx_t], labels_t[train_idx_t], pos_weight, gamma=2.0)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        val_m = evaluate(model, feats_t, labels_np, val_idx_t)
        elapsed = time.time() - t0

        if epoch % 10 == 0 or epoch <= 5:
            print(f"  {epoch:>4}  {loss.item():>8.4f}  {val_m['f1_macro']:>8.4f}  "
                  f"{val_m['auc_roc']:>8.4f}  {elapsed:>5.1f}s")

        if val_m['auc_roc'] > best_val_auc:
            best_val_auc = val_m['auc_roc']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stop at epoch {epoch} (best={best_epoch})")
                break

    model.load_state_dict(best_state)
    model.to(DEVICE)
    print(f"  Best epoch: {best_epoch}  Val AUC: {best_val_auc:.4f}")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    set_seed(SEED)
    print(f"Device: {DEVICE}")
    print(f"Patient-Conditioning Ablation: global-query HAN vs patient-conditioned HAN++\n")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading data...")
    t0 = time.time()
    data = MedicalGraphData(
        path_records=RECORDS_PATH, path_symptom=TESTS_PATH,
        symptom_freq_threshold=0.99, prune_per_patient=50,
        nnz_threshold=2_000_000_000, seed=SEED)
    data.load_data()
    data.build_labels_and_features()
    data.build_adjacency_matrices()
    print(f"  Graph loaded in {time.time()-t0:.1f}s  (P={data.P:,})")

    # ── Load labels (from support for consistent ordering) ────────────────────
    support = np.load(SUPPORT_PATH, allow_pickle=True)
    labels_np = support['labels_np'].astype(np.float32)

    # ── Split ─────────────────────────────────────────────────────────────────
    all_idx = np.arange(labels_np.shape[0])
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_val_idx, test_idx = next(msss.split(all_idx, labels_np))

    msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, val_idx = next(msss2.split(train_val_idx, labels_np[train_val_idx]))
    train_idx = train_val_idx[train_idx]
    val_idx = train_val_idx[val_idx]

    print(f"  Train: {len(train_idx):,}  Val: {len(val_idx):,}  Test: {len(test_idx):,}")

    # ── Build neighbors ───────────────────────────────────────────────────────
    print("Building neighbors...")
    t0 = time.time()
    nbr_dicts = build_neighbors(labels_np, data.patient_organ_score, META_PATHS,
                                max_rare=MAX_NBR, max_common=MAX_NBR_COMMON, seed=SEED)
    print(f"  Done in {time.time()-t0:.1f}s")

    # ── Tensors ───────────────────────────────────────────────────────────────
    feats_t = torch.from_numpy(data.patient_feats.astype(np.float32)).to(DEVICE)
    labels_t = torch.from_numpy(labels_np).float().to(DEVICE)
    train_idx_t = torch.tensor(train_idx, dtype=torch.long)
    val_idx_t = torch.tensor(val_idx, dtype=torch.long)
    test_idx_t = torch.tensor(test_idx, dtype=torch.long)
    in_dim = data.patient_feats.shape[1]

    nbr_tensors_global = {}
    for name in META_PATHS:
        if name in nbr_dicts:
            idx_t, mask_t = neighbors_to_tensors(nbr_dicts[name], data.P, MAX_NBR)
            nbr_tensors_global[name] = (idx_t.to(DEVICE), mask_t.to(DEVICE))

    # ── Pos weight ────────────────────────────────────────────────────────────
    pos_counts = labels_np[train_idx].sum(axis=0) + 1e-6
    neg_counts = len(train_idx) - pos_counts
    pos_weight = torch.tensor(neg_counts / pos_counts, dtype=torch.float32)
    pos_weight = torch.clamp(pos_weight, max=10.0).to(DEVICE)

    # ── Train both variants ───────────────────────────────────────────────────
    model_pc = train_model(feats_t, labels_t, labels_np, train_idx_t, val_idx_t,
                           pos_weight, in_dim, use_global_query=False,
                           label="HAN++ (patient-conditioned)")

    model_gq = train_model(feats_t, labels_t, labels_np, train_idx_t, val_idx_t,
                           pos_weight, in_dim, use_global_query=True,
                           label="HAN (global query)")

    # ── Evaluate both on test set ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)

    results = {}
    for name, model in [("patient_conditioned", model_pc), ("global_query", model_gq)]:
        model.set_vectorized_neighbors(nbr_tensors_global)
        opt_thr = find_optimal_thresholds(model, feats_t, labels_np, val_idx_t)
        test_m = evaluate(model, feats_t, labels_np, test_idx_t, threshold=opt_thr)

        print(f"\n  {name}:")
        print(f"    F1-Macro: {test_m['f1_macro']:.4f}")
        print(f"    F1-Micro: {test_m['f1_micro']:.4f}")
        print(f"    AUC-ROC:  {test_m['auc_roc']:.4f}")
        print(f"    Beta mean: {[f'{b:.4f}' for b in test_m['beta_mean']]}")
        print(f"    Beta std:  {[f'{b:.4f}' for b in test_m['beta_std']]}")
        print(f"    Per-disease F1:")
        for d, f1 in test_m['per_disease_f1'].items():
            print(f"      {d:<28} {f1:.4f}")

        results[name] = {
            'f1_macro': test_m['f1_macro'],
            'f1_micro': test_m['f1_micro'],
            'auc_roc': test_m['auc_roc'],
            'per_disease_f1': test_m['per_disease_f1'],
            'beta_mean': test_m['beta_mean'],
            'beta_std': test_m['beta_std'],
            'opt_thresholds': opt_thr.tolist(),
        }

    # ── Summary ───────────────────────────────────────────────────────────────
    delta = results['patient_conditioned']['f1_macro'] - results['global_query']['f1_macro']
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Patient-conditioned F1-Macro: {results['patient_conditioned']['f1_macro']:.4f}")
    print(f"  Global-query F1-Macro:        {results['global_query']['f1_macro']:.4f}")
    print(f"  Delta (PC - GQ):              {delta:+.4f}")
    if delta > 0:
        print(f"  --> Patient conditioning IMPROVES F1 by {delta:.4f}")
    elif delta < 0:
        print(f"  --> Patient conditioning HURTS F1 by {abs(delta):.4f}")
    else:
        print(f"  --> No difference")

    # ── Save ──────────────────────────────────────────────────────────────────
    output = {
        'experiment': 'Patient-Conditioned Semantic Attention Ablation',
        'description': 'Compares HAN++ (q_i = W_q * h_i) vs HAN (q = global parameter)',
        'dataset': f'CareAI April 2026 ({labels_np.shape[0]:,} patients, {len(DISEASE_NAMES)} diseases)',
        'hyperparams': {
            'hidden_dim': HIDDEN_DIM, 'out_dim': OUT_DIM, 'num_heads': NUM_HEADS,
            'dropout': DROPOUT, 'lr': LR, 'wd': WD, 'epochs': EPOCHS,
            'patience': PATIENCE, 'seed': SEED,
        },
        'split': {
            'train': len(train_idx), 'val': len(val_idx), 'test': len(test_idx),
        },
        'results': results,
        'delta_f1_macro': delta,
    }
    out_path = os.path.join(OUT_DIR, 'patient_conditioning_ablation.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")
