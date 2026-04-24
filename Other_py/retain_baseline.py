"""
retain_baseline.py — Day 5 of 13-day plan
==========================================
RETAIN (Choi et al., 2016) — interpretable baseline for comparison with HAN++.

Since all patients here are single-visit (one feature vector each), we implement
a 1-visit RETAIN:
  - Alpha (visit-level) attention: scalar weight per visit (trivially 1.0 for 1 visit)
  - Beta (feature-level) attention: feature-wise weights via bidirectional GRU
  - Final: h_i = alpha_i * (beta_i ⊙ x_i)

For single-visit, RETAIN reduces to:
    h = softmax(e_alpha) * (tanh(W_beta * x + b_beta) ⊙ x)
where e_alpha and W_beta come from the GRU (here: a single linear layer).

Outputs:
  output/careai_march/retain_results.json

Usage:
    python Other_py/retain_baseline.py
"""

import os, sys, json, random, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, roc_auc_score, brier_score_loss
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from HAN import MedicalGraphData

SEED       = 42
EPOCHS     = 50
BATCH_SIZE = 512
LR         = 1e-3
WEIGHT_DECAY = 1e-4
HIDDEN_DIM = 256
OUT_DIR    = 'output/careai_march'
RECORDS_PATH = 'data/dataset_careai_new/processed/records_labeled_new.csv'
TESTS_PATH   = 'data/dataset_careai_new/processed/test_reference_new.csv'
SCHEMA_PATH  = 'output/careai_march/inductive_schema.json'
SUPPORT_PATH = 'output/careai_march/inductive_support.npz'

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


# ── RETAIN Model ──────────────────────────────────────────────────────────────
class RETAIN1Visit(nn.Module):
    """
    Single-visit RETAIN.
    Alpha: scalar logit → trivially 1 (one visit)
    Beta:  feature-level attention via linear layer (bidirectional GRU → linear for 1 visit)
    """
    def __init__(self, in_dim, hidden_dim, num_diseases, dropout=0.3):
        super().__init__()
        self.embed  = nn.Linear(in_dim, hidden_dim)
        # Alpha: visit-level logit (scalar per visit)
        self.alpha  = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                                     nn.Linear(hidden_dim, 1))
        # Beta: feature-level attention weights (dim = in_dim)
        self.beta   = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                                     nn.Linear(hidden_dim, in_dim), nn.Tanh())
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(in_dim, num_diseases)

    def forward(self, x):
        # x: [B, in_dim]
        h = torch.relu(self.embed(x))            # [B, hidden_dim]
        h = self.dropout(h)
        # Alpha: single visit → weight = 1 (softmax of single element = 1)
        # Beta: feature-level attention
        beta = self.beta(h)                      # [B, in_dim]  (tanh output)
        # Context vector: element-wise product of beta attention and input
        context = beta * x                       # [B, in_dim]
        context = self.dropout(context)
        logits = self.classifier(context)        # [B, num_diseases]
        return logits


# ── Load data ─────────────────────────────────────────────────────────────────
print("\n[Load] Building graph data ...")
t0 = time.time()
data = MedicalGraphData(
    path_records=RECORDS_PATH, path_symptom=TESTS_PATH,
    symptom_freq_threshold=0.99, prune_per_patient=50,
    nnz_threshold=2_000_000_000, seed=SEED)
data.load_data(); data.build_labels_and_features(); data.build_adjacency_matrices()
feats_np  = data.patient_feats.astype(np.float32)
print(f"  Done in {time.time()-t0:.1f}s — P={data.P:,}, features={feats_np.shape[1]}")

support = np.load(SUPPORT_PATH, allow_pickle=True)
labels_np = support['labels_np'].astype(np.float32)
print(f"  labels_np from inductive_support.npz: {labels_np.shape}")

with open(SCHEMA_PATH) as f:
    schema = json.load(f)
opt_thresholds = schema['opt_thresholds']

# ── Reproduce split ───────────────────────────────────────────────────────────
all_idx = np.arange(data.P)
msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_val_idx, test_idx = next(msss.split(all_idx, labels_np))
msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_sub, val_sub = next(msss2.split(train_val_idx, labels_np[train_val_idx]))
train_idx = train_val_idx[train_sub]
val_idx   = train_val_idx[val_sub]
print(f"  Split — Train: {len(train_idx):,}  Val: {len(val_idx):,}  Test: {len(test_idx):,}")

X_train = torch.from_numpy(feats_np[train_idx]).float()
y_train = torch.from_numpy(labels_np[train_idx]).float()
X_val   = torch.from_numpy(feats_np[val_idx]).float()
y_val   = torch.from_numpy(labels_np[val_idx]).float()
X_test  = torch.from_numpy(feats_np[test_idx]).float()
y_test  = torch.from_numpy(labels_np[test_idx]).float()

train_loader = DataLoader(TensorDataset(X_train, y_train),
                          batch_size=BATCH_SIZE, shuffle=True)


# ── Train RETAIN ──────────────────────────────────────────────────────────────
print("\n[Train] Training RETAIN (1-visit) ...")
model = RETAIN1Visit(
    in_dim=feats_np.shape[1], hidden_dim=HIDDEN_DIM,
    num_diseases=len(DISEASE_NAMES), dropout=0.3
).to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_val_f1 = 0.0
best_epoch  = 0
best_state  = None

X_val_d  = X_val.to(DEVICE)
X_test_d = X_test.to(DEVICE)

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(xb)
        loss   = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(xb)
    scheduler.step()
    train_loss /= len(train_idx)

    # Validation
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_d).cpu().numpy()
    val_probs = 1 / (1 + np.exp(-val_logits))
    val_preds = (val_probs[:, PAPER_IDX] >= 0.5).astype(int)
    val_f1 = f1_score(y_val.numpy()[:, PAPER_IDX], val_preds, average='macro', zero_division=0)

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_epoch  = epoch
        best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if epoch % 10 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d}/{EPOCHS}  loss={train_loss:.4f}  val_F1={val_f1:.4f}  "
              f"best={best_val_f1:.4f} @ep{best_epoch}")

print(f"\n  Training done — best val F1={best_val_f1:.4f} at epoch {best_epoch}")


# ── Evaluate on test ──────────────────────────────────────────────────────────
print("\n[Eval] Evaluating on test set ...")
model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    test_logits = model(X_test_d).cpu().numpy()
test_probs_retain = 1 / (1 + np.exp(-test_logits))

# Use fixed 0.5 threshold (RETAIN has no per-disease tuning here)
test_preds = (test_probs_retain[:, PAPER_IDX] >= 0.5).astype(int)
y_test_np  = y_test.numpy()

f1_macro = f1_score(y_test_np[:, PAPER_IDX], test_preds, average='macro', zero_division=0)
f1_micro = f1_score(y_test_np[:, PAPER_IDX], test_preds, average='micro', zero_division=0)

try:
    auc = roc_auc_score(y_test_np[:, PAPER_IDX], test_probs_retain[:, PAPER_IDX], average='macro')
except Exception as e:
    auc = float('nan')
    print(f"  AUC warning: {e}")

brier_per_disease = {}
for k, i in enumerate(PAPER_IDX):
    d = DISEASE_NAMES[i]
    bs = brier_score_loss(y_test_np[:, i], test_probs_retain[:, i])
    brier_per_disease[d] = float(bs)
brier_mean = float(np.mean(list(brier_per_disease.values())))

# Per-disease F1
per_disease = {}
for k, i in enumerate(PAPER_IDX):
    d = DISEASE_NAMES[i]
    f1_d = f1_score(y_test_np[:, i], test_preds[:, k], zero_division=0)
    per_disease[d] = {'f1': float(f1_d), 'brier': brier_per_disease[d]}

print(f"\n  RETAIN Test Results (8 paper diseases):")
print(f"  F1-Macro : {f1_macro:.4f}")
print(f"  F1-Micro : {f1_micro:.4f}")
print(f"  AUC-ROC  : {auc:.4f}")
print(f"  Brier    : {brier_mean:.4f}")
print(f"\n  Per-disease F1:")
for d, v in per_disease.items():
    print(f"    {d:25s}  F1={v['f1']:.4f}  Brier={v['brier']:.4f}")


# ── Save ──────────────────────────────────────────────────────────────────────
results = {
    'model': 'RETAIN (1-visit, single linear layer)',
    'reference': 'Choi et al., 2016, KDD',
    'dataset': 'CareAI March 2026 (95836 patients)',
    'diseases': PAPER_DISEASES,
    'split': '64/16/20 (train/val/test), seed=42',
    'hyperparams': {
        'hidden_dim': HIDDEN_DIM, 'epochs': EPOCHS,
        'batch_size': BATCH_SIZE, 'lr': LR, 'weight_decay': WEIGHT_DECAY,
    },
    'best_val_f1': float(best_val_f1),
    'best_epoch': int(best_epoch),
    'test_f1_macro': float(f1_macro),
    'test_f1_micro': float(f1_micro),
    'test_auc_roc':  float(auc),
    'test_brier_mean': brier_mean,
    'test_brier_per_disease': brier_per_disease,
    'per_disease': per_disease,
}
with open(os.path.join(OUT_DIR, 'retain_results.json'), 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n[Done] Saved retain_results.json")
print(f"  RETAIN F1-Macro={f1_macro:.4f} vs HAN++ F1-Macro=0.9553")
print(f"\nNext: run Other_py/multiseed_eval.py")
