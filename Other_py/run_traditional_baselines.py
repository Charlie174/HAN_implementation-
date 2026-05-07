"""
run_traditional_baselines.py — Traditional ML baselines on new dataset
=====================================================================
Runs 7 GridSearch-tuned traditional ML baselines on the 48,546-patient
high-confidence dataset for fair comparison with HAN++ v8.

Models: Decision Tree, Random Forest, XGBoost, SVM (Linear),
        Logistic Regression, KNN, Gaussian Naive Bayes.

All use MultiOutputClassifier with MultilabelStratifiedShuffleSplit
(seed=42, 80/20) matching HAN++ split.

Outputs:
  output/careai_march/traditional_baselines_results.json

Usage:
    python Other_py/run_traditional_baselines.py
"""

import os, sys, json, time
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (f1_score, accuracy_score, hamming_loss,
                             roc_auc_score, precision_score, recall_score)
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: xgboost not installed, skipping XGBoost baseline")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from HAN import MedicalGraphData

SEED = 42
RECORDS_PATH = 'data/dataset_careai_new/processed/records_labeled_new.csv'
TESTS_PATH   = 'data/dataset_careai_new/processed/test_reference_new.csv'
SUPPORT_PATH = 'output/careai_march/inductive_support.npz'
OUT_DIR      = 'output/careai_march'
PREDS_DIR    = 'output/careai_march/baseline_predictions'
os.makedirs(PREDS_DIR, exist_ok=True)

DISEASE_NAMES = [
    'Anemia', 'CKD', 'Diabetes', 'Dyslipidemia',
    'Electrolyte_Imbalance', 'Hematology_Disorder',
    'Infection_Inflammation', 'Liver_Disease', 'Thyroid_Disorder'
]

np.random.seed(SEED)

# ── Load data ────────────────────────────────────────────────────────────────
print("\n[Load] Building graph data ...")
t0 = time.time()
data = MedicalGraphData(
    path_records=RECORDS_PATH, path_symptom=TESTS_PATH,
    symptom_freq_threshold=0.99, prune_per_patient=50,
    nnz_threshold=2_000_000_000, seed=SEED)
data.load_data(); data.build_labels_and_features(); data.build_adjacency_matrices()
feats_np = data.patient_feats.astype(np.float32)
print(f"  Done in {time.time()-t0:.1f}s — P={data.P:,}, features={feats_np.shape[1]}")

support = np.load(SUPPORT_PATH, allow_pickle=True)
labels_np = support['labels_np'].astype(np.float32)
print(f"  labels from inductive_support.npz: {labels_np.shape}")

# ── Split ────────────────────────────────────────────────────────────────────
all_idx = np.arange(labels_np.shape[0])
msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_idx, test_idx = next(msss.split(all_idx, labels_np))

X_train = feats_np[train_idx]
y_train = labels_np[train_idx].astype(int)
X_test  = feats_np[test_idx]
y_test  = labels_np[test_idx].astype(int)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print(f"  Train: {len(train_idx):,}  Test: {len(test_idx):,}")
print(f"  Features: {X_train_s.shape[1]}  Diseases: {y_train.shape[1]}")

# ── Model definitions with GridSearch grids ──────────────────────────────────
models = [
    ("Naive Bayes", GaussianNB(), None),
    ("KNN", KNeighborsClassifier(),
     {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]}),
    ("Decision Tree", DecisionTreeClassifier(random_state=SEED),
     {"max_depth": [10, 20, 30, None], "min_samples_leaf": [1, 5, 10]}),
    ("Random Forest", RandomForestClassifier(random_state=SEED, n_jobs=-1),
     {"n_estimators": [100, 200], "max_depth": [10, 20, None],
      "min_samples_leaf": [1, 5]}),
    ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=SEED),
     {"C": [0.01, 0.1, 1.0, 10.0]}),
    ("SVM (Linear)", LinearSVC(max_iter=5000, random_state=SEED),
     {"C": [0.01, 0.1, 1.0]}),
]

if HAS_XGB:
    models.append(
        ("XGBoost", XGBClassifier(random_state=SEED, n_jobs=-1,
                                   use_label_encoder=False, eval_metric='logloss'),
         {"n_estimators": [100, 200], "max_depth": [3, 6, 10],
          "learning_rate": [0.01, 0.1]})
    )

# ── Train and evaluate ───────────────────────────────────────────────────────
results = []
print(f"\n{'='*70}")
print(f"{'Model':25s} {'Accuracy':>10} {'F1-Micro':>10} {'F1-Macro':>10} {'HL':>10}")
print(f"{'='*70}")

for name, base_model, param_grid in models:
    t0 = time.time()
    print(f"  Training {name}...", flush=True)

    if param_grid:
        gs = GridSearchCV(
            base_model, param_grid, cv=5, scoring='f1_macro',
            n_jobs=-1, refit=True
        )
        moc = MultiOutputClassifier(gs, n_jobs=1)
    else:
        moc = MultiOutputClassifier(base_model, n_jobs=-1)

    try:
        moc.fit(X_train_s, y_train)
        y_pred = moc.predict(X_test_s)
    except Exception as e:
        print(f"  ERROR training {name}: {e}")
        continue

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_micro = f1_score(y_test, y_pred, average='micro', zero_division=0)
    prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    hl = hamming_loss(y_test, y_pred)

    try:
        if hasattr(moc.estimators_[0], 'predict_proba'):
            y_proba = np.column_stack([
                est.predict_proba(X_test_s)[:, 1] if hasattr(est, 'predict_proba')
                else est.decision_function(X_test_s)
                for est in moc.estimators_
            ])
            auc = roc_auc_score(y_test, y_proba, average='macro')
        else:
            auc = float('nan')
    except Exception:
        auc = float('nan')

    per_disease = {}
    for d_idx, d_name in enumerate(DISEASE_NAMES):
        f1_d = f1_score(y_test[:, d_idx], y_pred[:, d_idx], zero_division=0)
        p_d  = precision_score(y_test[:, d_idx], y_pred[:, d_idx], zero_division=0)
        r_d  = recall_score(y_test[:, d_idx], y_pred[:, d_idx], zero_division=0)
        per_disease[d_name] = {
            'f1':        float(f1_d),
            'precision': float(p_d),
            'recall':    float(r_d),
        }

    safe = name.replace(' ', '_').replace('(', '').replace(')', '')
    np.save(os.path.join(PREDS_DIR, f'{safe}_y_pred.npy'), y_pred.astype(np.int8))

    elapsed = time.time() - t0
    print(f"  {name:25s} {acc:10.4f} {f1_micro:10.4f} {f1_macro:10.4f} {hl:10.4f}  [{elapsed:.1f}s]")

    best_params = {}
    if param_grid and hasattr(moc.estimators_[0], 'best_params_'):
        best_params = moc.estimators_[0].best_params_

    results.append({
        'model': name,
        'accuracy': float(acc),
        'f1_macro': float(f1_macro),
        'f1_micro': float(f1_micro),
        'precision_macro': float(prec_macro),
        'recall_macro':    float(rec_macro),
        'hamming_loss':    float(hl),
        'auc_roc': float(auc) if not np.isnan(auc) else None,
        'best_params': best_params,
        'per_disease': per_disease,
        'time_seconds': float(elapsed),
    })

# ── Save ─────────────────────────────────────────────────────────────────────
output = {
    'dataset': f'CareAI April 2026 (n={labels_np.shape[0]:,}, high-confidence)',
    'n_train': int(len(train_idx)),
    'n_test': int(len(test_idx)),
    'n_features': int(X_train_s.shape[1]),
    'n_diseases': int(y_train.shape[1]),
    'split': 'MultilabelStratifiedShuffleSplit 80/20, seed=42',
    'results': results,
}

out_path = os.path.join(OUT_DIR, 'traditional_baselines_results.json')
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n[Done] Saved {out_path}")
print(f"\nNext: update paper tables with these numbers.")
