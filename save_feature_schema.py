#!/usr/bin/env python3
"""
Save Feature Schema from Training Data
=======================================
Run this ONCE to generate feature_schema.json alongside the trained models.
This permanently fixes the 182→203 feature dimension mismatch.

Usage:
    cd /path/to/HAN-implementation
    python save_feature_schema.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from HAN import MedicalGraphData
from HAN.feature_schema import save_schema, print_schema_diff

# ── Same parameters used during training (from Other_py/train.py) ──────────
PATH_RECORDS          = "data/filtered_patient_reports.csv"
PATH_SYMPTOM          = "data/test-disease-organ.csv"
SYMPTOM_FREQ_THRESHOLD = 0.08
PRUNE_PER_PATIENT      = 300
NNZ_THRESHOLD          = 80_000_000
SEED                   = 42

# Output paths — one schema per model save directory
SCHEMA_PATHS = [
    "models_saved/ruhunu_data_clustered/feature_schema.json",
    "models_saved/with_ruhunu_data/feature_schema.json",
]

print("=" * 60)
print("BUILDING FEATURE SCHEMA FROM TRAINING DATA")
print("=" * 60)
print(f"Records  : {PATH_RECORDS}")
print(f"Symptoms : {PATH_SYMPTOM}")
print(f"Threshold: {SYMPTOM_FREQ_THRESHOLD*100:.0f}%  |  Prune: {PRUNE_PER_PATIENT}")

data_loader = MedicalGraphData(
    path_records=PATH_RECORDS,
    path_symptom=PATH_SYMPTOM,
    symptom_freq_threshold=SYMPTOM_FREQ_THRESHOLD,
    prune_per_patient=PRUNE_PER_PATIENT,
    nnz_threshold=NNZ_THRESHOLD,
    seed=SEED
)

data_loader.load_data()
data_loader.build_labels_and_features()

print(f"\nTraining feature matrix shape: {data_loader.patient_feats.shape}")
print(f"  Symptoms : {len(data_loader.symptoms)}")
print(f"  Organs   : {len(data_loader.organs)}")
print(f"  Diseases : {len(data_loader.diseases)}")
print(f"  in_dim   : {data_loader.patient_feats.shape[1]}")

for path in SCHEMA_PATHS:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_schema(data_loader, path)

print("\n✅ Schema saved. Update predict_phase1_diagnosis.py to use:")
print("   from HAN.feature_schema import load_schema, align_features")
print("   schema = load_schema('models_saved/ruhunu_data_clustered/feature_schema.json')")
print("   aligned_feats = align_features(data_loader.patient_feats, data_loader, schema)")
