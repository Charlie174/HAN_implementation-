#!/usr/bin/env python3
"""
create_unseen_patients_v2.py — Generates 20 unseen patient records using
EXACT test names from the CareAI March 2026 training schema.

Seed: 2026 (different from training seed=42)
"""

import os
import json
import numpy as np
import pandas as pd

# ── Load training schema ──────────────────────────────────────────────────────
SCHEMA_PATH = 'output/careai_march/inductive_schema.json'
OUT_DIR     = 'data/unseen_patients'
os.makedirs(OUT_DIR, exist_ok=True)

with open(SCHEMA_PATH) as f:
    schema = json.load(f)

symptom_meta  = schema['symptom_meta']   # {test_name: {normal_low, normal_high, organ, ...}}
symptom_map   = schema['symptom_map']    # {test_name: idx}
disease_order = schema['disease_order']  # 9 diseases sorted

# All 108 tests with valid reference ranges
ALL_TESTS = [t for t in sorted(symptom_map.keys())
             if symptom_meta.get(t, {}).get('normal_low') is not None
             and symptom_meta.get(t, {}).get('normal_high') is not None]

print(f"Tests with reference ranges: {len(ALL_TESTS)}")

# Tests grouped by disease
DISEASE_TESTS = {}
for t in ALL_TESTS:
    d = symptom_meta[t].get('most_relevant_disease')
    if d:
        if d not in DISEASE_TESTS:
            DISEASE_TESTS[d] = []
        DISEASE_TESTS[d].append(t)

for d, tests in DISEASE_TESTS.items():
    print(f"  {d}: {len(tests)} tests")


def normal_value(test_name, rng):
    """Generate a normal value within [low, high]."""
    meta = symptom_meta[test_name]
    low  = meta['normal_low']
    high = meta['normal_high']
    # Use beta distribution for realistic distribution in-range
    mid  = (low + high) / 2.0
    std  = (high - low) / 6.0    # 3σ = range/2
    v    = rng.normal(mid, std)
    return float(np.clip(v, low + 0.01 * (high - low), high - 0.01 * (high - low)))


def abnormal_value(test_name, rng, factor=0.3, direction=None):
    """
    Generate an abnormal value OUTSIDE [low, high].
    factor: how far outside (0.3 = 30% beyond the boundary)
    direction: 'low', 'high', or None (random)
    """
    meta = symptom_meta[test_name]
    low  = meta['normal_low']
    high = meta['normal_high']
    rng_ = (high - low)

    if direction is None:
        direction = rng.choice(['low', 'high'])

    if direction == 'low':
        v = low - rng.uniform(factor * rng_, factor * rng_ * 2.5)
    else:
        v = high + rng.uniform(factor * rng_, factor * rng_ * 2.5)

    return float(max(0.0, v))


# ── Patient profile definitions ───────────────────────────────────────────────
# Each profile: {name, sex, age, organ_profile (list of disease strings), severity_factor}
PROFILES = [
    # 5 healthy
    ('UP_H001', 'Female', 28,  [],                                                    0.00, 0, 'healthy'),
    ('UP_H002', 'Male',   45,  [],                                                    0.00, 0, 'healthy'),
    ('UP_H003', 'Female', 62,  [],                                                    0.00, 0, 'healthy'),
    ('UP_H004', 'Male',   35,  [],                                                    0.00, 0, 'healthy'),
    ('UP_H005', 'Female', 55,  [],                                                    0.00, 0, 'healthy'),
    # 5 mild
    ('UP_M001', 'Male',   52,  ['CKD', 'Electrolyte_Imbalance'],                      0.25, 1, 'mild'),
    ('UP_M002', 'Female', 48,  ['Liver_Disease'],                                     0.20, 1, 'mild'),
    ('UP_M003', 'Male',   67,  ['Dyslipidemia'],                                      0.22, 1, 'mild'),
    ('UP_M004', 'Female', 41,  ['Diabetes'],                                          0.18, 1, 'mild'),
    ('UP_M005', 'Male',   59,  ['Thyroid_Disorder'],                                  0.15, 1, 'mild'),
    # 5 moderate
    ('UP_D001', 'Male',   64,  ['CKD', 'Dyslipidemia', 'Electrolyte_Imbalance'],      0.50, 2, 'moderate'),
    ('UP_D002', 'Female', 71,  ['Liver_Disease', 'Diabetes'],                         0.55, 2, 'moderate'),
    ('UP_D003', 'Male',   58,  ['Anemia', 'Hematology_Disorder', 'Infection_Inflammation'], 0.48, 2, 'moderate'),
    ('UP_D004', 'Female', 76,  ['CKD', 'Electrolyte_Imbalance', 'Thyroid_Disorder'],  0.52, 2, 'moderate'),
    ('UP_D005', 'Male',   55,  ['Dyslipidemia', 'Liver_Disease'],                     0.45, 2, 'moderate'),
    # 3 severe
    ('UP_S001', 'Male',   72,  ['CKD', 'Dyslipidemia', 'Electrolyte_Imbalance', 'Anemia'], 0.85, 3, 'severe'),
    ('UP_S002', 'Female', 68,  ['Liver_Disease', 'Diabetes', 'Infection_Inflammation'], 0.90, 3, 'severe'),
    ('UP_S003', 'Male',   80,  ['CKD', 'Liver_Disease', 'Thyroid_Disorder', 'Anemia', 'Electrolyte_Imbalance'], 0.95, 3, 'severe'),
    # 2 mixed
    ('UP_C001', 'Female', 63,  ['CKD', 'Diabetes', 'Dyslipidemia', 'Electrolyte_Imbalance'], 0.60, 2, 'mixed'),
    ('UP_C002', 'Male',   77,  ['Liver_Disease', 'Anemia', 'Hematology_Disorder', 'Thyroid_Disorder'], 0.70, 2, 'mixed'),
]

DOB_BASE_DATE = '03/06/2026'
SEED = 2026
rng = np.random.RandomState(SEED)


def generate_records(profiles, all_tests, symptom_meta, disease_tests, rng):
    rows = []
    for pid, sex, age, active_diseases, sev_factor, true_sev, profile_type in profiles:
        from datetime import datetime, timedelta
        dob_year = 2026 - int(age)
        dob = f'01/01/{dob_year}'

        for test in all_tests:
            meta = symptom_meta[test]
            lo   = meta['normal_low']
            hi   = meta['normal_high']
            if lo is None or hi is None:
                continue

            relevant_disease = meta.get('most_relevant_disease')
            is_active_test   = relevant_disease in active_diseases

            if is_active_test and sev_factor > 0:
                # Use higher abnormal severity for more severe patients
                factor = 0.25 + sev_factor * 0.5
                direction = rng.choice(['low', 'high'])
                value = abnormal_value(test, rng, factor=factor, direction=direction)
            else:
                value = normal_value(test, rng)

            rows.append({
                'patient_id':     pid,
                'report_date':    DOB_BASE_DATE + ' 00:00',
                'test_name':      test,
                'test_value':     round(value, 4),
                'date_of_birth':  dob + ' 0:00',
                'age_at_report':  float(age),
                'sex':            sex,
                'is_foreign':     0,
                'profile_type':   profile_type,
            })
    return pd.DataFrame(rows)


# ── Generate ──────────────────────────────────────────────────────────────────
print(f"\nGenerating records with seed={SEED}...")
df_records = generate_records(PROFILES, ALL_TESTS, symptom_meta, DISEASE_TESTS, rng)
print(f"Records generated: {len(df_records)} rows, {df_records['patient_id'].nunique()} patients")
print(f"Tests per patient: {len(df_records) // df_records['patient_id'].nunique()}")

records_out = os.path.join(OUT_DIR, 'unseen_patient_records_v2.csv')
df_records.to_csv(records_out, index=False)
print(f"Saved → {records_out}")

# ── Ground truth ──────────────────────────────────────────────────────────────
gt_rows = []
disease_col_order = ['Anemia', 'CKD', 'Diabetes', 'Dyslipidemia',
                     'Electrolyte_Imbalance', 'Hematology_Disorder',
                     'Infection_Inflammation', 'Liver_Disease', 'Thyroid_Disorder']

for pid, sex, age, active_diseases, sev_factor, true_sev, profile_type in PROFILES:
    row = {'patient_id': pid, 'profile_type': profile_type}
    for d in disease_col_order:
        row[d] = 1 if d in active_diseases else 0
    row['notes'] = ', '.join(active_diseases) if active_diseases else 'none'
    gt_rows.append(row)

df_gt = pd.DataFrame(gt_rows)
gt_out = os.path.join(OUT_DIR, 'unseen_disease_ground_truth_v2.csv')
df_gt.to_csv(gt_out, index=False)
print(f"Saved → {gt_out}")

print("\nGround truth summary:")
for d in disease_col_order:
    n_pos = int(df_gt[d].sum())
    print(f"  {d:<30}: {n_pos} positive patients")
