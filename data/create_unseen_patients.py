#!/usr/bin/env python3
"""
Create Unseen Patient Dataset for CareAI HAN++ Validation
==========================================================
Generates 20 truly unseen patients with known ground-truth health profiles.

Seed = 2026  (training used seed = 42 → no overlap)
Output  → data/unseen_patients/   (NEVER touches train/val/test data)

Patient breakdown:
  Healthy  : 5  (baseline – all organs normal)
  Mild     : 5  (1 organ mildly affected)
  Moderate : 5  (2 organs moderately affected)
  Severe   : 3  (3+ organs critically affected)
  Mixed    : 2  (realistic multi-system disease)
"""

import os, sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(2026)          # Different from training seed (42) → truly unseen

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unseen_patients")
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 70)
print("GENERATING UNSEEN PATIENT DATASET  (seed=2026)")
print("=" * 70)
print(f"Output : {OUT_DIR}\n")

# ─────────────────────────────────────────────────────────────
# CLINICAL TESTS  (matches MEDICAL_TESTS vocabulary used in
#                  the existing New_patient.ipynb pipeline)
# ─────────────────────────────────────────────────────────────
TESTS = {
    "Haemoglobin Absolute Value":    {"normal": (12.5, 16.0), "unit": "g/dL",    "organ": "blood"},
    "RBC Absolute Value":            {"normal": (4.11, 5.51), "unit": "10^12/L", "organ": "blood"},
    "WBC Absolute Value":            {"normal": (4.0,  11.0), "unit": "10^9/L",  "organ": "immune system"},
    "Platelet Count Absolute Value": {"normal": (150,  450),  "unit": "10^9/L",  "organ": "blood"},
    "MCV Result":                    {"normal": (80,   100),  "unit": "fL",      "organ": "blood"},
    "MCH Result":                    {"normal": (27,   33),   "unit": "pg",      "organ": "blood"},
    "MCHC Result":                   {"normal": (32,   36),   "unit": "g/dL",    "organ": "blood"},
    "PCV / HCT Result":              {"normal": (36,   46),   "unit": "%",       "organ": "blood"},
    # ── Kidney
    "Serum_Creatinine_Result":       {"normal": (0.65, 1.20), "unit": "mg/dL",   "organ": "kidney"},
    "eGFR Result":                   {"normal": (90,   120),  "unit": "mL/min",  "organ": "kidney"},
    "Blood Urea Result":             {"normal": (20,   40),   "unit": "mg/dL",   "organ": "kidney"},
    "Serum - Potassium":             {"normal": (3.5,  5.1),  "unit": "mmol/L",  "organ": "kidney"},
    "Serum - Sodium":                {"normal": (135,  145),  "unit": "mmol/L",  "organ": "kidney"},
    "Serum - Chloride":              {"normal": (98,   106),  "unit": "mmol/L",  "organ": "kidney"},
    "Uric Acid Result":              {"normal": (3.5,  7.2),  "unit": "mg/dL",   "organ": "kidney"},
    # ── Liver
    "SGPT (ALT) Result":             {"normal": (7,    56),   "unit": "U/L",     "organ": "liver"},
    "SGOT (AST) Result":             {"normal": (10,   40),   "unit": "U/L",     "organ": "liver"},
    "Alkaline Phosphatase Result":   {"normal": (44,   147),  "unit": "U/L",     "organ": "liver"},
    "Serum Bilirubin (Total) Result":{"normal": (0.2,  1.2),  "unit": "mg/dL",   "organ": "liver"},
    "Serum Albumin Result":          {"normal": (3.5,  5.0),  "unit": "g/dL",    "organ": "liver"},
    "Total Protein Result":          {"normal": (6.3,  8.2),  "unit": "g/dL",    "organ": "liver"},
    # ── Cardiovascular / Lipid
    "Total Cholesterol":             {"normal": (100,  200),  "unit": "mg/dL",   "organ": "cardiovascular system"},
    "LDL-Cholesterol":               {"normal": (50,   100),  "unit": "mg/dL",   "organ": "cardiovascular system"},
    "HDL Cholesterol":               {"normal": (45,   80),   "unit": "mg/dL",   "organ": "cardiovascular system"},
    "Triglycerides Result":          {"normal": (50,   150),  "unit": "mg/dL",   "organ": "cardiovascular system"},
    # ── Pancreas / Diabetes
    "HbA1c Result":                  {"normal": (4.0,  5.6),  "unit": "%",       "organ": "pancreas"},
    "Fasting Blood Glucose":         {"normal": (70,   100),  "unit": "mg/dL",   "organ": "pancreas"},
    "Random Blood Glucose":          {"normal": (70,   140),  "unit": "mg/dL",   "organ": "pancreas"},
    # ── Thyroid
    "TSH":                           {"normal": (0.4,  4.0),  "unit": "mIU/L",   "organ": "thyroid"},
    "Free T4":                       {"normal": (0.8,  1.8),  "unit": "ng/dL",   "organ": "thyroid"},
    # ── Inflammatory markers
    "CRP Result":                    {"normal": (0,    5),    "unit": "mg/L",    "organ": "immune system"},
    "ESR Result":                    {"normal": (0,    20),   "unit": "mm/hr",   "organ": "immune system"},
}

# ─────────────────────────────────────────────────────────────
# PATIENT MANIFEST  (20 unseen patients with ground truth)
# ─────────────────────────────────────────────────────────────
#  (pid, profile, age, sex, affected_organs, severity_factor)
PATIENTS = [
    # ── HEALTHY
    ("UP_H001", "healthy",  28, "Female", [],                                           0.00),
    ("UP_H002", "healthy",  45, "Male",   [],                                           0.00),
    ("UP_H003", "healthy",  62, "Female", [],                                           0.00),
    ("UP_H004", "healthy",  35, "Male",   [],                                           0.00),
    ("UP_H005", "healthy",  55, "Female", [],                                           0.00),
    # ── MILD
    ("UP_M001", "mild",     52, "Male",   ["kidney"],                                   0.25),
    ("UP_M002", "mild",     48, "Female", ["liver"],                                    0.20),
    ("UP_M003", "mild",     67, "Male",   ["cardiovascular system"],                    0.22),
    ("UP_M004", "mild",     41, "Female", ["pancreas"],                                 0.18),
    ("UP_M005", "mild",     59, "Male",   ["thyroid"],                                  0.15),
    # ── MODERATE
    ("UP_D001", "moderate", 64, "Male",   ["kidney", "cardiovascular system"],          0.50),
    ("UP_D002", "moderate", 71, "Female", ["liver", "pancreas"],                        0.55),
    ("UP_D003", "moderate", 58, "Male",   ["blood", "immune system"],                   0.48),
    ("UP_D004", "moderate", 76, "Female", ["kidney", "thyroid"],                        0.52),
    ("UP_D005", "moderate", 55, "Male",   ["liver", "cardiovascular system"],           0.45),
    # ── SEVERE
    ("UP_S001", "severe",   72, "Male",   ["kidney", "cardiovascular system", "blood"], 0.85),
    ("UP_S002", "severe",   68, "Female", ["liver", "pancreas", "immune system"],       0.90),
    ("UP_S003", "severe",   80, "Male",   ["kidney", "liver", "thyroid", "blood"],      0.95),
    # ── MIXED  (complex real-world presentation)
    ("UP_C001", "mixed",    63, "Female", ["pancreas", "cardiovascular system", "kidney"], 0.60),
    ("UP_C002", "mixed",    77, "Male",   ["liver", "blood", "thyroid"],                0.70),
]

# ─────────────────────────────────────────────────────────────
# VALUE GENERATION
# ─────────────────────────────────────────────────────────────
def generate_value(test_name, info, affected_organs, severity):
    lo, hi = info["normal"]
    rng     = hi - lo
    organ   = info["organ"]

    if organ not in affected_organs or severity == 0:
        # Normal range: small natural variation
        return round(float(np.random.uniform(lo + 0.1*rng, hi - 0.1*rng)), 2)

    # Abnormal: deviate by severity × range
    direction = "high" if np.random.random() < 0.55 else "low"
    delta     = severity * rng * np.random.uniform(0.8, 1.3)

    # Organ-specific physiological tweaks
    if test_name == "eGFR Result":
        value = max(5.0, lo - severity * 80)
    elif test_name == "HbA1c Result":
        value = hi + severity * 6
    elif test_name == "HDL Cholesterol":
        value = max(15.0, lo - severity * 25)
    elif test_name == "Serum Albumin Result":
        value = max(1.5, lo - severity * 2)
    elif direction == "high":
        value = hi + delta
    else:
        value = lo - delta

    return round(max(0.01, value), 2)


# ─────────────────────────────────────────────────────────────
# BUILD RECORDS DATAFRAME
# ─────────────────────────────────────────────────────────────
records, gt_rows = [], []
report_date = datetime(2026, 3, 6)

for (pid, profile, age, sex, affected, severity) in PATIENTS:
    dob = report_date - timedelta(days=age * 365.25)
    for tname, tinfo in TESTS.items():
        val = generate_value(tname, tinfo, affected, severity)
        records.append({
            "patient_id":    pid,
            "report_date":   report_date.strftime("%m/%d/%Y %H:%M"),
            "test_name":     tname,
            "test_value":    val,
            "date_of_birth": dob.strftime("%m/%d/%Y 0:00"),
            "age_at_report": float(age),
            "sex":           sex,
            "is_foreign":    0,
            "profile_type":  profile,        # kept in records for convenience
        })
    gt_rows.append({
        "patient_id":      pid,
        "profile_type":    profile,
        "age":             age,
        "sex":             sex,
        "affected_organs": "|".join(affected) if affected else "none",
        "severity_factor": severity,
        # Ground-truth severity class (0-3)
        "true_severity":   {"healthy": 0, "mild": 1, "moderate": 2,
                            "severe": 3, "mixed": 2}[profile],
    })

df_records = pd.DataFrame(records)
df_gt      = pd.DataFrame(gt_rows)

# ─────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────
records_path = os.path.join(OUT_DIR, "unseen_patient_records.csv")
gt_path      = os.path.join(OUT_DIR, "unseen_patient_ground_truth.csv")

df_records.to_csv(records_path, index=False)
df_gt.to_csv(gt_path, index=False)

print(f"{'Patient records':<22}: {records_path}")
print(f"{'Ground truth labels':<22}: {gt_path}")
print(f"\nDataset statistics:")
print(f"  Patients          : {df_records['patient_id'].nunique()}")
print(f"  Records           : {len(df_records)}")
print(f"  Tests per patient : {len(TESTS)}")
print(f"  Report date       : {report_date.strftime('%Y-%m-%d')}")
print(f"\nProfile breakdown:")
for p in ["healthy", "mild", "moderate", "severe", "mixed"]:
    n = df_gt[df_gt["profile_type"] == p].shape[0]
    print(f"  {p.capitalize():<10}: {n} patients")

print("\n✅  Done — data/unseen_patients/ is isolated from train/val/test splits")
