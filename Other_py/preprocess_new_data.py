"""
Preprocessing script for new CareAI dataset (Data_4_16).

Produces files expected by train_careai_march.ipynb:
  data/dataset_careai_March/merged_coop_ruhunu_patient_data.csv
  data/dataset_careai_March/unique_test_data_finalized.csv
  data/dataset_careai_March/processed/updated_patient_ground_truth_v2.csv

Run from project root:
    python Other_py/preprocess_new_data.py
"""

import os
import shutil
import numpy as np
import pandas as pd

BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC      = os.path.join(BASE, "Data_4_16")
DATA_DIR = os.path.join(BASE, "data", "dataset_careai_april")
PROC_DIR = os.path.join(DATA_DIR, "processed")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)

# ── Finalized disease list (9 diseases confirmed by user) ─────────────────────
# Maps long-format disease_name → output column name
DISEASE_MAP = {
    "Anemia":                  "Anemia",
    "CKD":                     "CKD",
    "Diabetes":                "Diabetes",
    "Dyslipidemia":            "Dyslipidemia",
    "Electrolyte Imbalance":   "Electrolyte_Imbalance",
    "Hematology Disorder":     "Hematology_Disorder",
    "Infection Inflammation":  "Infection_Inflammation",
    "Liver Disease":           "Liver_Disease",
    "Thyroid Disorder":        "Thyroid_Disorder",
}
TARGET_DISEASES = sorted(DISEASE_MAP.values())  # alphabetical


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Merge coop + ruhunu patient records
# ══════════════════════════════════════════════════════════════════════════════
def step1_merge_records():
    out_path = os.path.join(DATA_DIR, "merged_coop_ruhunu_patient_data.csv")
    if os.path.exists(out_path):
        print(f"[Step 1] Already exists — skipping: {out_path}")
        return out_path

    print("[Step 1] Merging coop + ruhunu patient records...")
    coop   = pd.read_csv(os.path.join(SRC, "cleaned_patient_data_no_duplicates_coop.csv"),
                         low_memory=False)
    ruhunu = pd.read_csv(os.path.join(SRC, "cleaned_patient_data_no_duplicates_ruhunu.csv"),
                         low_memory=False)

    print(f"  Coop   : {len(coop):>7,} rows, {coop['patient_id'].nunique():,} patients")
    print(f"  Ruhunu : {len(ruhunu):>7,} rows, {ruhunu['patient_id'].nunique():,} patients")

    combined = pd.concat([coop, ruhunu], ignore_index=True)
    print(f"  Combined (before clean): {len(combined):,} rows")

    # ── Clean outlier values ─────────────────────────────────────────────────
    # negative values (except valid negatives like temp below 0): 121 cases
    # extreme negatives (< -1000): 4 cases — data entry errors
    # extreme positives (> 10000): unlikely for clinical lab tests
    n_before = len(combined)
    invalid  = (combined["value"] < -0.1) | (combined["value"] > 10_000)
    combined.loc[invalid, "value"] = np.nan
    n_invalid = invalid.sum()
    print(f"  Outlier values set to NaN: {n_invalid:,} "
          f"({100*n_invalid/n_before:.2f}% of rows)")

    # Drop rows where value is NaN (cannot use for feature engineering)
    combined = combined.dropna(subset=["value"])
    print(f"  After dropping NaN values: {len(combined):,} rows, "
          f"{combined['patient_id'].nunique():,} patients")

    # Keep only columns the training pipeline uses
    keep = ["patient_id", "mapped_test_name", "value", "record_date"]
    combined = combined[[c for c in keep if c in combined.columns]]
    combined.to_csv(out_path, index=False)
    print(f"  Saved → {out_path}")
    return out_path


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Convert long-format GT → wide-format binary labels
# ══════════════════════════════════════════════════════════════════════════════
def step2_build_labels():
    out_path = os.path.join(PROC_DIR, "updated_patient_ground_truth_v2.csv")
    if os.path.exists(out_path):
        print(f"[Step 2] Already exists — skipping: {out_path}")
        return out_path

    print("[Step 2] Building wide-format binary disease labels...")
    df_long = pd.read_csv(os.path.join(SRC, "patient_disease_ground_truth_long.csv"))
    print(f"  Long format: {len(df_long):,} rows, "
          f"{df_long['patient_id'].nunique():,} unique patients")

    # Normalise disease names → target column names
    df_long["disease_col"] = df_long["disease_name"].map(DISEASE_MAP)
    unmapped = df_long[df_long["disease_col"].isna()]["disease_name"].dropna().unique()
    if len(unmapped):
        print(f"  Skipped diseases (not in target list): {sorted(str(x) for x in unmapped)}")

    # Keep only rows for target diseases
    df_target = df_long[df_long["disease_col"].notna()].copy()
    print(f"  Rows for target diseases: {len(df_target):,}")

    # Build binary wide format — patient × disease = 1 if present (any confidence)
    df_target["label"] = 1
    df_wide = (
        df_target
        .groupby(["patient_id", "disease_col"])["label"]
        .max()
        .unstack(fill_value=0)
        .reset_index()
    )

    # Ensure all 9 disease columns exist (fill 0 for diseases with no cases)
    for d in TARGET_DISEASES:
        if d not in df_wide.columns:
            df_wide[d] = 0

    # Reorder columns: patient_id first, then diseases alphabetically
    df_wide = df_wide[["patient_id"] + TARGET_DISEASES]
    df_wide["patient_id"] = df_wide["patient_id"].astype(int)

    df_wide.to_csv(out_path, index=False)
    print(f"  Saved → {out_path}")
    print(f"  Shape: {df_wide.shape}  (patients × diseases+1)")
    print()
    print(f"  {'Disease':<28} {'Positives':>10}  {'Prevalence':>10}")
    print("  " + "-" * 52)
    total = len(df_wide)
    for d in TARGET_DISEASES:
        n   = int(df_wide[d].sum())
        pct = 100 * n / total
        print(f"  {d:<28} {n:>10,}  {pct:>9.1f}%")
    print(f"\n  Total labeled patients: {total:,}")
    return out_path


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Copy test reference to DATA_DIR
# ══════════════════════════════════════════════════════════════════════════════
def step3_copy_test_reference():
    dst = os.path.join(DATA_DIR, "unique_test_data_finalized.csv")
    src = os.path.join(SRC, "unique_test_data_finalized.csv")
    if os.path.exists(dst):
        print(f"[Step 3] Already exists — skipping: {dst}")
        return dst
    shutil.copy2(src, dst)
    print(f"[Step 3] Copied test reference → {dst}")
    return dst


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Validation report
# ══════════════════════════════════════════════════════════════════════════════
def step4_validate():
    print("\n[Step 4] Validation report...")

    merged_path = os.path.join(DATA_DIR, "merged_coop_ruhunu_patient_data.csv")
    labels_path = os.path.join(PROC_DIR, "updated_patient_ground_truth_v2.csv")

    df_rec  = pd.read_csv(merged_path, usecols=["patient_id", "mapped_test_name"])
    df_lab  = pd.read_csv(labels_path)

    rec_pids = set(df_rec["patient_id"].unique())
    lab_pids = set(df_lab["patient_id"].unique())
    both     = rec_pids & lab_pids

    print(f"  Patients with records:    {len(rec_pids):,}")
    print(f"  Patients with labels:     {len(lab_pids):,}")
    print(f"  Patients in BOTH:         {len(both):,}  ← these will be trained on")
    print(f"  Patients with records only (unlabeled): {len(rec_pids - lab_pids):,}")
    print(f"  Patients with labels only (no records): {len(lab_pids - rec_pids):,}")

    n_tests = df_rec["mapped_test_name"].nunique()
    print(f"\n  Unique test names in records: {n_tests}")

    # Check any labels patient missing from records
    missing_in_rec = lab_pids - rec_pids
    if missing_in_rec:
        print(f"\n  WARNING: {len(missing_in_rec)} labeled patients have no records "
              f"— they will get all-zero features.")
    else:
        print(f"\n  All labeled patients have records.")

    # Multi-label stats for training subset
    df_train = df_lab[df_lab["patient_id"].isin(both)]
    label_sums = df_train[TARGET_DISEASES].sum(axis=1)
    print(f"\n  Multi-label distribution (labeled + has records):")
    for n in range(0, 6):
        cnt = (label_sums == n).sum()
        print(f"    {n} diseases: {cnt:,} patients ({100*cnt/len(df_train):.1f}%)")

    all_zero = (label_sums == 0).sum()
    if all_zero > 0:
        print(f"\n  WARNING: {all_zero:,} labeled patients have ALL-ZERO labels "
              f"(no target disease matched). These are still used but contribute "
              f"no positive signal.")

    print(f"\n  Ready for training. Run train_careai_march.ipynb.")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("CareAI Data Preprocessing — Data_4_16")
    print("=" * 60)
    step1_merge_records()
    print()
    step2_build_labels()
    print()
    step3_copy_test_reference()
    step4_validate()
    print("\nDone.")
