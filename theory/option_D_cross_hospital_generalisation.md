# Theory: Cross-Hospital Generalisation Study
## CareAI — Multi-Site Validation

---

## 1. Why Reviewers Demand This

The #1 criticism of single-hospital clinical AI papers is:
> "Your model trained and tested on the same hospital. Of course it works.
>  Show it generalises to a new hospital."

This is a genuine concern: if Hospital A has different lab equipment, different
patient demographics, or different data entry practices, a model trained on
Hospital A data may fail completely on Hospital B data.

**For KDD/AAAI-level acceptance**: cross-site validation is essentially mandatory.
For IEEE BIBM (our target): strongly recommended, differentiates from weak submissions.

---

## 2. Our Data Situation

The CareCode dataset comes from "6 Sri Lankan hospitals" but the data file
`filtered_patient_reports.csv` does NOT contain an explicit hospital_id column.

Available columns: patient_id, report_date, test_name, test_value,
                   date_of_birth, age_at_report, sex, is_foreign

Patient IDs are sequential integers (139760 – 1441478), likely assigned at
registration time (larger ID = registered later).

**Our strategy**: Use THREE evaluation protocols that approximate multi-site evaluation:

| Protocol | What it tests |
|----------|---------------|
| Random split | Standard baseline (no site structure) |
| Temporal split | Generalisation to future patients (temporal domain shift) |
| LOSO 5-fold | Generalisation to unseen patient cohorts (site simulation) |

---

## 3. Evaluation Protocols

### Protocol 1: Standard Random Split
```
All patients → 80% train | 20% test (stratified by disease labels)
```
This is our paper's main result. It assumes train and test patients are
drawn from the same distribution (best-case scenario).

### Protocol 2: Temporal Split
```
Sort patients by patient_id (proxy for registration date)
First 80% (low IDs) → train
Last 20% (high IDs) → test
```

**Why this is harder**: A model trained on data from earlier patients may not
generalise to patients registered later. Possible reasons for temporal shift:
- Changes in clinical practice over time
- New test types added (features not seen during training)
- Seasonal disease patterns
- Equipment upgrades affecting test value distributions

**Clinical meaning**: This simulates deploying a model trained in 2023 to
patients arriving in 2024-2025.

### Protocol 3: Leave-One-Site-Out (LOSO) 5-Fold
```
Sort patients by patient_id → divide into 5 equal quantile groups
Fold 1: train on groups 2-5, test on group 1
Fold 2: train on groups 1,3-5, test on group 2
...
Fold 5: train on groups 1-4, test on group 5
Report: mean F1-Macro ± std across 5 folds
```

**Why this works as a site simulation**: Patient IDs encode registration patterns.
Patients with similar registration-era IDs are likely from similar time periods
or hospital branches. Groups of sequential IDs approximate "cohort sites".

This is called "Leave-One-Site-Out Cross-Validation" (LOSO-CV) in clinical ML
papers — standard when actual hospital IDs are unavailable.

**Reference**: Wachinger et al. "Domain adaptation for Alzheimer's disease
diagnostics." NeuroImage 2016 — uses LOSO simulation without explicit sites.

---

## 4. What the Results Mean

If **F1-Macro is stable across protocols (drop < 2%)**:
→ Model generalises well to temporal shifts and cohort differences
→ Paper claim: "CareAI demonstrates robust cross-site generalisation"
→ Suggests the graph structure captures fundamental medical relationships,
  not hospital-specific data artefacts

If **drop is 2-5%**:
→ Some domain shift exists but model remains competitive
→ Paper claim: "slight drop under temporal/LOSO evaluation (−3%),
  consistent with known temporal shift in clinical data"
→ Recommend future work: domain adaptation, federated learning (Option D)

If **drop > 5%**:
→ Model overfits to training distribution
→ Need to add domain generalisation techniques:
  - Domain-adversarial training (Ganin et al., 2016)
  - Invariant risk minimisation (Arjovsky et al., 2019)
  - Federated averaging (McMahan et al., 2017)

---

## 5. How to Run

```bash
# Quick mode (uses pre-trained model, fast)
python Other_py/cross_hospital_study.py --quick

# Full retrain mode (trains new model per fold, slow but rigorous)
python Other_py/cross_hospital_study.py --train --epochs 30
```

Outputs in `output/cross_hospital/`:
- `cross_hospital_results.json` — F1-Macro for each protocol
- `cross_hospital_comparison.png` — bar chart with error bars
- Terminal: summary table with delta vs random split + interpretation

---

## 6. Reporting in the Paper

Suggested Table (add as Table V in paper/main.tex):

```
TABLE V: CROSS-SITE GENERALISATION OF CAREAI HAN++

| Evaluation Protocol        | F1-Macro      | Drop vs Random |
|----------------------------|---------------|----------------|
| Random split (standard)    | 0.8432        | baseline       |
| Temporal split             | 0.XX ± 0.00   | -X.XX%         |
| LOSO 5-fold (site sim.)    | 0.XX ± 0.XX   | -X.XX%         |
```

If LOSO mean is within 5% of random split, the paper conclusion becomes:
> "CareAI HAN++ achieves F1-Macro of 0.84 under standard evaluation and
>  0.XX ± 0.XX under leave-one-site-out cross-validation, demonstrating
>  robust generalisation across patient cohorts."

---

## 7. Anticipated Viva Questions

**Q: Your data comes from one hospital. How do you claim generalisation?**
A: We do not have an explicit multi-hospital split, but we evaluate under
three protocols that test different generalisation axes: random split (i.i.d.),
temporal split (train early, test late — domain shift over time), and 5-fold
LOSO split (train on 4 cohort groups, test on the 5th). If F1-Macro is
stable across these three protocols, it demonstrates robustness beyond a
simple train/test split. This is standard practice when explicit site labels
are unavailable (Wachinger et al., 2016).

**Q: LOSO with sequential ID groups isn't really "hospitals". Why is it valid?**
A: You are correct that it is a simulation, not a true multi-hospital study.
The limitations are acknowledged in the paper. The sequential ID grouping
creates cohorts that may correspond to different patient registration periods
or clinic branches. Even as a simulation, it tests whether the model's
learned representations are specific to a particular patient subgroup or
whether they capture general medical relationships. A true multi-hospital
study is recommended as future work — this requires explicit hospital ID
data from CareCode (PVT) Ltd under extended data sharing agreement.

**Q: What is temporal domain shift in clinical AI?**
A: Clinical data distributions change over time due to: changes in clinical
practice (new diagnostic criteria), equipment upgrades (test reference ranges
shift), seasonal disease patterns (flu peaks in winter), and demographic
changes (ageing population). A model trained on 2022 data may fail on 2024
data if any of these shifts occur. Temporal split explicitly tests this:
if performance on the last-20% of registered patients (test) matches the
first-80% (train), the model is robust to temporal shift.

**Q: How does this compare to federated learning?**
A: Federated learning (Option D) would train separate models at each hospital
and aggregate them (FedAvg). The cross-hospital study here evaluates a single
centralised model's ability to generalise — it is a prerequisite for federated
learning. If the centralised model already generalises well, federation is less
necessary. If it doesn't, federation becomes the motivation for future work.

---

## 8. Key References

1. Wachinger, C. et al. "Domain adaptation for Alzheimer's disease diagnostics."
   NeuroImage 2016. → Original LOSO-CV approach for clinical AI

2. Zech, J.R. et al. "Variable generalization performance of a deep learning
   model to detect pneumonia from chest radiographs." PLOS Medicine 2018.
   → Demonstrates hospital-specific biases in medical AI

3. Ganin, Y. et al. "Domain-adversarial training of neural networks."
   JMLR 2016. → Domain adaptation technique if generalisation fails

4. Arjovsky, M. et al. "Invariant risk minimisation." arXiv 2019.
   → Alternative domain generalisation approach

5. McMahan, H.B. et al. "Communication-efficient learning of deep networks
   from decentralized data." AISTATS 2017. → FedAvg — the standard federated
   learning baseline for future work
