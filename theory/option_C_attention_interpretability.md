# Theory: Attention Interpretability
## CareAI — Two-Level Explanation of Predictions

---

## 1. Why Interpretability Matters

A "black box" model predicting "Kidney: SEVERE" is useless in a hospital.
Doctors need to know WHY.

Without interpretability:
- Physicians reject AI tools (fear of liability for unexplained decisions)
- Regulatory bodies (FDA, MHRA) require explanation for AI medical devices
- Patients have a legal right (GDPR Article 22) to explanation of automated decisions

With attention interpretability, CareAI can say:
> "Patient 1042 was predicted SEVERE kidney because:
>  - The model primarily used the P-O-P meta-path (64% weight)
>    meaning it compared this patient to others with similar organ involvement
>  - The top 3 most similar patients in training all had confirmed SEVERE CKD
>  - The creatinine and GFR test values drove the node-level similarity"

This is clinically actionable and legally defensible.

---

## 2. Two-Level Attention in HAN++

HAN++ has TWO attention mechanisms, each interpretable at a different level:

### Level 1 — Semantic Attention (β): WHICH information source?

After our patient-conditioned extension, β is [N, K]:
- β_{i,k} = how much patient i relied on meta-path k
- Larger β_{i,PDP} → disease co-occurrence neighbours drove the prediction
- Larger β_{i,POP} → organ involvement neighbours drove the prediction
- Larger β_{i,PSP} → lab test value neighbours drove the prediction

**Clinical interpretation**:
| High β on | Meaning |
|-----------|---------|
| P-D-P | Patient's diagnosis is driven by disease comorbidity patterns |
| P-O-P | Prediction is based on sharing organ disease with similar patients |
| P-S-P | Prediction is based on lab value similarity to known cases |

### Level 2 — Node Attention (α): WHICH specific patients?

Within each meta-path, α_{ij} = attention weight from patient i to neighbour j:
- High α_{ij} → "patient j strongly influenced patient i's prediction"
- These are the MOST SIMILAR PATIENTS in the training set
- Show their confirmed diagnoses as evidence for the current prediction

**Clinical interpretation**:
> "Your prediction is SEVERE because the 3 most similar patients (α > 0.15)
>  all had confirmed severe kidney disease."

---

## 3. Mathematical Basis

### Semantic attention weights (after our patient-conditioned extension):
```
q_i  = W_q · h_i                        [patient-specific query]
e_ik = q_i^T · tanh(W_sem · z_i^Φk)    [patient-metapath relevance score]
β_ik = exp(e_ik) / Σ_k' exp(e_ik')     [softmax over K meta-paths]
```

β_ik ∈ (0,1) and Σ_k β_ik = 1.
This is a probability distribution over meta-paths — directly interpretable
as "what fraction of the prediction comes from each relationship type".

### Node-level attention weights:
```
e_ij = LeakyReLU( a_l^T · h_i + a_r^T · h_j )   [compatibility score]
α_ij = exp(e_ij) / Σ_{j' ∈ N(i)} exp(e_ij')      [softmax over neighbours]
```

α_ij ∈ (0,1) and Σ_j α_ij = 1 (over patient i's neighbourhood).
High α_ij → patient j is the most compatible "reference case" for patient i.

---

## 4. What the Plots Show

### metapath_heatmap.png
- Rows = patients (up to 50 shown), columns = meta-paths
- Colour intensity = attention weight (dark red = relied heavily on this path)
- If all rows are identical → the model is ignoring patient-specific conditioning
- Diverse row colours → the model learned meaningful patient-specific preferences

### metapath_distribution.png
- Box plot: distribution of β across all patients per meta-path
- If std ≈ 0 → global attention (bad, suggests our conditioning isn't helping)
- If std > 0.05 → patients genuinely differ in meta-path preference (good)
- Mean bar: which meta-path is generally most important

### top_neighbours_pXXX.png
- For a specific query patient, horizontal bar chart of top-10 neighbours
- Sorted by α (highest first)
- Shows which patients in the training set most influenced this prediction

### patient_explanations/explanation_pXXX.txt
- Human-readable text explanation: meta-path percentages + top neighbours

---

## 5. How to Run

```python
from HAN.interpretability import run_interpretability_analysis

# After loading model and data:
beta = run_interpretability_analysis(
    model=model,
    features=patient_feats,      # [N, in_dim]
    neighbor_dicts=neighs,       # dict of meta-path neighbours
    patient_ids=data.patient_ids,
    predictions=predictions,     # [N, O] from make_predictions
    organ_names=data.organs,
    output_dir='output/interpretability',
    sample_patients=10           # explain 10 random patients
)
```

Outputs:
- `output/interpretability/metapath_heatmap.png`
- `output/interpretability/metapath_distribution.png`
- `output/interpretability/beta_per_patient.npy`
- `output/interpretability/patient_explanations/explanation_pXXX.txt` (×10)
- `output/interpretability/top_neighbours_pXXX.png` (×10)

---

## 6. Limitations

1. **Attention ≠ causality**: High attention weight means "the model used this
   information heavily" but NOT "this caused the disease". Correlation, not causation.

2. **Node-level alpha is averaged over attention heads**: We average the 4-head
   attention weights. Different heads may attend to different aspects of similarity.
   A future extension: analyse each head separately (head specialisation).

3. **Explanations are post-hoc**: We are explaining what a trained model does,
   not verifying that it reasons correctly. A model can have high attention on
   a patient and still be wrong.

---

## 7. Anticipated Viva Questions

**Q: How does your model explain its predictions?**
A: HAN++ has two-level attention. The semantic-level (beta) shows which meta-path
(disease co-occurrence, organ similarity, lab test similarity) the model relied
on for each patient — this is our novel patient-conditioned attention. The node-level
(alpha) shows which specific training patients most influenced the prediction,
allowing physicians to see similar confirmed cases as evidence.

**Q: Is attention a valid explanation method?**
A: Attention is a useful approximation but not a perfect explanation.
Research (Jain & Wallace, 2019; Wiegreffe & Pinter, 2019) shows attention
weights correlate with but don't perfectly predict feature importance.
However, in our case the semantic attention (which meta-path to trust)
is structurally meaningful — P-D-P vs P-O-P are genuinely different
information sources, not just feature dimensions. So beta is more
interpretable than typical token-level attention in NLP.

**Q: How does this help a physician in practice?**
A: A physician seeing "Kidney: SEVERE, 64% P-O-P attention" can check:
(1) Was the P-O-P meta-path appropriate? (Did similar organ-disease patients
really have severe CKD?) and (2) Are the top alpha-weighted neighbours
plausible comparisons? This gives the physician a mechanism to sanity-check
the AI's reasoning, not just accept or reject a black-box number.

**Q: How does this compare to LIME or SHAP for explainability?**
A: LIME/SHAP are model-agnostic post-hoc methods that perturb the input and
measure output change. They explain WHICH INPUT FEATURES matter.
Our attention-based explanation explains WHICH RELATIONAL STRUCTURE
(meta-paths) and WHICH PATIENTS matter — this is more clinically natural
for a graph-based model where relationships ARE the representation.
The two approaches are complementary: SHAP could be added for feature-level
explanations on top of our attention-level explanations.

---

## 8. Key References

1. Wang, X. et al. "Heterogeneous Graph Attention Network." WWW 2019.
   → Original HAN; semantic attention is interpretable by design

2. Vaswani, A. et al. "Attention is All You Need." NeurIPS 2017.
   → Scaled dot-product attention; attention as explanation mechanism

3. Jain, S. & Wallace, B. "Attention is not Explanation." NAACL 2019.
   → Critical analysis of when attention IS and IS NOT a valid explanation

4. Wiegreffe, S. & Pinter, Y. "Attention is not not Explanation." EMNLP 2019.
   → Counter-argument: attention can be a valid explanation in many settings

5. Alvarez-Melis, D. & Jaakkola, T. "On the Robustness of Interpretability
   Methods." ICML Workshop 2018.
   → Stability of attention explanations

6. Arrieta, A.B. et al. "Explainable AI (XAI): Concepts, taxonomies, opportunities
   and challenges toward responsible AI." Information Fusion 2020.
   → Survey covering XAI requirements for clinical AI systems
