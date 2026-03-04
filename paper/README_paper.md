# Research Paper — Compilation Guide

## Files
| File | Description |
|------|-------------|
| `main.tex` | Full IEEE conference paper (LaTeX source) |
| `references.bib` | 20 IEEE-format BibTeX references |
| `generate_figures.py` | Python script to regenerate all 7 figures |
| `figures/` | Pre-generated figures (PDF + PNG, 300 DPI) |

## Figures
| File | Content |
|------|---------|
| `fig1_han_architecture` | HAN++ architecture diagram |
| `fig2_graph_structure` | Heterogeneous medical graph |
| `fig3_clinical_workflow` | Two-phase clinical workflow |
| `fig4_model_comparison` | All models F1-Macro comparison |
| `fig5_ablation_summary` | 4-panel ablation study |
| `fig6_training_convergence` | Training curves |
| `fig7_dataset_statistics` | Dataset statistics |

## To Compile PDF

### Option 1: Local LaTeX (recommended)
```bash
# Install MacTeX (macOS)
brew install --cask mactex

# Then compile
cd paper/
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex   # 3rd pass for cross-references
```

### Option 2: Overleaf (online, easiest)
1. Go to https://www.overleaf.com
2. New Project → Upload Project
3. Upload: `main.tex`, `references.bib`, and the entire `figures/` folder
4. Overleaf auto-detects IEEEtran and compiles

### Option 3: Docker LaTeX
```bash
docker run --rm -v $(pwd):/work texlive/texlive \
  bash -c "cd /work/paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex"
```

## Required LaTeX Packages
All standard — included with any TeX Live / MacTeX installation:
- `IEEEtran` class (included in TeX Live)
- `cite`, `amsmath`, `graphicx`, `booktabs`
- `multirow`, `hyperref`, `subcaption`
- `algorithm`, `algpseudocode`

## Regenerating Figures
```bash
cd /path/to/HAN-implementation
python paper/generate_figures.py
```

## Paper Structure
| Section | Content |
|---------|---------|
| Abstract | Problem, method, results (F1-Macro=0.8432) |
| I. Introduction | Motivation, limitations of flat ML, contributions |
| II. Related Work | GRAM, HGT, EHR GNNs, attention networks |
| III. Methodology | Graph construction, HAN++ architecture (equations), workflow |
| IV. Experimental Setup | Dataset, baselines, training config, metrics |
| V. Results | Main table, model comparison, ablation, convergence |
| VI. Conclusion | Summary, limitations, future work |
| References | 20 IEEE-format citations |

## Paper Details
- **Title:** CareAI: Graph-Based Multi-Head Disease Prediction and Early Warning System
- **Authors:** Anushka Samaranayake, JohnPeter Charles, Tashin Kavishan, Mailki Meghana Wickramasinghe
- **Affiliation:** Dept. Electronic and Telecommunication Engineering / Biomedical Engineering, University of Moratuwa, Sri Lanka
- **Ethics:** EDN/2026/004 — University of Moratuwa Ethics Review Committee
- **Data:** CareCode (PVT) Ltd, 6 Sri Lankan hospitals (de-identified)
- **21 BibTeX references** (includes SeHGNN yang_2023 — verify exact details before submission)

## Key Results (for slides/poster)
- HAN++ F1-Macro: **0.8432** (P-D-P meta-path)
- Best traditional: 0.4753 (Decision Tree) → **+77.4% improvement**
- pyHGT (mid-review best): 0.7550 → HAN++ better by **+11.6%**
- HGT-HAN (P-D-P): 0.8401 → HAN++ better by **+0.37%**
- SeHGNN (feasibility): ~0.63 → HAN++ better by **+33.8%**
- 156K parameters, 40 epochs, 132s training time

## ⚠ Before Final Submission
- Verify `yang_2023` (SeHGNN) citation in references.bib — check exact author list, page numbers, and DOI against the actual paper
- Confirm pyHGT results (F1-Macro = 0.7550) are on equivalent evaluation scope as HAN++
