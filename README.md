# TP53 CRISPR Guide Evaluator

A computational framework that classifies TP53 allelic state and recommends allelic-state-guided treatment strategies across the therapeutic armamentarium — with CRISPR correction guide design as one branch.

> **Status:** Research in progress (2026). Please contact the author before citing or building on this work.

## Overview

TP53 is the most frequently mutated gene in human cancers (~50%), yet therapeutic options depend on the allelic context of the mutation — whether the patient retains a wild-type copy (heterozygous), has lost it (LOH), or carries multiple mutations (biallelic). This tool integrates allelic state classification with cancer-type-adjusted drug-response analysis and therapeutic stratification to recommend context-appropriate treatment, with CRISPR correction guide design as one branch.

The tool performs three main functions:

1. **Allelic State Classification** — Classifies TP53 mutations into five allelic states (wildtype, heterozygous CN-neutral, heterozygous with gain, LOH with mutation, biallelic mutation) using copy number alteration data and purity-adjusted variant allele frequency.

2. **Pan-Cancer Survival and Drug Response Analysis** — Evaluates the survival impact of each allelic state across 10,000+ TCGA patients using Kaplan-Meier analysis and Cox proportional hazards regression stratified by cancer type. Validates findings in an independent MSK-IMPACT cohort. Analyzes drug sensitivity across six therapeutic classes (MDM2 inhibitors, DNA-damaging chemotherapy, p53-independent anti-mitotics, synthetic-lethality agents, mutant-p53 reactivators, gene correction) using DepMap/GDSC cell-line data, with cancer-type-adjusted OLS regression and Benjamini-Hochberg correction applied within each mechanistic panel.

3. **Allelic-State-Guided Treatment Recommendation** (primary output) — For a given mutation and allelic state, outputs a ranked therapeutic recommendation across the six classes above, grounded in the empirical survival and drug-response findings. Mutation-specific gates refine the reactivator branch (e.g. truncating → N/A; Y220C → rezatapopt). Each mutation is also evaluated for functional severity (DMS + IARC), IARC functional annotation, and a pre/post-correction tetramer-based functional restoration estimate. When CRISPR correction is selected as a branch, the tool selects the editing modality (CBE/ABE/PE/HDR), designs and scores guide RNAs across four Cas9 variants, and predicts correction efficiency via gradient-boosting regressors — group-by-spacer GroupKFold validation; CBE deployed on BE4 only (R² = 0.531, ρ = 0.747, RMSE σ = 0.193), ABE on the canonical ABE editor (R² = 0.730, ρ = 0.854, RMSE σ = 0.201). The full CRISPR guide-design report is opt-in (`--design-guides`); cell-line prognosis is opt-in (`--cell-line`).

## Installation

Requires Python ≥ 3.10 and [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync
```

## Usage

The primary output is an allelic-state-guided treatment recommendation. Provide a mutation (`-m`) and an allelic state (`--allelic-state`: `het`, `het_gain`, `loh`, or `biallelic`).

```bash
# Treatment recommendation (primary output)
uv run python main.py -m R175H --allelic-state loh
uv run python main.py -m R175H --allelic-state het --cancer-type BRCA

# Optional: cell-line-specific prognosis
uv run python main.py -m R175H --allelic-state loh --cell-line MCF7

# Optional: full CRISPR guide-design report
uv run python main.py -m Y220C --allelic-state loh --design-guides

# Optional: pan-cancer TCGA therapy landscape
uv run python main.py -m R175H --allelic-state loh --landscape

# VCF input
uv run python main.py --vcf patient.vcf --allelic-state biallelic
```

| Flag | Purpose |
|------|---------|
| `-m / --mutations` | mutation(s), comma-separated (e.g. `R175H,R248W`) |
| `--vcf` | VCF file input (alternative to `-m`) |
| `--allelic-state` | `het` / `het_gain` / `loh` / `biallelic` (primary driver) |
| `--zygosity` | alias: `heterozygous` / `loh` / `homozygous` |
| `--cell-line` | *optional* — adds cell-line prognosis (HCT116 / U2OS / MCF7) |
| `--design-guides` | *optional* — show the full CRISPR guide-design report |
| `--landscape` | *optional* — show the pan-cancer TCGA therapy landscape |
| `--cancer-type` | cancer-type context label |
| `--output` | save results (`.json` / `.tsv`) |

## Repository Structure

```
TP53-CRISPR-Guide-Evaluator/
├── main.py                          # CLI entry point (recommendation-first)
├── src/                             # Core modules
│   ├── treatmentrecommender.py      # Allelic-state-guided recommender
│   ├── survivalanalysis.py          # KM, Cox regression, BH correction
│   ├── depmapdrugresponse.py        # DepMap/GDSC drug response loader
│   ├── modalityselector.py          # CRISPR modality selection
│   ├── guidedesigner.py             # Spacer + PAM search
│   ├── guidescorer.py               # Guide heuristic + ML scoring
│   ├── efficiencypredictorml.py     # CBE/ABE gradient boosting models
│   ├── allelemodel.py               # Tetramer fraction model
│   └── ...
├── analysis/                        # Pipeline / analysis scripts
│   ├── gdsc_chemo_figure.py         # Chemo LOH-resistance forest plot
│   ├── gdsc_targeted_figure.py      # Targeted-class forest plot (MDM2i + synth-leth)
│   ├── fig_wtloss_correctability.py # Gene-correction ceiling figure
│   ├── fig_stratification_matrix.py # Stratification matrix figure
│   └── ...
├── data/                            # Local datasets (TCGA, IARC, etc.)
├── data/depmap/                     # DepMap/GDSC (manual download, gitignored)
└── figures/                         # Generated figures (PNG)
```

## Data Sources

Most data is included in the repository or fetched automatically via API. The following large datasets must be downloaded manually.

### Included in repository
- **TCGA PanCancer Atlas mutations/clinical**: fetched automatically via cBioPortal API (`src/fetchtcgamutations.py`)
- **TCGA CNA, ABSOLUTE purity, MSK-IMPACT data**: included in `data/`
- **IARC TP53 Database files**: included in `data/`
- **COSMIC cancer gene list, DMS scores, PDB structure**: included in `data/`

### Manual download required (DepMap/GDSC)

The drug response analysis requires five files placed in `data/depmap/`. Download them and **keep the exact filenames shown below**:

| File | Source | Download from |
|------|--------|---------------|
| `OmicsSomaticMutations.csv` | DepMap | https://depmap.org/portal/data_page/?tab=allData — search "OmicsSomaticMutations" |
| `OmicsCNGeneWGS.csv` | DepMap | https://depmap.org/portal/data_page/?tab=allData — search "OmicsCNGeneWGS" |
| `Model.csv` | DepMap | https://depmap.org/portal/data_page/?tab=allData — search "Model" |
| `GDSC2_fitted_dose_response_27Oct23(Sheet1).csv` | GDSC | https://www.cancerrxgene.org/downloads/bulk_download — GDSC2 fitted dose response |
| `GDSC1_fitted_dose_response_27Oct23.csv` | GDSC | https://www.cancerrxgene.org/downloads/bulk_download — GDSC1 fitted dose response |

```
data/depmap/
├── OmicsSomaticMutations.csv
├── OmicsCNGeneWGS.csv
├── Model.csv
├── GDSC2_fitted_dose_response_27Oct23(Sheet1).csv
└── GDSC1_fitted_dose_response_27Oct23.csv
```

> **Note:** The GDSC filenames include a date stamp from the October 2023 release. If you download a newer release, rename the files to match the names above, or update the paths in `src/depmapdrugresponse.py`. These external datasets are gitignored and must be downloaded by every user.

## Citation

This work has not yet been published. If you wish to cite or build on the framework, please contact the author first.

## Contact

Ekaterina V. Grebennikova
catherine.grebennikova@gmail.com

## License

Code released under the MIT License (see `LICENSE`).
