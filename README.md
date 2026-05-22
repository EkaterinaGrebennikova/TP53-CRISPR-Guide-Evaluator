# TP53 CRISPR Guide Evaluator

A computational framework that classifies TP53 allelic state and designs optimal CRISPR correction strategies for each patient context.

> **Status:** Research in progress (2026). Please contact the author before citing or building on this work.

## Overview

TP53 is the most frequently mutated gene in human cancers (~50%), yet therapeutic options depend on the allelic context of the mutation — whether the patient retains a wild-type copy (heterozygous), has lost it (LOH), or carries multiple mutations (biallelic). This tool integrates allelic state classification with CRISPR guide design to enable mutation-specific precision correction of TP53.

The tool performs three main functions:

1. **Allelic State Classification** — Classifies TP53 mutations into five allelic states (wildtype, heterozygous CN-neutral, heterozygous with gain, LOH with mutation, biallelic mutation) using copy number alteration data and purity-adjusted variant allele frequency.

2. **Pan-Cancer Survival and Drug Response Analysis** — Evaluates the survival impact of each allelic state across 10,000+ TCGA patients using Kaplan-Meier analysis and Cox proportional hazards regression stratified by cancer type. Validates findings in an independent MSK-IMPACT cohort and analyzes drug sensitivity (Nutlin-3a, Serdemetan, Tenovin-6) by allelic state using DepMap/GDSC data.

3. **CRISPR Correction Pipeline** — For a given TP53 mutation, evaluates functional severity (DMS data + IARC regression fallback), selects the optimal editing modality (CBE, ABE, prime editing, or HDR), and designs guide RNAs across four Cas9 variants. Guides are scored using a blend of heuristic features (GC content, bystander damage, off-target risk) and ML-predicted editing efficiency via gradient-boosting regressors trained on the Arbab et al. (2020) library. Model performance is evaluated by **5-fold cross-validation grouped by spacer** (GroupKFold) to prevent leakage from spacers assayed across multiple base editors. CBE is deployed on BE4 only; ABE is deployed on the canonical ABE editor — both matching inference. Honest group-CV: ABE R² = 0.730, ρ = 0.854; CBE R² = 0.531, ρ = 0.747. Post-correction outcomes are modeled through transcriptional target restoration (IARC yeast assay-derived domain penalties), cell line pathway competency (DepMap copy number), and tetramer-based prognosis classification.

## Installation

Requires Python ≥ 3.10 and [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync
```

## Usage

```bash
uv run python main.py -m R175H --cell-line HCT116 --zygosity heterozygous
uv run python main.py -m R175H,R248W --cell-line U2OS --zygosity loh --cancer-type BRCA
uv run python main.py --vcf patient.vcf --cell-line MCF7 --summary
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

> **Note:** The GDSC filenames include a date stamp from the October 2023 release. If you download a newer release, rename the files to match the names above, or update the paths in `src/depmapdrugresponse.py`.

## Citation

This work has not yet been published. If you wish to cite or build on the framework, please contact the author first.

## Contact

Ekaterina V. Grebennikova
catherine.grebennikova@gmail.com

## License

Code released under the MIT License (see `LICENSE`).
