# TP53 CRISPR Guide Evaluator

A computational framework that classifies TP53 allelic state and designs optimal CRISPR correction strategies for each patient context.

## Overview

TP53 is the most frequently mutated gene in human cancers (~50%), yet therapeutic options depend on the allelic context of the mutation — whether the patient retains a wild-type copy (heterozygous), has lost it (LOH), or carries multiple mutations (biallelic). This tool integrates allelic state classification with CRISPR guide design to enable mutation-specific precision correction of TP53.

The tool performs three main functions:

1. **Allelic State Classification** — Classifies TP53 mutations into five allelic states (wildtype, heterozygous CN-neutral, heterozygous with gain, LOH with mutation, biallelic mutation) using copy number alteration data and purity-adjusted variant allele frequency.

2. **Pan-Cancer Survival and Drug Response Analysis** — Evaluates the survival impact of each allelic state across 10,000+ TCGA patients using Kaplan-Meier analysis and Cox proportional hazards regression stratified by cancer type. Validates findings in an independent MSK-IMPACT cohort and analyzes drug sensitivity (Nutlin-3a, Serdemetan, Tenovin-6) by allelic state using DepMap/GDSC data.

3. **CRISPR Guide Design and Scoring** — For a given TP53 mutation, selects the optimal editing modality (CBE, ABE, prime editing, or HDR), designs guide RNAs with PAM-aware spacer selection across multiple Cas9 variants (SpCas9, SaCas9, Cas9-NG, SpRY), scores guides based on GC content, bystander damage (using DMS functional data), and off-target risk, and models expected p53 tetramer restoration using a binomial probability framework.

## Usage

```bash
python main.py -m R175H --cell-line HCT116 --zygosity heterozygous
python main.py -m R175H,R248W --cell-line U2OS --zygosity loh --cancer-type BRCA
python main.py --vcf patient.vcf --cell-line MCF7 --summary
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
