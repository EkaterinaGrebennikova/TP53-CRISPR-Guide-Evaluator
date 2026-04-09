# src/tcgaloader.py
# Loads TCGA Pan-Cancer Atlas data from cBioPortal
# Source: TCGA Research Network, PanCancer Atlas 2018

import os
import pandas as pd

DATA_DIR       = os.path.join(os.path.dirname(__file__), '..', 'data')
MUTATIONS_FILE = os.path.join(DATA_DIR, 'tcga_mutations.txt')
CNA_FILE       = os.path.join(DATA_DIR, 'tcga_cna.txt')
CLINICAL_FILE  = os.path.join(DATA_DIR, 'tcga_clinical.txt')

_mutations_cache = None
_cna_cache       = None
_clinical_cache  = None


def load_mutations():
    global _mutations_cache
    if _mutations_cache is not None:
        return _mutations_cache
    df = pd.read_csv(MUTATIONS_FILE, sep='\t', comment='#', low_memory=False)
    df = df[df['Hugo_Symbol'] == 'TP53'].copy()
    df['patient_id'] = df['Tumor_Sample_Barcode'].str[:12]
    df['aa_change'] = df['HGVSp_Short'].str.replace('p.', '', n=1, regex=False)
    df = df[['patient_id', 'aa_change', 'Variant_Classification',
             't_ref_count', 't_alt_count', 'Chromosome', 'Start_Position',
             'Reference_Allele', 'Tumor_Seq_Allele2', 'HGVSp_Short']].copy()
    _mutations_cache = df
    return _mutations_cache


def load_cna():
    global _cna_cache
    if _cna_cache is not None:
        return _cna_cache
    _cna_cache = pd.read_csv(CNA_FILE, sep='\t')
    return _cna_cache


def load_clinical():
    global _clinical_cache
    if _clinical_cache is not None:
        return _clinical_cache
    df = pd.read_csv(CLINICAL_FILE, sep='\t', comment='#')
    df = df.rename(columns={
        'Patient ID': 'patient_id',
        'TCGA PanCanAtlas Cancer Type Acronym': 'cancer_type',
        'Overall Survival Status': 'os_status',
        'Overall Survival (Months)': 'os_months',
        'Disease Free Status': 'dfs_status',
        'Disease Free (Months)': 'dfs_months',
        'Diagnosis Age': 'age',
        'Sex': 'sex'
    })
    df['patient_id'] = df['patient_id'].str[:12]
    df['os_months'] = pd.to_numeric(df['os_months'], errors='coerce')
    df['dfs_months'] = pd.to_numeric(df['dfs_months'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df = df[['patient_id', 'cancer_type', 'os_status', 'os_months',
             'dfs_status', 'dfs_months', 'age', 'sex']].copy()
    _clinical_cache = df
    return _clinical_cache


def merge_patient_data(mutations_df, cna_df, clinical_df):
    merged = mutations_df.merge(cna_df, on='patient_id', how='left')
    merged = merged.merge(clinical_df, on='patient_id', how='left')
    return merged

