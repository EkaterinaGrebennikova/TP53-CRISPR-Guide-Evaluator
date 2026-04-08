# Source: Bouaoun et al. Hum Mutat. 2016;37(9):865-876 -- p53.iarc.fr

import os
import pandas as pd

DATA_DIR        = os.path.join(os.path.dirname(__file__), '..', 'data')
SOMATIC_FILE    = os.path.join(DATA_DIR, 'iarc_somatic.csv')
FUNCTIONAL_FILE = os.path.join(DATA_DIR, 'iarc_functional.csv')
GERMLINE_FILE   = os.path.join(DATA_DIR, 'iarc_germline.csv')
PREVALENCE_FILE = os.path.join(DATA_DIR, 'iarc_prevalence.csv')
ASSAYS_FILE     = os.path.join(DATA_DIR, 'iarc_functional_assays.csv')
YEAST_FILE      = os.path.join(DATA_DIR, 'iarc_yeast_assays.csv')

_somatic_cache    = None
_functional_cache = None
_germline_cache   = None
_prevalence_cache = None
_assays_cache     = None
_yeast_cache      = None

def load_somatic():
    global _somatic_cache
    if _somatic_cache is not None:
        return _somatic_cache
    if not os.path.exists(SOMATIC_FILE):
        return None
    _somatic_cache = pd.read_csv(SOMATIC_FILE)
    return _somatic_cache

def load_functional():
    global _functional_cache
    if _functional_cache is not None:
        return _functional_cache
    if not os.path.exists(FUNCTIONAL_FILE):
        return None
    _functional_cache = pd.read_csv(FUNCTIONAL_FILE)
    return _functional_cache

def load_germline():
    global _germline_cache
    if _germline_cache is not None:
        return _germline_cache
    if not os.path.exists(GERMLINE_FILE):
        return None
    _germline_cache = pd.read_csv(GERMLINE_FILE)
    return _germline_cache

def load_prevalence():
    global _prevalence_cache
    if _prevalence_cache is not None:
        return _prevalence_cache
    if not os.path.exists(PREVALENCE_FILE):
        return None
    _prevalence_cache = pd.read_csv(PREVALENCE_FILE)
    return _prevalence_cache

def load_assays():
    global _assays_cache
    if _assays_cache is not None:
        return _assays_cache
    if not os.path.exists(ASSAYS_FILE):
        return None
    _assays_cache = pd.read_csv(ASSAYS_FILE)
    return _assays_cache

def load_yeast():
    global _yeast_cache
    if _yeast_cache is not None:
        return _yeast_cache
    if not os.path.exists(YEAST_FILE):
        return None
    _yeast_cache = pd.read_csv(YEAST_FILE)
    return _yeast_cache

def get_iarc_annotation(aa_change):
    func_df = load_functional()
    rows = func_df[func_df['AAchange'] == aa_change]
    if rows.empty:
        return {'found': False}
    row = rows.iloc[0]
    domain_function = row['Domain_function']
    structural_motif = row['Structural_motif']
    residue_function = row['Residue_function']
    transactivation_class = row['TransactivationClass']
    dne_lof_class = row['DNE_LOFclass']
    structure_function_class = row['StructureFunctionClass']
    hotspot = row['Hotspot']

    somatic_df = load_somatic()
    if somatic_df is not None:
        somatic_count = int((somatic_df['ProtDescription'] == 'p.' + aa_change).sum())
    else:
        somatic_count = int(row['Somatic_count']) if not pd.isna(row['Somatic_count']) else 0

    germline_df = load_germline()
    if germline_df is not None:
        germline_count = int((germline_df['ProtDescription'] == 'p.' + aa_change).sum())
    else:
        germline_count = int(row['Germline_count']) if not pd.isna(row['Germline_count']) else 0

    experimental_gof = None
    experimental_dne = None
    experimental_lof = None
    temperature_sensitive = None
    assays_df = load_assays()
    if assays_df is not None:
        rows = assays_df[assays_df['AAchange'] == aa_change]
        if not rows.empty:
            row = rows.iloc[0]
            experimental_gof = None if pd.isna(row['Gain_of_Function']) else row['Gain_of_Function']
            experimental_dne = None if pd.isna(row['Dominant_Negative_Activity']) else row['Dominant_Negative_Activity']
            experimental_lof = None if pd.isna(row['Loss_of_Function']) else row['Loss_of_Function']
            temperature_sensitive = None if pd.isna(row['Temperature_Sensitivity']) else row['Temperature_Sensitivity']

    yeast_waf1 = None
    yeast_mdm2 = None
    yeast_bax = None
    yeast_puma = None
    yeast_df = load_yeast()
    if yeast_df is not None:
        rows = yeast_df[yeast_df['AAchange'] == aa_change]
        if not rows.empty:
            row = rows.iloc[0]
            yeast_waf1 = None if pd.isna(row['WAF1nWT']) else row['WAF1nWT']
            yeast_mdm2 = None if pd.isna(row['MDM2nWT']) else row['MDM2nWT']
            yeast_bax = None if pd.isna(row['BAXnWT']) else row['BAXnWT']
            yeast_puma = None if pd.isna(row['PUMAnWT_Saos2']) else row['PUMAnWT_Saos2']

    top_cancer_types = []
    if somatic_df is not None:
        prot_desc = 'p.' + aa_change
        matches = somatic_df[somatic_df['ProtDescription'] == prot_desc]
        if not matches.empty:
            counts = matches['Short_topo'].value_counts().head(5)
            total = len(matches)
            for cancer, count in counts.items():
                top_cancer_types.append({
                    'cancer': cancer,
                    'count': int(count),
                    'fraction': round(count / total, 3)
                })
    return {
        'aa_change': aa_change,
        'domain_function': domain_function,
        'structural_motif': structural_motif,
        'residue_function': residue_function,
        'transactivation_class': transactivation_class,
        'dne_lof_class': dne_lof_class,
        'structure_function_class': structure_function_class,
        'hotspot': hotspot,
        'somatic_count': somatic_count,
        'germline_count': germline_count,
        'experimental_gof': experimental_gof,
        'experimental_dne': experimental_dne,
        'experimental_lof': experimental_lof,
        'temperature_sensitive': temperature_sensitive,
        'yeast_waf1': yeast_waf1,
        'yeast_mdm2': yeast_mdm2,
        'yeast_bax': yeast_bax,
        'yeast_puma': yeast_puma,
        'top_cancer_types': top_cancer_types,
        'found': True
    }

def get_cancer_type_distribution(aa_change):
    somatic_df = load_somatic()
    if somatic_df is None:
        return []
    prot_desc = 'p.' + aa_change
    matches = somatic_df[somatic_df['ProtDescription'] == prot_desc]
    if matches.empty:
        return []
    counts = matches['Short_topo'].value_counts()
    total = len(matches)
    result = []
    for cancer, count in counts.items():
        result.append({
            'cancer': cancer,
            'count': int(count),
            'fraction': round(count / total, 3)
        })
    return result

def get_germline_status(aa_change):
    germline_df = load_germline()
    if germline_df is None:
        return {'germline_present': False, 'family_count': 0, 'top_cancer_types': []}
    prot_desc = 'p.' + aa_change
    matches = germline_df[germline_df['ProtDescription'] == prot_desc]
    if matches.empty:
        return {'germline_present': False, 'family_count': 0, 'top_cancer_types': []}
    family_count = matches['Family_ID'].nunique()
    cancer_counts = matches['Short_topo'].dropna().value_counts()
    total = len(matches['Short_topo'].dropna())
    top_cancer_types = []
    for cancer, count in cancer_counts.head(5).items():
        top_cancer_types.append({
            'cancer': cancer,
            'count': int(count),
            'fraction': round(count/total, 3)
        })
    return {
        'germline_present': True,
        'family_count': family_count,
        'top_cancer_types': top_cancer_types
    }