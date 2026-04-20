import os
import pandas as pd
from tcgaloader import load_mutations, load_cna
import statistics

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
PURITY_FILE = os.path.join(DATA_DIR, 'tcga_absolute_purity.txt')

_purity_cache = None

def load_purity():
    global _purity_cache
    if _purity_cache is not None:
        return _purity_cache
    df = pd.read_csv(PURITY_FILE, sep='\t')
    df = df[df['call status'] == 'called']
    df['patient_id'] = df['array'].str[:12]
    _purity_cache = dict(zip(df['patient_id'], df['purity']))
    return _purity_cache


def compute_vaf(row, purity=None):
    ref = row['t_ref_count']
    alt = row['t_alt_count']
    if pd.isna(ref) or pd.isna(alt):
        return None
    total = alt + ref
    if total == 0:
        return None
    raw_vaf = alt / total
    if purity is not None and purity > 0.1:
        return min(raw_vaf / purity, 1.0)
    return raw_vaf

def classify_allelic_state(vaf, tp53_cna, n_mutations_in_patient, platform='tcga'):
    # platform='tcga': GISTIC integer CNA (-2..+2), requires VAF
    # platform='depmap': relative copy number (~0.5-1.5), no VAF available
    if platform == 'depmap':
        # DepMap thresholds: relative CN centered on 1.0
        # Relaxed to capture partial gains/losses (dataset-specific calibration)
        loss_threshold = 0.85  # < 0.85 = loss (LOH-like)
        gain_threshold = 1.15  # > 1.15 = gain
        neutral_low, neutral_high = 0.95, 1.05
        if n_mutations_in_patient >= 2:
            return 'biallelic_mutation'
        elif n_mutations_in_patient >= 1 and tp53_cna < loss_threshold:
            return 'loh_with_mutation'
        elif n_mutations_in_patient >= 1 and tp53_cna > gain_threshold:
            return 'heterozygous_with_gain'
        elif n_mutations_in_patient >= 1 and neutral_low <= tp53_cna <= neutral_high:
            return 'heterozygous_cn_neutral'
        else:
            return 'unknown'

    # TCGA (GISTIC integer) path
    if n_mutations_in_patient >= 2 and tp53_cna >= 0:
        return 'biallelic_mutation'
    elif tp53_cna <= -1 and vaf is not None and vaf > 0.7:
        return 'loh_with_mutation'
    elif tp53_cna >= 1:
        return 'heterozygous_with_gain'
    elif -0.5 < tp53_cna < 0.5 and vaf is not None and 0.3 < vaf < 0.7:
        return 'heterozygous_cn_neutral'
    else:
        return 'unknown'
    
def get_allelic_context(mutations_df, cna_df, use_purity=True):
    n_muts_per_patient = mutations_df.groupby('patient_id').size().to_dict()
    cna_lookup = cna_df.set_index('patient_id')['tp53_cna'].to_dict()
    purity_lookup = load_purity() if use_purity and os.path.exists(PURITY_FILE) else {}
    priority = {
        'biallelic_mutation': 4,
        'loh_with_mutation': 3,
        'heterozygous_with_gain': 2,
        'heterozygous_cn_neutral': 1,
        'unknown': 0
    }
    patient_state = {}
    for _, row in mutations_df.iterrows():
        patient_id = row['patient_id']
        purity = purity_lookup.get(patient_id)
        vaf = compute_vaf(row, purity=purity)
        tp53_cna = cna_lookup.get(patient_id, 0)
        n_muts = n_muts_per_patient.get(patient_id, 1)
        state = classify_allelic_state(vaf, tp53_cna, n_muts)
        if patient_id not in patient_state or priority[state] > priority[patient_state[patient_id]]:
            patient_state[patient_id] = state
    states = {k: 0 for k in priority}
    for s in patient_state.values():
        states[s] += 1
    total = len(patient_state)
    return {
        'total_tp53_mutant_patients': total,
        'states': states,
        'loh_fraction': round(states['loh_with_mutation'] / total, 3) if total > 0 else 0
    }

def get_vaf_distribution(mutations_df, aa_change):
    filtered = mutations_df[mutations_df['aa_change'] == aa_change]
    vafs = [compute_vaf(row) for _, row in filtered.iterrows()]
    vafs = [v for v in vafs if v is not None]
    if len(vafs) == 0:
        return {'aa_change': aa_change, 'n': 0, 'mean_vaf': 0, 'median_vaf': 0, 'vaf_gt_0.7_fraction': 0}
    high_vaf_count = sum(1 for v in vafs if v > 0.7)
    return {
        'aa_change': aa_change,
        'n': len(vafs),
        'mean_vaf': round(statistics.mean(vafs), 3),
        'median_vaf': round(statistics.median(vafs), 3),
        'vaf_gt_0.7_fraction': round(high_vaf_count / len(vafs), 3)
    }

def get_allelic_context_by_cancer_type(mutations_df, cna_df, clinical_df):
    merged = mutations_df.merge(clinical_df[['patient_id', 'cancer_type']], on='patient_id', how='left')
    merged = merged.dropna(subset=['cancer_type'])
    results = {}
    for cancer_type, group_df in merged.groupby('cancer_type'):
        result = get_allelic_context(group_df, cna_df)
        results[cancer_type] = result
    return results