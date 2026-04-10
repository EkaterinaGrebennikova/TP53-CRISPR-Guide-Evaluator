import pandas as pd
from tcgaloader import load_mutations, load_cna
import statistics

def compute_vaf(row):
    ref = row['t_ref_count']
    alt = row['t_alt_count']
    if pd.isna(ref) or pd.isna(alt):
        return None
    total = alt + ref
    if total == 0:
        return None
    return alt / total

def classify_allelic_state(vaf, tp53_cna, n_mutations_in_patient):
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
    
def get_allelic_context(mutations_df, cna_df):
    n_muts_per_patient = mutations_df.groupby('patient_id').size().to_dict()
    cna_lookup = cna_df.set_index('patient_id')['tp53_cna'].to_dict()
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
        vaf = compute_vaf(row)
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