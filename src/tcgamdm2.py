from tcgaloader import load_mutations, load_cna, load_clinical, merge_patient_data
from scipy.stats import fisher_exact

def get_mdm2_amplification_stats(cna_df, mutations_df):
    mutation_set = set(mutations_df['patient_id'])
    tp53_and_mdm2 = 0
    tp53mut = 0
    mdm2amp = 0
    neither = 0
    for _, row in cna_df.iterrows():
        patient_id = row['patient_id']
        tp53_mut = patient_id in mutation_set
        mdm2_amp = row['mdm2_cna'] >= 2
        if tp53_mut and mdm2_amp:
            tp53_and_mdm2 += 1
        elif tp53_mut:
            tp53mut += 1
        elif mdm2_amp:
            mdm2amp += 1
        else:
            neither += 1
    table = [[tp53_and_mdm2, tp53mut], [mdm2amp, neither]]
    odds_ratio, p_value = fisher_exact(table, alternative = 'less')
    return {
        'total_patients': len(cna_df),
        'tp53_mutant': tp53_and_mdm2 + tp53mut,
        'mdm2_amplified': tp53_and_mdm2 + mdm2amp,
        'cooccurrence': {
            'tp53mut_and_mdm2amp': tp53_and_mdm2,
            'tp53mut_only': tp53mut,
            'mdm2amp_only': mdm2amp,
            'neither': neither
        },
        'cooccurrence_fraction': round(tp53_and_mdm2/(tp53_and_mdm2+tp53mut), 3) if (tp53_and_mdm2+tp53mut) > 0 else 0,
        'odds_ratio': round(odds_ratio, 3),
        'fisher_p_value': p_value
    }

def get_mdm2_by_cancer_type(cna_df, mutations_df, clinical_df):
    merged = cna_df.merge(clinical_df[['patient_id', 'cancer_type']], on = 'patient_id', how = 'left')
    mutation_set = set(mutations_df['patient_id'])
    merged = merged.dropna(subset = ['cancer_type'])
    results = {}
    for cancer_type, group_df in merged.groupby('cancer_type'):
        total = len(group_df)
        tp53mut_num = 0
        mdm2amp_num = 0
        both_num = 0
        for _, row in group_df.iterrows():
            is_tp53mut = row['patient_id'] in mutation_set
            is_mdm2amp = row['mdm2_cna'] >= 2
            if is_tp53mut:
                tp53mut_num += 1
            if is_mdm2amp:
                mdm2amp_num += 1
            if is_mdm2amp and is_tp53mut:
                both_num += 1
        results[cancer_type] = {
            'total': total,
            'tp53_mut_rate': round(tp53mut_num / total, 3),
            'mdm2_amp_rate': round(mdm2amp_num/total, 3),
            'cooccurrence_rate': round(both_num/total, 3)
        }
    return results

def classify_patient_therapy_candidate(patient_row, mutation_set):
    has_tp53mut = (patient_row['patient_id'] in mutation_set)
    has_mdm2amp = (patient_row['mdm2_cna'] >= 2)
    if has_mdm2amp and has_tp53mut:
        return 'compound_therapy_candidate'
    elif has_tp53mut:
        return 'base_editing_candidate'
    elif has_mdm2amp:
        return 'mdm2_inhibitor_candidate'
    else:
        return 'no_intervention_needed'
    
def get_therapy_candidates_by_cancer_type(cna_df, mutations_df, clinical_df):
    merged = cna_df.merge(clinical_df[['patient_id', 'cancer_type']], on = 'patient_id', how = 'left')
    mutation_set = set(mutations_df['patient_id'])
    merged = merged.dropna(subset = ['cancer_type'])
    results = {}
    for cancer_type, group_df in merged.groupby('cancer_type'):
        base_editing = 0
        mdm2_inhibitor = 0
        compound = 0
        none = 0
        for _, row in group_df.iterrows():
            rec = classify_patient_therapy_candidate(row, mutation_set)
            if rec == 'base_editing_candidate':
                base_editing += 1
            elif rec == 'compound_therapy_candidate':
                compound += 1
            elif rec == 'mdm2_inhibitor_candidate':
                mdm2_inhibitor += 1
            else:
                none += 1
        results[cancer_type] = {
            'base_editing': base_editing,
            'mdm2_inhibitor': mdm2_inhibitor,
            'compound': compound,
            'none': none
        }
    return results

def get_nutlin_candidate_fraction(mutations_df, cna_df):
    merged = mutations_df.merge(cna_df[['patient_id', 'mdm2_cna']], on='patient_id', how='left')
    merged['mdm2_cna'] = merged['mdm2_cna'].fillna(0)
    merged = merged.drop_duplicates(subset=['patient_id', 'aa_change'])
    results = []
    for aa_change, group_df in merged.groupby('aa_change'):
        total = group_df['patient_id'].nunique()
        amp_count = int((group_df['mdm2_cna'] >= 2).sum())
        fraction = round(amp_count / total, 3) if total > 0 else 0
        results.append({
            'aa_change': aa_change,
            'total_patients': total,
            'mdm2_amp_count': amp_count,
            'mdm2_amp_fraction': fraction,
            'strong_candidate': fraction > 0.30
        })
    results.sort(key=lambda x: x['mdm2_amp_fraction'], reverse=True)
    return results
