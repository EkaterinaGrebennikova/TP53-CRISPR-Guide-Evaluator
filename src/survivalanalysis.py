from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
import pandas as pd
from tcgaloader import load_mutations, load_cna, load_clinical
from tcgaallelic import compute_vaf, classify_allelic_state

def build_survival_df(mutations_df, cna_df, clinical_df):
    df = clinical_df[['patient_id', 'cancer_type', 'os_status', 'os_months', 'age', 'sex']].copy()
    df = df.dropna(subset=['os_status', 'os_months'])
    df['os_event'] = (df['os_status'] == '1:DECEASED').astype(int)

    mutation_set = set(mutations_df['patient_id'])
    df['tp53_mut'] = df['patient_id'].isin(mutation_set)

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
        pid = row['patient_id']
        vaf = compute_vaf(row)
        tp53_cna = cna_lookup.get(pid, 0)
        n_muts = n_muts_per_patient.get(pid, 1)
        state = classify_allelic_state(vaf, tp53_cna, n_muts)
        if pid not in patient_state or priority[state] > priority[patient_state[pid]]:
            patient_state[pid] = state

    df['allelic_state'] = df['patient_id'].map(lambda p: patient_state.get(p, 'wildtype'))
    return df

def km_tp53_mut_vs_wt(survival_df):
    kmf_wt = KaplanMeierFitter()
    kmf_mut = KaplanMeierFitter()
    mut = survival_df[survival_df['tp53_mut']]
    wt  = survival_df[~survival_df['tp53_mut']]
    kmf_wt = kmf_wt.fit(wt['os_months'], wt['os_event'], label = 'TP53-WT')
    kmf_mut = kmf_mut.fit(mut['os_months'], mut['os_event'], label = 'TP53-MUT')
    logrank = logrank_test(
        mut['os_months'], wt['os_months'],
        event_observed_A=mut['os_event'],
        event_observed_B=wt['os_event']
    )
    return {
        'n_mut': len(mut),
        'n_wt': len(wt),
        'median_os_mut': kmf_mut.median_survival_time_,
        'median_os_wt': kmf_wt.median_survival_time_,
        'logrank_p': logrank.p_value,
        'km_mut': kmf_mut,
        'km_wt': kmf_wt,
    }

def km_by_allelic_state(survival_df):
    df = survival_df[survival_df['allelic_state'] != 'unknown']
    wt = df[df['allelic_state'] == 'wildtype']
    states = ['heterozygous_cn_neutral', 'heterozygous_with_gain', 'loh_with_mutation', 'biallelic_mutation']
    kmf_wt = KaplanMeierFitter()
    kmf_wt.fit(wt['os_months'], wt['os_event'], label='wildtype')
    results = {
        'wildtype': {
            'n': len(wt),
            'median_os': kmf_wt.median_survival_time_,
            'logrank_p_vs_wt': None,
            'km': kmf_wt
        }
    }
    for state in states:
        subset = df[df['allelic_state'] == state]
        sub_kmf = KaplanMeierFitter()
        sub_kmf.fit(subset['os_months'], subset['os_event'], label=state)
        logrank = logrank_test(
            subset['os_months'], wt['os_months'],
            event_observed_A=subset['os_event'],
            event_observed_B=wt['os_event']
        )
        results[state] = {
            'n': len(subset),
            'median_os': sub_kmf.median_survival_time_,
            'logrank_p_vs_wt': logrank.p_value,
            'km': sub_kmf
        }
    return results

def cox_regression(survival_df):
    df = survival_df.copy()
    df = df.dropna(subset=['age', 'sex', 'cancer_type'])
    df = df[df['allelic_state'] != 'unknown']
    df['sex_binary'] = (df['sex'] == 'Male').astype(int)

    df = pd.get_dummies(df, columns=['allelic_state'], drop_first=False)
    df = df.drop(columns=['allelic_state_wildtype'])

    keep = ['os_months', 'os_event', 'age', 'sex_binary', 'cancer_type',
            'allelic_state_heterozygous_cn_neutral',
            'allelic_state_heterozygous_with_gain',
            'allelic_state_loh_with_mutation',
            'allelic_state_biallelic_mutation']
    df = df[keep].dropna()

    cph = CoxPHFitter()
    cph.fit(df, duration_col='os_months', event_col='os_event', strata=['cancer_type'])

    return {
        'summary': cph.summary,
        'concordance': cph.concordance_index_,
        'n_observations': len(df),
        'model': cph,
    }

def survival_by_cancer_type(survival_df, min_patients = 30):
    results = {}
    for cancer_type, group in survival_df.groupby('cancer_type'):
        if len(group) < min_patients:
            continue
        mut = group[group['tp53_mut']]
        wt  = group[~group['tp53_mut']]
        if len(mut) < 5 or len(wt) < 5:
            continue
        kmf_wt = KaplanMeierFitter()
        kmf_mut = KaplanMeierFitter()
        kmf_wt.fit(wt['os_months'], wt['os_event'], label = 'TP53-WT')
        kmf_mut.fit(mut['os_months'], mut['os_event'], label = 'TP53-MUT')
        logrank = logrank_test(
            mut['os_months'], wt['os_months'],
            event_observed_A=mut['os_event'],
            event_observed_B=wt['os_event']
        )
        results[cancer_type] = {
            'n_mut': len(mut),
            'n_wt': len(wt),
            'median_os_mut': kmf_mut.median_survival_time_,
            'median_os_wt': kmf_wt.median_survival_time_,
            'logrank_p': logrank.p_value,
        }
    return results