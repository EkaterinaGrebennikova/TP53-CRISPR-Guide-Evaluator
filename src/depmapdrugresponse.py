import os
import pandas as pd
from scipy.stats import mannwhitneyu
from tcgaallelic import classify_allelic_state

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'depmap')

MUTATIONS_FILE = os.path.join(DATA_DIR, 'OmicsSomaticMutations.csv')
CNA_FILE       = os.path.join(DATA_DIR, 'OmicsCNGeneWGS.csv')
MODEL_FILE     = os.path.join(DATA_DIR, 'Model.csv')
GDSC2_FILE     = os.path.join(DATA_DIR, 'GDSC2_fitted_dose_response_27Oct23(Sheet1).csv')
GDSC1_FILE     = os.path.join(DATA_DIR, 'GDSC1_fitted_dose_response_27Oct23.csv')

DRUGS_OF_INTEREST = ['Nutlin-3a (-)', 'Serdemetan', 'Tenovin-6']


def load_tp53_mutations():
    df = pd.read_csv(MUTATIONS_FILE, low_memory=False)
    df = df[df['HugoSymbol'] == 'TP53'].copy()
    df['aa_change'] = df['ProteinChange'].fillna('').str.replace('p.', '', n=1, regex=False)
    return df[['ModelID', 'aa_change', 'VariantType', 'RefCount', 'AltCount']].copy()


def load_tp53_cna():
    df = pd.read_csv(CNA_FILE, low_memory=False)
    tp53_col = [c for c in df.columns if c.startswith('TP53')]
    if not tp53_col:
        raise ValueError("TP53 column not found in CNA file")
    result = df[['ModelID', tp53_col[0]]].copy()
    result = result.rename(columns={tp53_col[0]: 'tp53_cna'})
    return result


def _build_sanger_to_model_map():
    model = pd.read_csv(MODEL_FILE, low_memory=False)
    model = model.dropna(subset=['SangerModelID'])
    return dict(zip(model['SangerModelID'], model['ModelID']))


def load_drug_response():
    df2 = pd.read_csv(GDSC2_FILE)
    df2['screen'] = 'GDSC2'
    df1 = pd.read_csv(GDSC1_FILE)
    df1['screen'] = 'GDSC1'
    df = pd.concat([df1, df2], ignore_index=True)
    df = df[df['DRUG_NAME'].isin(DRUGS_OF_INTEREST)].copy()
    sanger_map = _build_sanger_to_model_map()
    df['ModelID'] = df['SANGER_MODEL_ID'].map(sanger_map)
    df = df.dropna(subset=['ModelID'])

    # For duplicates (same cell line × drug in both screens), prefer GDSC2
    df = df.sort_values('screen', ascending=False).drop_duplicates(['ModelID', 'DRUG_NAME'])
    return df[['ModelID', 'DRUG_NAME', 'LN_IC50', 'screen']].copy()


def classify_cell_lines(mutations_df, cna_df):
    n_muts = mutations_df.groupby('ModelID').size().to_dict()
    cna_lookup = cna_df.set_index('ModelID')['tp53_cna'].to_dict()

    all_models = set(cna_lookup.keys())
    records = []
    for model_id in all_models:
        tp53_cna = cna_lookup.get(model_id, 0)
        n = n_muts.get(model_id, 0)
        if n == 0:
            state = 'wildtype'
        else:
            state = classify_allelic_state(None, tp53_cna, n, platform='depmap')
        records.append({
            'ModelID': model_id,
            'allelic_state': state,
            'n_tp53_muts': n,
            'tp53_cna': tp53_cna,
        })
    return pd.DataFrame(records)


def test_drug_response_by_state(classified_df, drug_df):
    merged = drug_df.merge(classified_df, on='ModelID', how='inner')
    results = {}

    for drug_name, drug_group in merged.groupby('DRUG_NAME'):
        wt = drug_group[drug_group['allelic_state'] == 'wildtype']['LN_IC50']
        drug_results = {'drug': drug_name, 'states': {}}

        states = drug_group['allelic_state'].unique()
        for state in states:
            subset = drug_group[drug_group['allelic_state'] == state]['LN_IC50']
            entry = {'n': len(subset), 'median_ln_ic50': subset.median()}
            if state != 'wildtype' and len(subset) >= 3 and len(wt) >= 3:
                _, p = mannwhitneyu(subset, wt, alternative='two-sided')
                entry['p_vs_wt'] = p
            else:
                entry['p_vs_wt'] = None
            drug_results['states'][state] = entry
        results[drug_name] = drug_results
    return results


def run_analysis():
    print("Loading DepMap data...")
    muts = load_tp53_mutations()
    cna = load_tp53_cna()
    drugs = load_drug_response()

    print(f"  TP53 mutations: {len(muts)} across {muts['ModelID'].nunique()} cell lines")
    print(f"  CNA records: {len(cna)} cell lines")
    print(f"  Drug response records: {len(drugs)} ({drugs['DRUG_NAME'].nunique()} drugs)")

    print("\nClassifying cell lines by TP53 allelic state...")
    classified = classify_cell_lines(muts, cna)
    print(f"\n--- Allelic State Distribution ({len(classified)} cell lines) ---")
    for state, count in classified['allelic_state'].value_counts().items():
        print(f"  {state:<26} {count:>5} ({count/len(classified)*100:.1f}%)")

    print("\nTesting drug response by allelic state...")
    results = test_drug_response_by_state(classified, drugs)

    state_order = ['wildtype', 'heterozygous_cn_neutral', 'heterozygous_with_gain',
                   'loh_with_mutation', 'biallelic_mutation', 'unknown']
    for drug_name, dr in results.items():
        print(f"\n--- {drug_name} ---")
        print(f"  {'State':<26} {'N':>5} {'Med ln(IC50)':>13} {'p vs WT':>12}")
        for state in state_order:
            if state not in dr['states']:
                continue
            s = dr['states'][state]
            med = f"{s['median_ln_ic50']:.3f}" if s['median_ln_ic50'] is not None else 'N/A'
            p_str = f"{s['p_vs_wt']:.2e}" if s['p_vs_wt'] is not None else '(ref)' if state == 'wildtype' else 'n<3'
            print(f"  {state:<26} {s['n']:>5} {med:>13} {p_str:>12}")


if __name__ == "__main__":
    run_analysis()
