"""Audit GDSC1+GDSC2 for drug-class coverage in TP53-mutant cell lines,
stratified by allelic state. Determines what's buildable for the
expanded paper §3.7 (allelic-state-stratified drug sensitivity beyond
MDM2 inhibitors).

For each target drug class:
  1. Search GDSC by name (case-insensitive substring match).
  2. For each found drug, count cell lines tested overall and by
     TP53 allelic state.
  3. Mann-Whitney ln(IC50) vs WT for each non-WT state with n>=3.
  4. Flag drugs as "buildable" (>=30 TP53-mut lines, >=10 per state)
     or "marginal" / "insufficient".
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from scipy.stats import mannwhitneyu

from depmapdrugresponse import (
    load_tp53_mutations, load_tp53_cna, classify_cell_lines,
    _build_sanger_to_model_map, GDSC1_FILE, GDSC2_FILE,
)

# Target drugs to search for by class (lowercase substrings).
TARGETS = {
    'MDM2 inhibitors (extended)': [
        'nutlin', 'idasanutlin', 'rg7388', 'amg-232', 'amg232',
        'serdemetan', 'tenovin', 'mdm2',
    ],
    'Mutant-p53 reactivators': [
        'apr-246', 'apr246', 'eprenetapopt', 'prima-1',
        'coti-2', 'coti2', 'arsenic', 'ato', 'zmc1',
    ],
    'WEE1 inhibitors': [
        'adavosertib', 'azd1775', 'mk-1775', 'mk1775', 'wee1',
    ],
    'ATR inhibitors': [
        'ceralasertib', 'azd6738', 'berzosertib', 've-822',
        'vx-970', 'm6620', 'atr inhibitor', 'azd6738',
    ],
    'PARP inhibitors': [
        'olaparib', 'talazoparib', 'niraparib', 'rucaparib',
        'veliparib', 'iniparib', 'parp',
    ],
    'AURK / CHK1 (replication stress)': [
        'alisertib', 'mln8237', 'prexasertib', 'ly2606368',
        'mk-8776', 'mk8776', 'azd7762', 'aurora', 'chk1',
    ],
}

STATE_ORDER = [
    'wildtype', 'heterozygous_cn_neutral', 'heterozygous_with_gain',
    'loh_with_mutation', 'biallelic_mutation', 'unknown',
]


def load_all_drugs():
    df2 = pd.read_csv(GDSC2_FILE); df2['screen'] = 'GDSC2'
    df1 = pd.read_csv(GDSC1_FILE); df1['screen'] = 'GDSC1'
    df = pd.concat([df1, df2], ignore_index=True)
    sanger = _build_sanger_to_model_map()
    df['ModelID'] = df['SANGER_MODEL_ID'].map(sanger)
    df = df.dropna(subset=['ModelID'])
    df = df.sort_values('screen', ascending=False).drop_duplicates(
        ['ModelID', 'DRUG_NAME'])
    return df[['ModelID', 'DRUG_NAME', 'LN_IC50', 'screen']].copy()


def find_matches(all_drug_names, terms):
    lowered = {n: n.lower() for n in all_drug_names}
    matches = set()
    for term in terms:
        t = term.lower()
        for orig, low in lowered.items():
            if t in low:
                matches.add(orig)
    return sorted(matches)


def audit_drug(drug_name, drugs_df, classified):
    sub = drugs_df[drugs_df['DRUG_NAME'] == drug_name]
    merged = sub.merge(classified, on='ModelID', how='inner')
    wt = merged[merged['allelic_state'] == 'wildtype']['LN_IC50']
    out = {'drug': drug_name, 'total': len(merged),
           'screen': sub['screen'].iloc[0] if len(sub) else '?',
           'states': {}}
    n_tp53_mut = 0
    for state in STATE_ORDER:
        s = merged[merged['allelic_state'] == state]['LN_IC50']
        entry = {'n': len(s), 'median_ln_ic50': s.median() if len(s) else None,
                 'p_vs_wt': None}
        if state != 'wildtype' and len(s) >= 3 and len(wt) >= 3:
            _, p = mannwhitneyu(s, wt, alternative='two-sided')
            entry['p_vs_wt'] = p
        if state != 'wildtype' and state != 'unknown':
            n_tp53_mut += len(s)
        out['states'][state] = entry
    out['n_tp53_mut'] = n_tp53_mut
    return out


def verdict(audit_result):
    n_mut = audit_result['n_tp53_mut']
    states = audit_result['states']
    n_loh = states['loh_with_mutation']['n']
    n_het = states['heterozygous_cn_neutral']['n']
    n_gain = states['heterozygous_with_gain']['n']
    n_bia = states['biallelic_mutation']['n']
    state_min = min(n_loh, n_het, n_gain, n_bia)
    if n_mut >= 30 and state_min >= 10:
        return 'BUILDABLE'
    if n_mut >= 20 and state_min >= 5:
        return 'marginal'
    return 'insufficient'


def main():
    print("Loading DepMap mutations + CNA, classifying cell lines...")
    muts = load_tp53_mutations()
    cna = load_tp53_cna()
    classified = classify_cell_lines(muts, cna)
    state_counts = classified['allelic_state'].value_counts().to_dict()
    print(f"  cell lines classified: {len(classified)}")
    for s in STATE_ORDER:
        print(f"    {s:<26} {state_counts.get(s, 0):>5}")

    print("\nLoading GDSC1 + GDSC2 (all drugs)...")
    drugs = load_all_drugs()
    all_names = sorted(drugs['DRUG_NAME'].unique())
    print(f"  unique drugs: {len(all_names)}")
    print(f"  total cell-line x drug records: {len(drugs)}")

    print("\n" + "=" * 84)
    print("DRUG CLASS AUDIT")
    print("=" * 84)
    summary = []
    for class_name, terms in TARGETS.items():
        matches = find_matches(all_names, terms)
        print(f"\n--- {class_name} ---")
        if not matches:
            print(f"  no matches found for: {', '.join(terms)}")
            continue
        for drug in matches:
            r = audit_drug(drug, drugs, classified)
            v = verdict(r)
            s = r['states']
            print(f"  {drug} ({r['screen']}): "
                  f"n_total={r['total']}  n_TP53mut={r['n_tp53_mut']}  "
                  f"[{v}]")
            print(f"    WT n={s['wildtype']['n']}  "
                  f"hetCN n={s['heterozygous_cn_neutral']['n']}  "
                  f"hetGain n={s['heterozygous_with_gain']['n']}  "
                  f"LOH n={s['loh_with_mutation']['n']}  "
                  f"biallelic n={s['biallelic_mutation']['n']}")
            wt_med = s['wildtype']['median_ln_ic50']
            for st in STATE_ORDER:
                if st in ('wildtype', 'unknown'): continue
                e = s[st]
                if e['p_vs_wt'] is not None and e['median_ln_ic50'] is not None:
                    delta = e['median_ln_ic50'] - wt_med if wt_med else 0
                    sig = '***' if e['p_vs_wt'] < 0.001 else (
                          '**' if e['p_vs_wt'] < 0.01 else (
                          '*' if e['p_vs_wt'] < 0.05 else ''))
                    print(f"    {st:<26} med ln(IC50)={e['median_ln_ic50']:.3f} "
                          f"(d={delta:+.3f})  p={e['p_vs_wt']:.2e} {sig}")
            summary.append({'class': class_name, 'drug': drug,
                            'n_mut': r['n_tp53_mut'], 'verdict': v})

    print("\n" + "=" * 84)
    print("SUMMARY -- buildable drugs by class")
    print("=" * 84)
    buildable = [r for r in summary if r['verdict'] == 'BUILDABLE']
    marginal  = [r for r in summary if r['verdict'] == 'marginal']
    by_class = {}
    for r in buildable:
        by_class.setdefault(r['class'], []).append(r['drug'])
    for cls, drugs_in_cls in by_class.items():
        print(f"  {cls}: {', '.join(drugs_in_cls)}")
    if not by_class:
        print("  (no drugs cleared the BUILDABLE threshold)")
    if marginal:
        print("\n  Marginal (n>=20 TP53-mut, state n>=5):")
        for r in marginal:
            print(f"    {r['class']}: {r['drug']} (n_mut={r['n_mut']})")


if __name__ == '__main__':
    main()
