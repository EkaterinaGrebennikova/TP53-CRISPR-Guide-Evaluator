"""Cancer-type-adjusted LOH-vs-WT analysis for conventional chemotherapy.

p53 mediates DNA-damage-induced apoptosis, so LOH could confer
chemoresistance. Chemo is standard-of-care for TP53-mutant patients,
making it more clinically relevant than the targeted agents. Same
rigorous lineage-adjusted test applied in gdsc_cancer_type_stratified.py.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd

from depmapdrugresponse import (
    load_tp53_mutations, load_tp53_cna, classify_cell_lines,
)
from gdsc_cancer_type_stratified import load_lineage_map, analyse_drug
from gdsc_drug_class_audit import load_all_drugs, find_matches

CHEMO_TERMS = [
    'cisplatin', 'carboplatin', 'oxaliplatin',
    'doxorubicin', 'epirubicin', 'daunorubicin', 'mitoxantrone',
    'etoposide', 'teniposide',
    'paclitaxel', 'docetaxel', 'vinblastine', 'vincristine', 'vinorelbine',
    'gemcitabine', 'cytarabine', 'fluorouracil', '5-fu',
    'temozolomide', 'dacarbazine',
    'irinotecan', 'sn-38', 'topotecan', 'camptothecin',
    'methotrexate', 'bleomycin', 'mitomycin',
]


def main():
    print("Loading DepMap + classifying...")
    muts = load_tp53_mutations()
    cna = load_tp53_cna()
    classified = classify_cell_lines(muts, cna)
    lineage_map = load_lineage_map()

    print("Loading all GDSC drugs + matching chemo agents...")
    drugs_all = load_all_drugs()
    names = sorted(drugs_all['DRUG_NAME'].unique())
    matches = find_matches(names, CHEMO_TERMS)
    print(f"  matched chemo agents ({len(matches)}): {matches}")

    print("\n" + "=" * 84)
    print("CANCER-TYPE-ADJUSTED LOH vs WT  --  CONVENTIONAL CHEMOTHERAPY")
    print("=" * 84)
    summary = []
    for drug in matches:
        per_lin, adj = analyse_drug(drug, drugs_all, classified, lineage_map)
        if per_lin.empty or adj is None or 'error' in adj:
            print(f"\n--- {drug} ---  insufficient lineage coverage; skipped")
            continue
        n_lin = len(per_lin)
        n_pos = int((per_lin['delta'] > 0).sum())
        n_sig_pos = int(((per_lin['delta'] > 0) & (per_lin['p'] < 0.05)).sum())
        n_sig_neg = int(((per_lin['delta'] < 0) & (per_lin['p'] < 0.05)).sum())
        tag = '***' if adj['p'] < 0.001 else ('**' if adj['p'] < 0.01
              else ('*' if adj['p'] < 0.05 else ''))
        print(f"\n--- {drug} ---")
        print(f"  lineages tested: {n_lin}  |  LOH>WT dir: {n_pos}/{n_lin}  "
              f"|  sig+: {n_sig_pos}  sig-: {n_sig_neg}")
        print(f"  LINEAGE-ADJUSTED LOH effect: coef={adj['coef']:+.3f} "
              f"[{adj['ci_lo']:+.3f}, {adj['ci_hi']:+.3f}]  "
              f"p={adj['p']:.3e} {tag}  (n={adj['n']}, lineages={adj['n_lineages']})")
        summary.append({'drug': drug, 'n_lin': n_lin, 'n_pos': n_pos,
                        'n_sig_pos': n_sig_pos, 'n_sig_neg': n_sig_neg,
                        'coef': adj['coef'], 'p': adj['p']})

    print("\n" + "=" * 84)
    print("SUMMARY (lineage-adjusted LOH effect; positive = LOH chemoresistant)")
    print("=" * 84)
    print(f"{'drug':<24}{'n_lin':>6}{'dir+':>7}{'sig+':>6}{'sig-':>6}"
          f"{'adj coef':>10}{'adj p':>12}")
    for r in sorted(summary, key=lambda x: x['p']):
        print(f"{r['drug']:<24}{r['n_lin']:>6}{r['n_pos']:>7}"
              f"{r['n_sig_pos']:>6}{r['n_sig_neg']:>6}"
              f"{r['coef']:>+10.3f}{r['p']:>12.2e}")
    sig = [r for r in summary if r['p'] < 0.05 and r['coef'] > 0]
    print(f"\n  Chemo agents with significant LOH-resistance after "
          f"cancer-type adjustment: {len(sig)}/{len(summary)}")
    for r in sig:
        print(f"    {r['drug']}: coef={r['coef']:+.3f}, p={r['p']:.2e}")


if __name__ == '__main__':
    main()
