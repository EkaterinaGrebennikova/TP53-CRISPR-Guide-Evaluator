"""Cancer-type-stratified LOH-vs-WT drug sensitivity analysis.

Tests whether the LOH-specific broad-resistance pattern (across MDM2i,
PARPi, ATRi, CHK1i, reactivators) survives adjustment for cancer-type
composition, or is driven by LOH cell lines concentrating in
intrinsically drug-resistant lineages.

For each focused drug:
  1. Per-lineage breakdown: n_LOH, n_WT, median ln(IC50) diff, MW p
  2. Lineage-adjusted: OLS ln(IC50) ~ is_LOH + C(OncotreeLineage),
     reports the LOH coefficient + p (this is the formal test).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, t as t_dist

from depmapdrugresponse import (
    load_tp53_mutations, load_tp53_cna, classify_cell_lines,
    _build_sanger_to_model_map, GDSC1_FILE, GDSC2_FILE, MODEL_FILE,
)

# Focused drug list: the headline drug per class with clean LOH effect.
FOCUS_DRUGS = [
    'Nutlin-3a (-)',          # MDM2 inhibitor (the original finding)
    'Serdemetan',             # MDM2 inhibitor (replication)
    'Tenovin-6',              # MDM2 inhibitor (replication)
    'PRIMA-1MET',             # mutant-p53 reactivator
    'AZD6738',                # ATR inhibitor
    'AZD7762',                # CHK1 inhibitor
    'Olaparib',               # PARP inhibitor
    'Niraparib',              # PARP inhibitor
    'Talazoparib',            # PARP inhibitor
]

MIN_PER_GROUP = 3   # minimum per (lineage, state) cell to run Mann-Whitney
MIN_LINEAGE_PAIR = 3   # require both LOH and WT n>=3 within a lineage


def load_lineage_map():
    m = pd.read_csv(MODEL_FILE, low_memory=False)
    # OncotreeLineage is the high-level grouping (Breast / Lung / etc.)
    return dict(zip(m['ModelID'], m['OncotreeLineage']))


def load_focus_drugs(focus_list):
    df2 = pd.read_csv(GDSC2_FILE); df2['screen'] = 'GDSC2'
    df1 = pd.read_csv(GDSC1_FILE); df1['screen'] = 'GDSC1'
    df = pd.concat([df1, df2], ignore_index=True)
    df = df[df['DRUG_NAME'].isin(focus_list)].copy()
    sanger = _build_sanger_to_model_map()
    df['ModelID'] = df['SANGER_MODEL_ID'].map(sanger)
    df = df.dropna(subset=['ModelID'])
    df = df.sort_values('screen', ascending=False).drop_duplicates(
        ['ModelID', 'DRUG_NAME'])
    return df[['ModelID', 'DRUG_NAME', 'LN_IC50', 'screen']].copy()


def analyse_drug(drug_name, drug_df, classified, lineage_map):
    sub = drug_df[drug_df['DRUG_NAME'] == drug_name]
    m = sub.merge(classified, on='ModelID', how='inner')
    m['lineage'] = m['ModelID'].map(lineage_map)
    m = m.dropna(subset=['lineage'])
    # restrict to WT + LOH (the focused contrast)
    m = m[m['allelic_state'].isin(['wildtype', 'loh_with_mutation'])].copy()
    m['is_loh'] = (m['allelic_state'] == 'loh_with_mutation').astype(int)

    # per-lineage breakdown
    rows = []
    for lin, g in m.groupby('lineage'):
        loh = g[g['is_loh'] == 1]['LN_IC50']
        wt = g[g['is_loh'] == 0]['LN_IC50']
        if len(loh) < MIN_PER_GROUP or len(wt) < MIN_PER_GROUP:
            continue
        _, p = mannwhitneyu(loh, wt, alternative='two-sided')
        rows.append({
            'lineage': lin, 'n_wt': len(wt), 'n_loh': len(loh),
            'med_wt': float(wt.median()),
            'med_loh': float(loh.median()),
            'delta': float(loh.median() - wt.median()), 'p': float(p),
        })
    per_lin = pd.DataFrame(rows).sort_values('p') if rows else pd.DataFrame()

    # lineage-adjusted regression: LN_IC50 ~ is_loh + C(lineage)
    # only use lineages with both LOH and WT representation
    valid_lins = [r['lineage'] for r in rows]
    if not valid_lins:
        return per_lin, None
    fit_df = m[m['lineage'].isin(valid_lins)].copy()
    # build design matrix manually: intercept + is_loh + lineage dummies
    # (drop one lineage as reference to avoid collinearity with intercept)
    dummies = pd.get_dummies(fit_df['lineage'], drop_first=True, dtype=float)
    X = np.column_stack([
        np.ones(len(fit_df)),
        fit_df['is_loh'].astype(float).values,
        dummies.values,
    ])
    y = fit_df['LN_IC50'].astype(float).values
    n, p_ = X.shape
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        sigma2 = float(resid @ resid) / max(n - p_, 1)
        xtx_inv = np.linalg.inv(X.T @ X)
        var_beta = sigma2 * np.diag(xtx_inv)
        se = np.sqrt(var_beta)
        # is_loh is column index 1
        coef = float(beta[1])
        se_loh = float(se[1])
        t_stat = coef / se_loh if se_loh > 0 else 0.0
        df = n - p_
        p_val = float(2 * (1 - t_dist.cdf(abs(t_stat), df=df)))
        crit = float(t_dist.ppf(0.975, df=df))
        ci_lo = coef - crit * se_loh
        ci_hi = coef + crit * se_loh
        adj = {
            'n': int(n), 'n_lineages': len(valid_lins),
            'coef': coef, 'se': se_loh, 'p': p_val,
            'ci_lo': ci_lo, 'ci_hi': ci_hi,
        }
    except Exception as e:
        adj = {'error': str(e)}
    return per_lin, adj


def main():
    print("Loading DepMap + classifying...")
    muts = load_tp53_mutations()
    cna = load_tp53_cna()
    classified = classify_cell_lines(muts, cna)
    lineage_map = load_lineage_map()

    print(f"  lines classified: {len(classified)}")
    print(f"  lineages in Model.csv: {len(set(lineage_map.values()) - {None})}")

    print("\nLoading focus drugs from GDSC1+GDSC2...")
    drugs = load_focus_drugs(FOCUS_DRUGS)
    found = sorted(drugs['DRUG_NAME'].unique())
    print(f"  drugs found: {len(found)} -> {found}")

    print("\n" + "=" * 84)
    print("CANCER-TYPE-STRATIFIED LOH vs WT")
    print("=" * 84)
    summary = []
    for drug in FOCUS_DRUGS:
        if drug not in found:
            continue
        per_lin, adj = analyse_drug(drug, drugs, classified, lineage_map)
        print(f"\n--- {drug} ---")
        if per_lin.empty:
            print("  no lineage with >=3 LOH and >=3 WT; skipped")
            continue
        # show per-lineage table (top by sample size first)
        per_lin_disp = per_lin.sort_values('n_loh', ascending=False).head(12)
        for _, r in per_lin_disp.iterrows():
            sig = '***' if r['p'] < 0.001 else ('**' if r['p'] < 0.01
                  else ('*' if r['p'] < 0.05 else ''))
            print(f"  {r['lineage']:<28} n_WT={int(r['n_wt']):>3}  "
                  f"n_LOH={int(r['n_loh']):>3}  d={r['delta']:+.3f}  "
                  f"p={r['p']:.2e} {sig}")
        # directional summary
        n_lin = len(per_lin)
        n_pos = int((per_lin['delta'] > 0).sum())
        n_sig_pos = int(((per_lin['delta'] > 0) & (per_lin['p'] < 0.05)).sum())
        n_sig_neg = int(((per_lin['delta'] < 0) & (per_lin['p'] < 0.05)).sum())
        print(f"  -> lineages tested: {n_lin}  |  "
              f"LOH > WT direction: {n_pos}/{n_lin}  |  "
              f"sig+ : {n_sig_pos}   sig- : {n_sig_neg}")
        if adj:
            tag = '***' if adj['p'] < 0.001 else ('**' if adj['p'] < 0.01
                  else ('*' if adj['p'] < 0.05 else ''))
            print(f"  -> LINEAGE-ADJUSTED LOH effect: "
                  f"coef={adj['coef']:+.3f} [{adj['ci_lo']:+.3f}, "
                  f"{adj['ci_hi']:+.3f}]  p={adj['p']:.3e} {tag}  "
                  f"(n={adj['n']}, lineages={adj['n_lineages']})")
            summary.append({
                'drug': drug, 'n_lin': n_lin, 'n_pos': n_pos,
                'n_sig_pos': n_sig_pos, 'n_sig_neg': n_sig_neg,
                'adj_coef': adj['coef'], 'adj_p': adj['p'],
            })

    print("\n" + "=" * 84)
    print("SUMMARY (lineage-adjusted LOH effect; positive = LOH resistant)")
    print("=" * 84)
    print(f"{'drug':<18}{'n_lin':>7}{'dir+':>8}{'sig+':>6}"
          f"{'sig-':>6}{'adj coef':>10}{'adj p':>12}")
    for r in summary:
        print(f"{r['drug']:<18}{r['n_lin']:>7}{r['n_pos']:>8}"
              f"{r['n_sig_pos']:>6}{r['n_sig_neg']:>6}"
              f"{r['adj_coef']:>+10.3f}{r['adj_p']:>12.2e}")


if __name__ == '__main__':
    main()
