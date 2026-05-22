"""Circularity / winner's-curse check on the rescuability cliff.

The pipeline selects the best guide per mutation using ML-predicted
efficiency (directly, or via the composite score that is ~30% ML), then
reports a HARD pass/fail: 1[selected_eff >= threshold]. But the selected
efficiency is a prediction with held-out residual SD ~0.15. Near a hard
threshold that indicator is optimistic: a prediction of 0.83 against a
0.82 threshold is "pass" but its true value is 0.83 +/- 0.15.

GBM predictions are ~calibrated conditional means on held-out data
(resid_mean ~ 0: ABE -0.006, CBE -0.004), so the model-consistent honest
quantity is P(true_eff >= threshold | prediction) = Phi((p - t)/sigma),
not the hard indicator. This propagates the model's OWN error; it adds no
new assumption. Comparing naive (hard) vs honest (Phi) clear% per cell
shows how much of the cliff is selection-on-noise optimism.

sigma is the per-modality held-out residual SD.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
from scipy.stats import norm

from rescuability import efficiency_thresholds, recurrent_missense_panel, EXPONENTS
from rescuability_best_scored import collect_two_rules

# held-out residual SD (see residual extraction step)
SIGMA = {'ABE': 0.156, 'CBE': 0.151, 'Prime Editing': 0.151}


def cell_rows(by_mod, thresholds, rule_label):
    rows = []
    for modality, eff in sorted(by_mod.items()):
        if len(eff) == 0:
            continue
        sig = SIGMA.get(modality, 0.155)
        for n in EXPONENTS:
            t = thresholds[n]
            for state, thr in (('het', t['het_cn_neutral']),
                               ('loh', t['loh_or_biallelic'])):
                naive = 100.0 * np.mean(eff >= thr)
                # honest: mean P(true >= thr | prediction) = mean Phi((p-thr)/sig)
                honest = 100.0 * np.mean(norm.cdf((eff - thr) / sig))
                rows.append({
                    'rule': rule_label, 'modality': modality, 'state': state,
                    'k': n, 'n_mut': len(eff),
                    'naive_clear': naive, 'honest_clear': honest,
                    'optimism': naive - honest,
                })
    return pd.DataFrame(rows)


if __name__ == '__main__':
    th = efficiency_thresholds()
    panel = recurrent_missense_panel(top_n=150)
    print(f"Designing guides for {len(panel)} recurrent missense mutations...")
    by_eff, by_score = collect_two_rules(panel)

    df = pd.concat([
        cell_rows(by_eff, th, 'best-by-efficiency'),
        cell_rows(by_score, th, 'best-by-score'),
    ], ignore_index=True)

    fmt = {'naive_clear': '{:.1f}'.format, 'honest_clear': '{:.1f}'.format,
           'optimism': '{:+.1f}'.format}

    print("\n=== Naive (hard threshold) vs Honest (P[true>=thr|pred]) ===")
    for rule in ['best-by-efficiency', 'best-by-score']:
        for mod in sorted(df['modality'].unique()):
            sub = df[(df['rule'] == rule) & (df['modality'] == mod)]
            if sub.empty:
                continue
            print(f"\n--- {rule} | {mod} (sigma={SIGMA.get(mod):.3f}, "
                  f"n={sub['n_mut'].iloc[0]}) ---")
            print(sub[['state', 'k', 'naive_clear', 'honest_clear',
                       'optimism']].to_string(index=False, formatters=fmt))

    print("\n=== Does the qualitative cliff survive de-biasing? ===")
    for rule in ['best-by-efficiency', 'best-by-score']:
        for n in EXPONENTS:
            def cell(mod, st):
                r = df[(df.rule == rule) & (df.modality == mod)
                       & (df.state == st) & (df.k == n)]
                return r['honest_clear'].iloc[0] if not r.empty else float('nan')
            abe_loh = cell('ABE', 'loh')
            cbe_het = cell('CBE', 'het')
            cbe_loh = cell('CBE', 'loh')
            ordered = (abe_loh > cbe_het >= cbe_loh)
            print(f"  {rule:18s} k={n}: honest ABE+loh={abe_loh:.1f}  "
                  f"CBE+het={cbe_het:.1f}  CBE+loh={cbe_loh:.1f}  "
                  f"ordering_holds={ordered}  het>loh(CBE)={cbe_het> cbe_loh}")
