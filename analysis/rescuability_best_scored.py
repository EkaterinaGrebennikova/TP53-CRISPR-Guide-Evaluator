"""Does the rescuability picture change if we deploy the best COMPOSITE-SCORED
guide instead of the most EFFICIENT guide?

best-by-efficiency : per (mutation, modality) take max ml_efficiency
best-by-score      : per (mutation, modality) take the candidate with the
                     highest composite score
                       composite = score_guide*0.6 + score_offtarget*0.4
                     (score_guide already folds in ML eff, bystander, GC),
                     then read THAT guide's ml_efficiency for the tetramer calc.

If safety/specificity-optimal guides sacrifice the efficiency needed to clear
the 0.45 tetramer threshold, best-by-score clears less than best-by-efficiency
-- and the gap is the real clinical cost of multi-objective guide selection.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd

from mutationparser import parse_mutations
from mutationevaluator import evaluate_mutations
from strategybuilder import build_strategies
from rescuability import (
    efficiency_thresholds, recurrent_missense_panel, EXPONENTS,
)


def collect_two_rules(mutation_list):
    """Per (mutation, modality) return ml_efficiency under both selection
    rules. Returns {modality: {'eff': np.array, 'score': np.array}} where
    'eff' = best-by-efficiency, 'score' = best-by-composite-score."""
    muts = parse_mutations(mutation_list)
    evals = evaluate_mutations(muts)
    strats = build_strategies(muts, evals)

    by_eff, by_score = {}, {}
    for s in strats:
        # group this mutation's candidates by modality
        per_mod = {}
        for cand in s.get('all_guides', []):
            ml = cand['guide'].get('ml_efficiency')
            if ml is None:
                continue
            per_mod.setdefault(cand['modality'], []).append(
                (float(ml), float(cand['score'])))
        for mod, pairs in per_mod.items():
            best_eff = max(p[0] for p in pairs)
            # ml_efficiency of the candidate with the highest composite score
            best_eff_at_best_score = max(pairs, key=lambda p: p[1])[0]
            by_eff.setdefault(mod, []).append(best_eff)
            by_score.setdefault(mod, []).append(best_eff_at_best_score)
    return ({m: np.array(v) for m, v in by_eff.items()},
            {m: np.array(v) for m, v in by_score.items()})


def clear_table(by_mod, thresholds, rule_label):
    rows = []
    for modality, effs in sorted(by_mod.items()):
        if len(effs) == 0:
            continue
        for n in EXPONENTS:
            t = thresholds[n]
            rows.append({
                'rule': rule_label,
                'modality': modality,
                'n_mut': len(effs),
                'median_eff': float(np.median(effs)),
                'k': n,
                'pct_clear_het': 100.0 * np.mean(effs >= t['het_cn_neutral']),
                'pct_clear_loh': 100.0 * np.mean(effs >= t['loh_or_biallelic']),
            })
    return pd.DataFrame(rows)


if __name__ == '__main__':
    th = efficiency_thresholds()
    panel = recurrent_missense_panel(top_n=150)
    print(f"Designing guides for {len(panel)} recurrent missense mutations...")
    by_eff, by_score = collect_two_rules(panel)

    fmt = {'median_eff': '{:.3f}'.format,
           'pct_clear_het': '{:.1f}'.format,
           'pct_clear_loh': '{:.1f}'.format}

    te = clear_table(by_eff, th, 'best-by-efficiency')
    ts = clear_table(by_score, th, 'best-by-score')
    both = pd.concat([te, ts], ignore_index=True)

    print("\n=== Rescuability under both guide-selection rules ===")
    print("(pct = % of mutations whose deployed guide clears 0.45 tetramer)\n")
    for mod in sorted(both['modality'].unique()):
        sub = both[both['modality'] == mod].sort_values(['k', 'rule'])
        print(f"--- {mod} ---")
        print(sub[['rule', 'k', 'n_mut', 'median_eff',
                   'pct_clear_het', 'pct_clear_loh']]
              .to_string(index=False, formatters=fmt))
        print()

    # headline: the efficiency cost of choosing the safest guide
    print("=== Cost of composite-optimal selection (median eff drop) ===")
    for mod in sorted(set(by_eff) & set(by_score)):
        de = np.median(by_eff[mod])
        ds = np.median(by_score[mod])
        print(f"  {mod:14s}: best-eff median={de:.3f}  "
              f"best-score median={ds:.3f}  drop={de-ds:+.3f}")
