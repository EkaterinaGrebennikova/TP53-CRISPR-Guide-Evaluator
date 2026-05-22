"""Allelic state x editing modality jointly determine CRISPR rescuability.

Composes three components already in the tool:
  1. Tetramer model (tetramodel.py / allelemodel.py): functional p53 requires
     a fraction of wild-type monomers; functional tetramer fraction = f**n
     where f is the WT-monomer fraction and n the assembly exponent.
  2. Allelic baseline (allelemodel.py): the WT-monomer fraction reachable by
     correction depends on allelic state --
        het CN-neutral : f = 0.5 + 0.5*e   (one intact WT allele as baseline)
        LOH / biallelic: f = e             (no WT baseline; all from correction)
  3. The empirical ML-predicted editing-efficiency distribution per modality
     (efficiencypredictorml via the guide-design pipeline).

Functional threshold: tetramer fraction >= 0.45 (Ventura et al. 2007).
Solving f**n >= 0.45 for the editing efficiency e gives a per-allelic-state
efficiency requirement. Overlaying the real per-modality efficiency
distribution shows what fraction of designed guides actually clears it.

Robustness: the assembly exponent is reported for n = 2, 3, 4. The
qualitative cliff (LOH requires substantially higher efficiency than het, and
CBE-correctable mutations clear neither) must persist across all three or the
conclusion is exponent-dependent and should not be claimed.

This analysis uses allelic state as a *category* and the tetramer-baseline
math; it does NOT use the VAF-gated LOH classifier or any mutation-type x LOH
contingency, so it is structurally immune to the indel-VAF calling artifact.
"""
import numpy as np
import pandas as pd

from mutationparser import parse_mutations
from mutationevaluator import evaluate_mutations
from strategybuilder import build_strategies
from tcgaloader import load_mutations

VENTURA_THRESHOLD = 0.45
EXPONENTS = (2, 3, 4)


def efficiency_thresholds(threshold=VENTURA_THRESHOLD):
    """Editing efficiency required to reach the functional tetramer threshold,
    per allelic state, for each assembly exponent.

    Returns {n: {'het_cn_neutral': e*, 'loh_or_biallelic': e*, 'f_star': f*}}.
    """
    out = {}
    for n in EXPONENTS:
        f_star = threshold ** (1.0 / n)        # required WT-monomer fraction
        het_e = max(0.0, 2.0 * (f_star - 0.5))  # f = 0.5 + 0.5 e
        loh_e = f_star                          # f = e
        out[n] = {
            'f_star': f_star,
            'het_cn_neutral': het_e,
            'loh_or_biallelic': loh_e,
        }
    return out


def collect_ml_efficiencies(mutation_list):
    """Run the guide-design pipeline once and collect, per modality:
      - all_guides:  every ML-predicted efficiency (the landscape)
      - best_per_mut: the single best ML efficiency per mutation (deployed case)
    Returns (by_mod_all, by_mod_best), each {modality: np.array([...])}.
    """
    muts = parse_mutations(mutation_list)
    evals = evaluate_mutations(muts)
    strats = build_strategies(muts, evals)
    by_all, by_best = {}, {}
    for s in strats:
        per_mut = {}
        for cand in s.get('all_guides', []):
            ml = cand['guide'].get('ml_efficiency')
            if ml is None:
                continue
            mod = cand['modality']
            by_all.setdefault(mod, []).append(float(ml))
            if mod not in per_mut or ml > per_mut[mod]:
                per_mut[mod] = float(ml)
        for mod, best in per_mut.items():
            by_best.setdefault(mod, []).append(best)
    return ({m: np.array(v) for m, v in by_all.items()},
            {m: np.array(v) for m, v in by_best.items()})


def recurrent_missense_panel(min_recurrence=2, top_n=150):
    """Most frequently recurrent TP53 missense mutations from TCGA."""
    m = load_mutations()
    vc = m[m['Variant_Classification'] == 'Missense_Mutation']['aa_change'].value_counts()
    vc = vc[vc >= min_recurrence]
    return list(vc.head(top_n).index)


def rescuability_table(by_mod, thresholds):
    """For each (modality, allelic state, exponent), the fraction of designed
    guides whose ML-predicted efficiency clears the required threshold.

    Returns a tidy DataFrame.
    """
    rows = []
    for modality, effs in by_mod.items():
        if len(effs) == 0:
            continue
        for n in EXPONENTS:
            t = thresholds[n]
            rows.append({
                'modality': modality,
                'n_guides': len(effs),
                'median_eff': float(np.median(effs)),
                'exponent': n,
                'het_thresh': t['het_cn_neutral'],
                'loh_thresh': t['loh_or_biallelic'],
                'pct_clear_het': 100.0 * np.mean(effs >= t['het_cn_neutral']),
                'pct_clear_loh': 100.0 * np.mean(effs >= t['loh_or_biallelic']),
            })
    return pd.DataFrame(rows)


if __name__ == '__main__':
    print("Efficiency thresholds (editing efficiency to reach 0.45 tetramer):")
    th = efficiency_thresholds()
    print(f"  {'exponent':>8} {'f*':>7} {'het CN-neutral':>16} {'LOH/biallelic':>15}")
    for n in EXPONENTS:
        t = th[n]
        print(f"  {n:>8} {t['f_star']:>7.3f} {t['het_cn_neutral']:>16.3f} "
              f"{t['loh_or_biallelic']:>15.3f}")
    print()

    panel = recurrent_missense_panel(top_n=150)
    print(f"Designing guides for {len(panel)} recurrent missense mutations...")
    by_all, by_best = collect_ml_efficiencies(panel)

    pd.set_option('display.width', 120)
    fmt = {'median_eff': '{:.3f}'.format, 'het_thresh': '{:.3f}'.format,
           'loh_thresh': '{:.3f}'.format, 'pct_clear_het': '{:.1f}'.format,
           'pct_clear_loh': '{:.1f}'.format}

    print("\n--- All designed guides (efficiency landscape) ---")
    for m, v in sorted(by_all.items()):
        print(f"  {m}: {len(v)} guides, median ML eff = {np.median(v):.3f}")
    print(rescuability_table(by_all, th).to_string(index=False, formatters=fmt))

    print("\n--- Best guide per mutation (deployed scenario; "
          "n = mutations rescuable) ---")
    for m, v in sorted(by_best.items()):
        print(f"  {m}: {len(v)} mutations, median best ML eff = {np.median(v):.3f}")
    print(rescuability_table(by_best, th).to_string(index=False, formatters=fmt))
