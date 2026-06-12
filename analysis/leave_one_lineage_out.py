"""Leave-one-lineage-out (LOO) sensitivity for the §3.7 headline drugs.

Tests whether the cancer-type-adjusted LOH coefficient survives when each
OncotreeLineage is dropped in turn. Preempts the homogeneity-assumption
reviewer attack: if the LOH effect is being driven by a single lineage, the
worst-case LOO will reveal it; if it's distributed across lineages, the
LOO coefficient range will be tight around the baseline.

Drugs tested (mirrors paper §3.7):
  - 5 BH-significant DNA-damaging chemo agents
  - Nutlin-3a (canonical MDM2i)

Output: per-drug baseline coefficient + LOO min/max range + worst-case
lineage + worst-case p-value. Intended for supplementary table.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from depmapdrugresponse import (
    load_tp53_mutations, load_tp53_cna, classify_cell_lines,
)
from gdsc_cancer_type_stratified import load_lineage_map, analyse_drug
from gdsc_drug_class_audit import load_all_drugs

# (GDSC drug name, display label, class)
HEADLINE_DRUGS = [
    ('5-Fluorouracil', '5-FU',         'chemo'),
    ('Gemcitabine',    'Gemcitabine',  'chemo'),
    ('Mitomycin-C',    'Mitomycin-C',  'chemo'),
    ('Doxorubicin',    'Doxorubicin',  'chemo'),
    ('Cytarabine',     'Cytarabine',   'chemo'),
    ('Nutlin-3a (-)',  'Nutlin-3a',    'mdm2i'),
]


def baseline(drug, drugs_all, classified, lineage_map):
    """Run OLS on the full panel; return (coef, ci_lo, ci_hi, p, valid_lins)."""
    per_lin, adj = analyse_drug(drug, drugs_all, classified, lineage_map)
    if adj is None or 'error' in adj or per_lin.empty:
        return None
    return {
        'coef': adj['coef'], 'ci_lo': adj['ci_lo'], 'ci_hi': adj['ci_hi'],
        'p': adj['p'], 'n_lineages': adj['n_lineages'],
        'lineages': list(per_lin['lineage']),
    }


def loo(drug, drugs_all, classified, lineage_map, valid_lins):
    """Run OLS excluding each valid lineage in turn; return list of results."""
    results = []
    for excl in valid_lins:
        lin_map_filtered = {k: v for k, v in lineage_map.items() if v != excl}
        _, adj = analyse_drug(drug, drugs_all, classified, lin_map_filtered)
        if adj is None or 'error' in adj:
            continue
        results.append({
            'excluded': excl, 'coef': adj['coef'], 'p': adj['p'],
            'ci_lo': adj['ci_lo'], 'ci_hi': adj['ci_hi'],
        })
    return results


def main():
    print("Loading DepMap + classifying...")
    muts = load_tp53_mutations()
    cna = load_tp53_cna()
    classified = classify_cell_lines(muts, cna)
    lineage_map = load_lineage_map()
    drugs_all = load_all_drugs()

    print("\n" + "=" * 96)
    print("LEAVE-ONE-LINEAGE-OUT SENSITIVITY — paper §3.7 headline drugs")
    print("=" * 96)
    print(f"{'drug':<14}{'class':<8}{'baseline coef':>14}"
          f"{'baseline p':>14}{'LOO coef range':>18}"
          f"{'worst-case lineage (max p)':>30}")
    print("-" * 96)

    for drug_name, label, klass in HEADLINE_DRUGS:
        base = baseline(drug_name, drugs_all, classified, lineage_map)
        if base is None:
            print(f"{label:<14}{klass:<8}  [skip] insufficient coverage")
            continue

        loo_rows = loo(drug_name, drugs_all, classified, lineage_map,
                       base['lineages'])
        if not loo_rows:
            print(f"{label:<14}{klass:<8}  [skip] no LOO results")
            continue

        coefs = [r['coef'] for r in loo_rows]
        ps = [r['p'] for r in loo_rows]
        worst = max(loo_rows, key=lambda r: r['p'])
        coef_range = f"[{min(coefs):+.2f}, {max(coefs):+.2f}]"
        worst_desc = f"{worst['excluded'][:18]:<18} (p={worst['p']:.2e})"
        print(f"{label:<14}{klass:<8}{base['coef']:>+14.3f}"
              f"{base['p']:>14.2e}{coef_range:>18}{worst_desc:>30}")

    # detailed per-drug per-lineage report
    print("\n" + "=" * 96)
    print("PER-LINEAGE DETAIL (for supplement)")
    print("=" * 96)
    for drug_name, label, klass in HEADLINE_DRUGS:
        base = baseline(drug_name, drugs_all, classified, lineage_map)
        if base is None:
            continue
        loo_rows = loo(drug_name, drugs_all, classified, lineage_map,
                       base['lineages'])
        print(f"\n--- {label} ({klass}) | baseline coef={base['coef']:+.3f}, "
              f"p={base['p']:.2e}, n_lineages={base['n_lineages']} ---")
        print(f"  {'excluded lineage':<25}{'coef':>9}{'95% CI':>20}{'p':>11}")
        for r in sorted(loo_rows, key=lambda x: x['coef']):
            ci = f"[{r['ci_lo']:+.2f},{r['ci_hi']:+.2f}]"
            print(f"  {r['excluded'][:24]:<25}{r['coef']:>+9.3f}"
                  f"{ci:>20}{r['p']:>11.2e}")


if __name__ == '__main__':
    main()
