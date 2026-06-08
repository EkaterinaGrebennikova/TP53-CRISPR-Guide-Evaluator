"""BH-corrected LOH-vs-WT analysis + forest plot for targeted drug classes.

Mirrors gdsc_chemo_figure.py for the targeted (non-cytotoxic) panel:
MDM2 inhibitors, mutant-p53 reactivator, and synthetic-lethality agents
(PARP / ATR / CHK1). Lineage-adjusted OLS coefficients with 95% CIs,
Benjamini-Hochberg applied across the targeted panel. Reports BH-adjusted
p-values throughout (paper convention: write "p" everywhere, mention BH
once in methods).

Output: figures/targeted_loh_resistance.png  (becomes Fig 6b)
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from depmapdrugresponse import (
    load_tp53_mutations, load_tp53_cna, classify_cell_lines,
)
from survivalanalysis import benjamini_hochberg
from gdsc_cancer_type_stratified import (
    load_lineage_map, load_focus_drugs, analyse_drug, FOCUS_DRUGS,
)

# (GDSC drug name, display label, category)
# A = MDM2 inhibitors, C = synthetic lethality
# Note: mutant-p53 reactivators (e.g. PRIMA-1MET) are mutation-class-specific
# (Bykov 2018) and were not tested as a pooled LOH-vs-WT contrast; see §3.7
# prose.
DRUGS = [
    ('Nutlin-3a (-)', 'Nutlin-3a',   'A'),
    ('Serdemetan',    'Serdemetan',  'A'),
    ('Tenovin-6',     'Tenovin-6',   'A'),
    ('Olaparib',      'Olaparib',    'C'),
    ('Niraparib',     'Niraparib',   'C'),
    ('Talazoparib',   'Talazoparib', 'C'),
    ('AZD6738',       'AZD6738 (ATR)',  'C'),
    ('AZD7762',       'AZD7762 (CHK1)', 'C'),
]

CAT_LABEL = {
    'A': 'MDM2 inhibitors',
    'C': 'Synthetic lethality (PARP / ATR / CHK1)',
}
CAT_COLOR = {'A': '#6A1B9A', 'C': '#00838F'}


def main():
    muts = load_tp53_mutations()
    cna = load_tp53_cna()
    classified = classify_cell_lines(muts, cna)
    lineage_map = load_lineage_map()
    drugs_all = load_focus_drugs(FOCUS_DRUGS)

    rows = []
    for name, label, cat in DRUGS:
        per_lin, adj = analyse_drug(name, drugs_all, classified, lineage_map)
        if adj is None or 'error' in adj or per_lin.empty:
            print(f"  [skip] {name}: insufficient coverage")
            continue
        rows.append({'drug': name, 'label': label, 'cat': cat,
                     'coef': adj['coef'], 'lo': adj['ci_lo'],
                     'hi': adj['ci_hi'], 'p': adj['p'], 'n': adj['n']})

    # BH across the targeted panel
    pvals = [r['p'] for r in rows]
    bh = benjamini_hochberg(pvals)
    for r, q in zip(rows, bh):
        r['p_bh'] = q   # paper writes this as "p" (BH already applied)

    # ---- table ----
    print("\n" + "=" * 80)
    print("TARGETED PANEL: LOH-vs-WT (lineage-adjusted, BH across panel)")
    print("=" * 80)
    print(f"{'drug':<18}{'cat':>4}{'coef':>10}{'95% CI':>20}"
          f"{'raw p':>11}{'p (BH)':>11}")
    for r in sorted(rows, key=lambda x: (x['cat'], x['p_bh'])):
        sig = '***' if r['p_bh'] < 0.001 else ('**' if r['p_bh'] < 0.01
              else ('*' if r['p_bh'] < 0.05 else ''))
        ci = f"[{r['lo']:+.2f},{r['hi']:+.2f}]"
        print(f"{r['label']:<18}{r['cat']:>4}{r['coef']:>+10.3f}{ci:>20}"
              f"{r['p']:>11.2e}{r['p_bh']:>11.2e} {sig}")

    # ---- forest figure ----
    # Order: MDM2i (top), Synth-leth (bottom). Within class, sort by coefficient.
    order = []
    for cat in ('A', 'C'):
        cls = sorted([r for r in rows if r['cat'] == cat],
                     key=lambda x: x['coef'])
        if order:
            order.append(None)          # spacer between groups
        order.extend(cls)

    ypos, labels, used = [], [], []
    y = 0
    for r in order:
        if r is None:
            y += 1; continue
        ypos.append(y); labels.append(r['label']); used.append(r); y += 1

    fig, ax = plt.subplots(figsize=(9, 7.2))
    for yp, r in zip(ypos, used):
        col = CAT_COLOR[r['cat']]
        filled = r['p_bh'] < 0.05
        ax.plot([r['lo'], r['hi']], [yp, yp], color=col, lw=2, zorder=2)
        ax.scatter([r['coef']], [yp],
                   s=70, color=col if filled else 'white',
                   edgecolors=col, linewidths=1.6, zorder=3)
        star = ' *' if filled else ''
        ax.text(r['hi'] + 0.08, yp,
                f"p = {r['p_bh']:.2g}{star}", va='center',
                fontsize=8, color='#444')

    ax.axvline(0, color='black', lw=1, ls='--')
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Lineage-adjusted LOH effect on ln(IC50)\n'
                  '(positive → LOH-mutant cells more resistant)', fontsize=10)
    ax.set_title('Only canonical MDM2 inhibition shows an allelic-state '
                 'effect\nafter cancer-type adjustment in the targeted panel',
                 fontsize=12, fontweight='bold')

    legend = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=CAT_COLOR['A'], markeredgecolor=CAT_COLOR['A'],
               markersize=9, label=CAT_LABEL['A']),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=CAT_COLOR['C'], markeredgecolor=CAT_COLOR['C'],
               markersize=9, label=CAT_LABEL['C']),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#555',
               markeredgecolor='#555', markersize=9,
               label='filled / *  =  significant after BH (p < 0.05)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
               markeredgecolor='#888', markersize=9,
               label='open marker  =  not significant after BH'),
    ]
    ax.legend(handles=legend, loc='lower right', fontsize=8, framealpha=0.95)
    ax.set_xlim(min(r['lo'] for r in used) - 0.4,
                max(r['hi'] for r in used) + 1.6)
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), '..', 'figures',
                       'targeted_loh_resistance.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved {out}")


if __name__ == '__main__':
    main()
