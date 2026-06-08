"""BH-corrected chemo LOH-resistance analysis + mechanism-grouped figure.

Deduplicates to canonical single agents (drops dose-variants and combos),
runs the lineage-adjusted LOH-vs-WT regression per drug, applies
Benjamini-Hochberg across the panel, and renders a forest plot grouped
by mechanism (p53-dependent DNA-damaging vs p53-independent).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt

from depmapdrugresponse import (
    load_tp53_mutations, load_tp53_cna, classify_cell_lines,
)
from survivalanalysis import benjamini_hochberg
from gdsc_cancer_type_stratified import load_lineage_map, analyse_drug
from gdsc_drug_class_audit import load_all_drugs

# canonical GDSC single-agent name -> (mechanism, category)
# category A = DNA-damaging (p53-apoptosis-dependent)
# category B = p53-independent (anti-mitotic / MGMT-dependent)
DRUGS = [
    ('Gemcitabine',    'antimetabolite',      'A'),
    ('5-Fluorouracil', 'antimetabolite',      'A'),
    ('Cytarabine',     'antimetabolite',      'A'),
    ('Methotrexate',   'antifolate',          'A'),
    ('Doxorubicin',    'anthracycline',       'A'),
    ('Epirubicin',     'anthracycline',       'A'),
    ('Mitoxantrone',   'anthracenedione',     'A'),
    ('Cisplatin',      'platinum',            'A'),
    ('Oxaliplatin',    'platinum',            'A'),
    ('Mitomycin-C',    'crosslinker',         'A'),
    ('Camptothecin',   'topoisomerase I',     'A'),
    ('Topotecan',      'topoisomerase I',     'A'),
    ('Irinotecan',     'topoisomerase I',     'A'),
    ('SN-38',          'topoisomerase I',     'A'),
    ('Etoposide',      'topoisomerase II',    'A'),
    ('Teniposide',     'topoisomerase II',    'A'),
    ('Bleomycin',      'strand break',        'A'),
    ('Paclitaxel',     'microtubule',         'B'),
    ('Docetaxel',      'microtubule',         'B'),
    ('Vinblastine',    'vinca',               'B'),
    ('Vinorelbine',    'vinca',               'B'),
    ('Temozolomide',   'alkylator (MGMT-dep)', 'B'),
    ('Dacarbazine',    'alkylator (MGMT-dep)', 'B'),
]

CAT_LABEL = {
    'A': 'DNA-damaging (p53-dependent apoptosis)',
    'B': 'p53-independent (anti-mitotic / MGMT-dependent)',
}
CAT_COLOR = {'A': '#C62828', 'B': '#1565C0'}


def main():
    muts = load_tp53_mutations()
    cna = load_tp53_cna()
    classified = classify_cell_lines(muts, cna)
    lineage_map = load_lineage_map()
    drugs_all = load_all_drugs()
    avail = set(drugs_all['DRUG_NAME'].unique())

    rows = []
    for name, mech, cat in DRUGS:
        if name not in avail:
            print(f"  [skip] {name} not in GDSC")
            continue
        per_lin, adj = analyse_drug(name, drugs_all, classified, lineage_map)
        if adj is None or 'error' in adj or per_lin.empty:
            print(f"  [skip] {name}: insufficient coverage")
            continue
        rows.append({'drug': name, 'mech': mech, 'cat': cat,
                     'coef': adj['coef'], 'lo': adj['ci_lo'],
                     'hi': adj['ci_hi'], 'p': adj['p'], 'n': adj['n']})

    # BH correction across the panel
    pvals = [r['p'] for r in rows]
    bh = benjamini_hochberg(pvals)
    for r, q in zip(rows, bh):
        r['bh'] = q

    # ---- table ----
    print("\n" + "=" * 80)
    print("CHEMO LOH-RESISTANCE (lineage-adjusted) with BH correction")
    print("=" * 80)
    print(f"{'drug':<16}{'cat':>4}{'mech':<20}{'coef':>8}{'p':>11}{'BH q':>11}")
    for r in sorted(rows, key=lambda x: x['bh']):
        sig = '***' if r['bh'] < 0.001 else ('**' if r['bh'] < 0.01
              else ('*' if r['bh'] < 0.05 else ''))
        print(f"{r['drug']:<16}{r['cat']:>4} {r['mech']:<19}"
              f"{r['coef']:>+8.3f}{r['p']:>11.2e}{r['bh']:>11.2e} {sig}")
    n_bh = sum(1 for r in rows if r['bh'] < 0.05)
    print(f"\n  BH-significant (q<0.05): {n_bh}/{len(rows)}")
    print("  survivors:", ", ".join(r['drug'] for r in rows if r['bh'] < 0.05))

    # ---- forest figure ----
    order = (sorted([r for r in rows if r['cat'] == 'B'],
                    key=lambda x: x['coef']) +
             [None] +  # spacer between groups
             sorted([r for r in rows if r['cat'] == 'A'],
                    key=lambda x: x['coef']))
    ypos, labels, used = [], [], []
    y = 0
    for r in order:
        if r is None:
            y += 1; continue
        ypos.append(y); labels.append(r['drug']); used.append(r); y += 1

    fig, ax = plt.subplots(figsize=(9, 9))
    for yp, r in zip(ypos, used):
        col = CAT_COLOR[r['cat']]
        filled = r['bh'] < 0.05
        ax.plot([r['lo'], r['hi']], [yp, yp], color=col, lw=2, zorder=2)
        ax.scatter([r['coef']], [yp], s=70, color=col if filled else 'white',
                   edgecolors=col, linewidths=1.6, zorder=3)
        star = ' *' if r['bh'] < 0.05 else ''
        ax.text(r['hi'] + 0.08, yp, f"{r['mech']}{star}", va='center',
                fontsize=8, color='#444')
    ax.axvline(0, color='black', lw=1, ls='--')
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Lineage-adjusted LOH effect on ln(IC50)\n'
                  '(positive → LOH-mutant cells more resistant)', fontsize=10)
    ax.set_title('TP53 LOH confers resistance to p53-dependent DNA-damaging\n'
                 'chemotherapy, but not p53-independent agents',
                 fontsize=12, fontweight='bold')
    # group legend
    from matplotlib.lines import Line2D
    legend = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=CAT_COLOR['A'],
               markeredgecolor=CAT_COLOR['A'], markersize=9,
               label=CAT_LABEL['A']),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=CAT_COLOR['B'],
               markeredgecolor=CAT_COLOR['B'], markersize=9,
               label=CAT_LABEL['B']),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#555',
               markeredgecolor='#555', markersize=9,
               label='filled marker / *  =  significant after Benjamini-Hochberg (q < 0.05)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
               markeredgecolor='#888', markersize=9,
               label='open marker  =  not significant after BH correction'),
    ]
    ax.legend(handles=legend, loc='lower right', fontsize=8, framealpha=0.95)
    ax.set_xlim(-1.0, max(r['hi'] for r in used) + 1.6)
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), '..', 'figures',
                       'chemo_loh_resistance.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved {out}")


if __name__ == '__main__':
    main()
