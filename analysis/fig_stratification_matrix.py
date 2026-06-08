"""Fig 9 - Therapeutic stratification matrix (paper section 3.10).

Allelic state (rows) x therapeutic modality class (columns). Cells encode
the recommended/expected efficacy, synthesizing:
  (i)  cancer-type-adjusted empirical findings (LOH row: MDM2i resistant
       beta=+2.18; DNA-damaging chemo resistant; anti-mitotics spared),
  (ii) mechanistic prediction (WT-dependence of MDM2i; p53-dependence of
       DNA-damage apoptosis),
  (iii) absence of allelic-state stratification for synthetic-lethality and
        reactivator classes (our null findings -> shown grey).
Cells with a dagger are directly supported by cancer-type-adjusted data.
"""
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# colors
EFFECTIVE  = '#2E7D32'   # green  (favorable allelic context for a p53-pathway drug)
REDUCED    = '#F9A825'   # amber  (reduced efficacy)
INEFFECTIVE = '#C62828'  # red    (resistant; unfavorable allelic context)
P53INDEP   = '#00838F'   # teal   (p53-independent; efficacy NOT allelic-state-limited)
NOSTRAT    = '#616161'   # dark grey (no allelic-state stratification found)
INDICATED  = '#1565C0'   # blue   (gene correction indicated)

STATES = ['Het CN-neutral\n(WT retained)', 'Het + gain', 'LOH', 'Biallelic']
MODALITIES = ['MDM2\ninhibitors', 'DNA-damaging\nchemo',
              'Anti-mitotic\nchemo', 'Synthetic lethality\n(PARP/ATR/CHK1)',
              'Mutant-p53\nreactivators', 'Gene correction\n(CRISPR)']

# (label, color, empirically_anchored)
CELLS = {
    # Het CN-neutral
    (0, 0): ('Effective', EFFECTIVE, False),
    (0, 1): ('Effective', EFFECTIVE, False),
    (0, 2): ('Retained\n(p53-independent)', P53INDEP, False),
    (0, 3): ('No allelic\nstratification', NOSTRAT, True),
    (0, 4): ('Unproven', NOSTRAT, True),
    (0, 5): ('Not required\n(WT present)', EFFECTIVE, False),
    # Het + gain  (verdicts are mechanistic; flagged "not directly evaluated")
    (1, 0): ('Reduced', REDUCED, False),
    (1, 1): ('Reduced', REDUCED, False),
    (1, 2): ('Retained', P53INDEP, False),
    (1, 3): ('No data', NOSTRAT, False),
    (1, 4): ('Unproven', NOSTRAT, False),
    (1, 5): ('Adjunct', REDUCED, False),
    # LOH
    (2, 0): ('Ineffective\n(beta=+2.18)†', INEFFECTIVE, True),
    (2, 1): ('Resistant\n(5 agents, BH)†', INEFFECTIVE, True),
    (2, 2): ('Retained\n(p53-indep.)†', P53INDEP, True),
    (2, 3): ('No allelic\nstratification†', NOSTRAT, True),
    (2, 4): ('No effect\n(PRIMA-1MET)†', NOSTRAT, True),
    (2, 5): ('Indicated;\nceiling†', INDICATED, True),
    # Biallelic
    (3, 0): ('Ineffective', INEFFECTIVE, False),
    (3, 1): ('Resistant', INEFFECTIVE, False),
    (3, 2): ('Retained\n(p53-independent)', P53INDEP, False),
    (3, 3): ('No allelic\nstratification', NOSTRAT, True),
    (3, 4): ('Unproven', NOSTRAT, True),
    (3, 5): ('Indicated;\nhardest', INDICATED, False),
}


def main():
    nrow, ncol = len(STATES), len(MODALITIES)
    fig, ax = plt.subplots(figsize=(13, 6.5))

    HET_GAIN_ROW = 1   # verdicts are mechanistic -> annotate "not directly evaluated"
    for (r, c), (label, color, anchored) in CELLS.items():
        # row 0 at top
        y = nrow - 1 - r
        rect = Rectangle((c, y), 1, 1, facecolor=color, alpha=0.78,
                         edgecolor='white', linewidth=2)
        ax.add_patch(rect)
        if anchored:
            # bold border for empirically-anchored cells
            ax.add_patch(Rectangle((c + 0.02, y + 0.02), 0.96, 0.96,
                         fill=False, edgecolor='black', linewidth=1.6))
        if r == HET_GAIN_ROW:
            ax.text(c + 0.5, y + 0.68, label, ha='center', va='center',
                    fontsize=8.5, color='white', fontweight='bold')
            ax.text(c + 0.5, y + 0.32, 'not directly\nevaluated',
                    ha='center', va='center', fontsize=7.8, color='white',
                    style='italic')
        else:
            ax.text(c + 0.5, y + 0.5, label, ha='center', va='center',
                    fontsize=8.5, color='white', fontweight='bold')

    ax.set_xlim(0, ncol)
    ax.set_ylim(0, nrow)
    ax.set_xticks([c + 0.5 for c in range(ncol)])
    ax.set_xticklabels(MODALITIES, fontsize=9.5)
    ax.set_yticks([nrow - 1 - r + 0.5 for r in range(nrow)])
    ax.set_yticklabels(STATES, fontsize=9.5)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    ax.set_title('Therapeutic stratification of TP53-mutant cancers by allelic state',
                 fontsize=13, fontweight='bold', pad=38)

    # legend
    from matplotlib.patches import Patch
    legend = [
        Patch(facecolor=EFFECTIVE, alpha=0.78, label='Effective (favorable allelic context)'),
        Patch(facecolor=REDUCED, alpha=0.78, label='Reduced / adjunct'),
        Patch(facecolor=INEFFECTIVE, alpha=0.78, label='Ineffective / resistant'),
        Patch(facecolor=P53INDEP, alpha=0.78,
              label='p53-independent: efficacy not allelic-state-limited\n'
                    '(selection driven by cancer type, not p53 status)'),
        Patch(facecolor=NOSTRAT, alpha=0.78, label='No allelic-state stratification found'),
        Patch(facecolor=INDICATED, alpha=0.78, label='Gene correction indicated'),
        Patch(facecolor='white', edgecolor='black',
              label='† / bold border = cancer-type-adjusted empirical support'),
    ]
    ax.legend(handles=legend, loc='upper center', bbox_to_anchor=(0.5, -0.03),
              ncol=2, fontsize=8.5, frameon=False)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), '..', 'figures',
                       'stratification_matrix.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == '__main__':
    main()
