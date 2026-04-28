import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import os, sys

sys.path.insert(0, os.path.dirname(__file__))

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from tcgaloader import load_mutations, load_cna, load_clinical
from tcgaallelic import get_allelic_context_by_cancer_type
from survivalanalysis import build_survival_df, survival_by_cancer_type
from depmapdrugresponse import (
    load_tp53_mutations, load_tp53_cna, load_drug_response, classify_cell_lines,
)
from mskvalidation import load_msk_mutations, load_msk_cna, load_msk_clinical

FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

STATE_COLORS = {
    'wildtype':                '#4CAF50',
    'heterozygous_cn_neutral': '#2196F3',
    'heterozygous_with_gain':  '#FF9800',
    'loh_with_mutation':       '#F44336',
    'biallelic_mutation':      '#9C27B0',
    'unknown':                 '#9E9E9E',
}

STATE_LABELS = {
    'wildtype':                'Wildtype',
    'heterozygous_cn_neutral': 'Het CN-neutral',
    'heterozygous_with_gain':  'Het + Gain',
    'loh_with_mutation':       'LOH + Mutation',
    'biallelic_mutation':      'Biallelic',
    'unknown':                 'Unknown',
}

STATE_ORDER = [
    'wildtype', 'heterozygous_cn_neutral', 'heterozygous_with_gain',
    'loh_with_mutation', 'biallelic_mutation', 'unknown',
]


def fig_allelic_bar_by_cancer(mutations_df, cna_df, clinical_df, min_patients=20):
    """Fig 1: Stacked bar — allelic state distribution per cancer type."""
    allelic_by_cancer = get_allelic_context_by_cancer_type(mutations_df, cna_df, clinical_df)

    # Build data, filter to cancer types with enough patients
    cancer_types = []
    fractions = {s: [] for s in STATE_ORDER if s != 'wildtype'}
    for ct, info in sorted(allelic_by_cancer.items()):
        if info['total_tp53_mutant_patients'] < min_patients:
            continue
        cancer_types.append(ct)
        total = info['total_tp53_mutant_patients']
        for state in STATE_ORDER:
            if state == 'wildtype':
                continue
            fractions[state].append(info['states'].get(state, 0) / total * 100)

    if not cancer_types:
        print("  No cancer types with enough patients for allelic bar chart.")
        return

    # Sort by LOH fraction descending
    loh_fracs = fractions['loh_with_mutation']
    sort_idx = np.argsort(loh_fracs)[::-1]
    cancer_types = [cancer_types[i] for i in sort_idx]
    for state in fractions:
        fractions[state] = [fractions[state][i] for i in sort_idx]

    fig, ax = plt.subplots(figsize=(14, 7))
    y = np.arange(len(cancer_types))
    left = np.zeros(len(cancer_types))

    for state in STATE_ORDER:
        if state == 'wildtype':
            continue
        vals = np.array(fractions[state])
        ax.barh(y, vals, left=left, color=STATE_COLORS[state], label=STATE_LABELS[state],
                edgecolor='white', linewidth=0.5)
        left += vals

    ax.set_yticks(y)
    ax.set_yticklabels(cancer_types, fontsize=9)
    ax.set_xlabel('Percentage of TP53-Mutant Patients (%)', fontsize=11)
    ax.set_title('TP53 Allelic State Distribution by Cancer Type (TCGA)', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, 'allelic_state_by_cancer_type.png')
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  Saved {path}")


def fig_per_cancer_forest(survival_df, min_patients=30):
    """Fig 4: Forest plot — HR of TP53-mut vs WT per cancer type."""
    records = []
    for cancer_type, group in survival_df.groupby('cancer_type'):
        mut = group[group['tp53_mut']]
        wt = group[~group['tp53_mut']]
        if len(mut) < 10 or len(wt) < 10:
            continue
        # Univariate Cox for HR + CI
        cox_df = group[['os_months', 'os_event', 'tp53_mut']].copy()
        cox_df['tp53_mut'] = cox_df['tp53_mut'].astype(int)
        cox_df = cox_df.dropna()
        try:
            cph = CoxPHFitter()
            cph.fit(cox_df, duration_col='os_months', event_col='os_event')
            hr = np.exp(cph.params_['tp53_mut'])
            ci = np.exp(cph.confidence_intervals_.loc['tp53_mut'].values)
            p = cph.summary.loc['tp53_mut', 'p']
            records.append({
                'cancer_type': cancer_type,
                'hr': hr, 'ci_low': ci[0], 'ci_high': ci[1], 'p': p,
                'n_mut': len(mut), 'n_wt': len(wt),
            })
        except Exception:
            continue

    if not records:
        print("  No cancer types with enough data for forest plot.")
        return

    df = pd.DataFrame(records).sort_values('hr', ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.4)))
    y = np.arange(len(df))

    for i, row in enumerate(df.itertuples()):
        color = '#D32F2F' if row.p < 0.05 else '#757575'
        ax.plot([row.ci_low, row.ci_high], [i, i], color=color, linewidth=2, zorder=2)
        ax.plot(row.hr, i, 'D', color=color, markersize=7, zorder=3)

    ax.axvline(x=1.0, color='black', linestyle='--', linewidth=1, zorder=1)
    labels = [f"{r.cancer_type}  (n={r.n_mut}+{r.n_wt})" for r in df.itertuples()]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Hazard Ratio (log scale, 95% CI)', fontsize=11)
    ax.set_title('TP53 Mutation Hazard Ratio by Cancer Type (TCGA)', fontsize=13, fontweight='bold')
    ax.set_xscale('log')

    # Legend
    sig_patch = mpatches.Patch(color='#D32F2F', label='p < 0.05')
    ns_patch = mpatches.Patch(color='#757575', label='p ≥ 0.05')
    ax.legend(handles=[sig_patch, ns_patch], loc='lower right', fontsize=9)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'per_cancer_forest_plot.png')
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  Saved {path}")


def fig_depmap_nutlin_box():
    """Fig 5: Box plot — ln(IC50) for Nutlin-3a by allelic state."""
    muts = load_tp53_mutations()
    cna = load_tp53_cna()
    drugs = load_drug_response()
    classified = classify_cell_lines(muts, cna)

    nutlin = drugs[drugs['DRUG_NAME'] == 'Nutlin-3a (-)']
    merged = nutlin.merge(classified, on='ModelID', how='inner')

    plot_order = [s for s in STATE_ORDER if s in merged['allelic_state'].unique()]

    fig, ax = plt.subplots(figsize=(10, 6))
    data_by_state = [merged[merged['allelic_state'] == s]['LN_IC50'].values for s in plot_order]
    labels = [f"{STATE_LABELS[s]}\n(n={len(d)})" for s, d in zip(plot_order, data_by_state)]
    colors = [STATE_COLORS[s] for s in plot_order]

    bp = ax.boxplot(data_by_state, tick_labels=labels, patch_artist=True, widths=0.6)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('ln(IC50)', fontsize=11)
    ax.set_title('Nutlin-3a Sensitivity by TP53 Allelic State (DepMap/GDSC)', fontsize=13, fontweight='bold')
    ax.axhline(y=merged[merged['allelic_state'] == 'wildtype']['LN_IC50'].median(),
               color=STATE_COLORS['wildtype'], linestyle='--', alpha=0.5, label='WT median')

    # Add significance annotation for biallelic vs WT
    wt_ic50 = merged[merged['allelic_state'] == 'wildtype']['LN_IC50']
    bi_ic50 = merged[merged['allelic_state'] == 'biallelic_mutation']['LN_IC50']
    if len(wt_ic50) >= 3 and len(bi_ic50) >= 3:
        from scipy.stats import mannwhitneyu
        _, p = mannwhitneyu(bi_ic50, wt_ic50, alternative='two-sided')
        # Find positions of wildtype and biallelic in plot_order
        wt_pos = plot_order.index('wildtype') + 1
        bi_pos = plot_order.index('biallelic_mutation') + 1
        y_max = max(merged['LN_IC50'].max(), 8)
        ax.plot([wt_pos, wt_pos, bi_pos, bi_pos], [y_max + 0.3, y_max + 0.5, y_max + 0.5, y_max + 0.3],
                color='black', linewidth=1)
        ax.text((wt_pos + bi_pos) / 2, y_max + 0.6, f'p = {p:.1e}', ha='center', fontsize=9)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'depmap_nutlin_boxplot.png')
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  Saved {path}")


def fig_msk_km():
    """Fig S2: KM curve — TP53-mut vs WT in MSK-IMPACT."""
    m = load_msk_mutations()
    c = load_msk_cna()
    cl = load_msk_clinical()
    sdf = build_survival_df(m, c, cl)

    mut = sdf[sdf['tp53_mut']]
    wt = sdf[~sdf['tp53_mut']]

    kmf_mut = KaplanMeierFitter()
    kmf_wt = KaplanMeierFitter()
    kmf_mut.fit(mut['os_months'], mut['os_event'], label=f'TP53-MUT (n={len(mut)})')
    kmf_wt.fit(wt['os_months'], wt['os_event'], label=f'TP53-WT (n={len(wt)})')

    lr = logrank_test(mut['os_months'], wt['os_months'],
                      event_observed_A=mut['os_event'], event_observed_B=wt['os_event'])

    fig, ax = plt.subplots(figsize=(8, 6))
    kmf_wt.plot_survival_function(ax=ax, color=STATE_COLORS['wildtype'], linewidth=2)
    kmf_mut.plot_survival_function(ax=ax, color='#D32F2F', linewidth=2)

    ax.set_xlabel('Time (months)', fontsize=11)
    ax.set_ylabel('Survival Probability', fontsize=11)
    ax.set_title('MSK-IMPACT: TP53-MUT vs WT Overall Survival', fontsize=13, fontweight='bold')

    med_mut = kmf_mut.median_survival_time_
    med_wt = kmf_wt.median_survival_time_
    med_mut_str = f"{med_mut:.1f}" if med_mut != float('inf') else 'NR'
    med_wt_str = f"{med_wt:.1f}" if med_wt != float('inf') else 'NR'
    stats_text = (f"Log-rank p = {lr.p_value:.2e}\n"
                  f"Median OS MUT: {med_mut_str} mo\n"
                  f"Median OS WT: {med_wt_str} mo")
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'msk_km_mut_vs_wt.png')
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  Saved {path}")


if __name__ == '__main__':
    print("Loading TCGA data...")
    mutations = load_mutations()
    cna = load_cna()
    clinical = load_clinical()
    survival_df = build_survival_df(mutations, cna, clinical)

    print("\nGenerating Fig 1: Allelic state by cancer type...")
    fig_allelic_bar_by_cancer(mutations, cna, clinical)

    print("\nGenerating Fig 4: Per-cancer forest plot...")
    fig_per_cancer_forest(survival_df)

    print("\nGenerating Fig 5: DepMap Nutlin-3a box plot...")
    fig_depmap_nutlin_box()

    print("\nGenerating Fig S2: MSK-IMPACT KM curve...")
    fig_msk_km()

    print("\nDone. All figures saved to figures/")
