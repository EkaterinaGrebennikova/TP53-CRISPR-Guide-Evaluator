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


def fig_cox_forest(survival_df):
    """Fig 4: Forest plot — Cox regression covariates (stratified by cancer type)."""
    from survivalanalysis import cox_regression
    cox = cox_regression(survival_df)
    summary = cox['summary']

    covariates = [
        ('age', 'Age (per year)'),
        ('sex_binary', 'Sex (Male vs Female)'),
        ('allelic_state_heterozygous_cn_neutral', 'Het CN-neutral'),
        ('allelic_state_heterozygous_with_gain', 'Het + Gain'),
        ('allelic_state_loh_with_mutation', 'LOH + Mutation'),
        ('allelic_state_biallelic_mutation', 'Biallelic Mutation'),
    ]

    records = []
    for key, label in covariates:
        if key not in summary.index:
            continue
        row = summary.loc[key]
        records.append({
            'label': label,
            'hr': row['exp(coef)'],
            'ci_low': row['exp(coef) lower 95%'],
            'ci_high': row['exp(coef) upper 95%'],
            'p': row['p'],
        })

    df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.8)))
    y = np.arange(len(df))

    for i, row in enumerate(df.itertuples()):
        color = '#D32F2F' if row.p < 0.05 else '#757575'
        ax.plot([row.ci_low, row.ci_high], [i, i], color=color, linewidth=2.5, zorder=2)
        ax.plot(row.hr, i, 'o', color=color, markersize=9, zorder=3)
        hr_str = f"HR={row.hr:.2f}  p={row.p:.2e}" if row.p < 0.01 else f"HR={row.hr:.2f}  p={row.p:.3f}"
        ax.text(row.ci_high + 0.02, i, hr_str, va='center', fontsize=9, color=color)

    ax.axvline(x=1.0, color='black', linestyle='--', linewidth=1, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(df['label'], fontsize=10)
    ax.set_xlabel('Hazard Ratio (95% CI)', fontsize=11)
    ax.set_title(f'Cox Regression (stratified by cancer type)\nN={cox["n_observations"]}, C-index={cox["concordance"]:.3f}',
                 fontsize=13, fontweight='bold')

    sig_patch = mpatches.Patch(color='#D32F2F', label='p < 0.05')
    ns_patch = mpatches.Patch(color='#757575', label='p >= 0.05')
    ax.legend(handles=[sig_patch, ns_patch], loc='lower right', fontsize=9)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'cox_forest_plot.png')
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  Saved {path}")


def fig_ml_predicted_vs_observed():
    """Honest predicted-vs-observed scatter for the deployed ABE and CBE
    models. Predictions are OUT-OF-FOLD from 5-fold GroupKFold by spacer
    (same gRNA never crosses train/test) -- the leak-free measure of the
    deployed model class's generalization. The reported R^2 and Spearman
    match the headline metrics cited in the paper.

    Deployed configurations (matches src/efficiencypredictorml.py):
      CBE: BE4-only training, 125 features,
           n_est=200, depth=8, lr=0.05, leaf=10, sub=0.8.
      ABE: ABE-only training, 113 features (legacy truncation),
           n_est=400, depth=8, lr=0.03, leaf=10, sub=0.8.
    """
    from efficiencypredictorml import (
        _collect_editor_rows, _df_to_Xy, CBE_CSV, ABE_CSV, MIN_READ_COUNT,
    )
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import r2_score
    from sklearn.model_selection import GroupKFold
    from scipy.stats import spearmanr

    def oof_predict(csv_path, editors, n_features, params):
        df = pd.read_csv(csv_path, low_memory=False)
        c = _collect_editor_rows(df, editors)
        c = c[c['reads'] >= MIN_READ_COUNT].reset_index(drop=True)
        X, y = _df_to_Xy(c, n_features=n_features)
        groups = c['gRNA (20nt)'].to_numpy()
        yhat = np.zeros_like(y)
        for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
            m = GradientBoostingRegressor(random_state=42, **params)
            m.fit(X[tr], y[tr])
            yhat[te] = np.clip(m.predict(X[te]), 0.0, 1.0)
        return y, yhat

    configs = [
        (0, 'ABE (ABE-only, deployed)',
         ABE_CSV, ['ABE'], 113,
         dict(n_estimators=400, max_depth=8, learning_rate=0.03,
              min_samples_leaf=10, subsample=0.8)),
        (1, 'CBE (BE4-only, deployed)',
         CBE_CSV, ['BE4'], 125,
         dict(n_estimators=200, max_depth=8, learning_rate=0.05,
              min_samples_leaf=10, subsample=0.8)),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for idx, label, csv_path, editors, n_features, params in configs:
        ax = axes[idx]
        y, yhat = oof_predict(csv_path, editors, n_features, params)
        r2 = r2_score(y, yhat)
        rho, _ = spearmanr(y, yhat)

        ax.scatter(y, yhat, alpha=0.25, s=10, color='#1976D2',
                   edgecolors='none')
        ax.plot([0, 1], [0, 1], '--', color='#D32F2F', linewidth=1.5,
                label='Perfect prediction')
        ax.set_xlabel('Observed AA correction precision', fontsize=11)
        ax.set_ylabel('Predicted (out-of-fold)', fontsize=11)
        ax.set_title(f'{label}\nn={len(y)}, 5-fold GroupKFold by spacer',
                     fontsize=11, fontweight='bold')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        ax.text(0.05, 0.95,
                f'$R^2$ = {r2:.3f}\nSpearman $\\rho$ = {rho:.3f}',
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
        ax.legend(loc='lower right', fontsize=9)

    fig.text(0.5, -0.01,
             'Out-of-fold predictions: every point predicted by a model trained '
             'on the other 4 folds, with same-spacer rows held together '
             '(no train/test contamination via cross-editor or cross-cell-type '
             'duplicate spacers).',
             ha='center', fontsize=8.5, style='italic')

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'ml_predicted_vs_observed.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


def fig_rescuability():
    """Two-panel: (A) the allelic-state efficiency cliff, robust across
    assembly exponents; (B) modality x allelic-state rescuability
    (best-guide-per-mutation deployed scenario)."""
    from rescuability import (
        efficiency_thresholds, collect_ml_efficiencies,
        recurrent_missense_panel, rescuability_table, EXPONENTS,
    )

    th = efficiency_thresholds()
    panel = recurrent_missense_panel(top_n=150)
    _, by_best = collect_ml_efficiencies(panel)
    tbl = rescuability_table(by_best, th)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- Panel A: the efficiency cliff ---
    x = np.arange(len(EXPONENTS))
    w = 0.35
    het = [th[n]['het_cn_neutral'] for n in EXPONENTS]
    loh = [th[n]['loh_or_biallelic'] for n in EXPONENTS]
    axA.bar(x - w / 2, het, w, label='Het CN-neutral', color='#2196F3',
            edgecolor='black', linewidth=0.6)
    axA.bar(x + w / 2, loh, w, label='LOH / biallelic', color='#F44336',
            edgecolor='black', linewidth=0.6)
    for i in range(len(EXPONENTS)):
        axA.text(x[i] - w / 2, het[i] + 0.02, f'{het[i]:.2f}', ha='center',
                 fontsize=9)
        axA.text(x[i] + w / 2, loh[i] + 0.02, f'{loh[i]:.2f}', ha='center',
                 fontsize=9)
    axA.set_xticks(x)
    axA.set_xticklabels([f'n = {n}' for n in EXPONENTS])
    axA.set_xlabel('Tetramer assembly exponent', fontsize=11)
    axA.set_ylabel('Editing efficiency required\nto reach 0.45 tetramer fraction',
                   fontsize=11)
    axA.set_ylim(0, 1.0)
    axA.set_title('A. The allelic-state efficiency cliff',
                  fontsize=12, fontweight='bold')
    axA.legend(fontsize=9, loc='upper left')

    # --- Panel B: rescuability at n=4, robustness band from n=2,3 ---
    mods = ['ABE', 'CBE']
    states = [('Het\nCN-neutral', 'pct_clear_het'),
              ('LOH /\nbiallelic', 'pct_clear_loh')]
    groups = [f'{m}\n{s[0]}' for m in mods for s in states]
    vals_n4, lo, hi = [], [], []
    for m in mods:
        for _, col in states:
            sub = tbl[tbl['modality'] == m]
            v4 = float(sub[sub['exponent'] == 4][col].iloc[0])
            v_all = [float(sub[sub['exponent'] == n][col].iloc[0])
                     for n in EXPONENTS]
            vals_n4.append(v4)
            lo.append(v4 - min(v_all))
            hi.append(max(v_all) - v4)
    colors = ['#4CAF50', '#F44336', '#4CAF50', '#F44336']
    xb = np.arange(len(groups))
    bars = axB.bar(xb, vals_n4, color=colors, edgecolor='black',
                   linewidth=0.6, yerr=[lo, hi], capsize=5,
                   error_kw={'elinewidth': 1.2})
    for b, v in zip(bars, vals_n4):
        axB.text(b.get_x() + b.get_width() / 2, v + 3, f'{v:.0f}%',
                 ha='center', fontsize=10, fontweight='bold')
    axB.set_xticks(xb)
    axB.set_xticklabels(groups, fontsize=9.5)
    axB.set_ylabel('% of recurrent missense mutations\nrescuable (best guide)',
                   fontsize=11)
    axB.set_ylim(0, 112)
    axB.set_title('B. Rescuability by modality x allelic state',
                  fontsize=12, fontweight='bold')
    axB.text(0.5, 0.97, 'bars: n=4   error bars: n=2-4 range',
             transform=axB.transAxes, ha='center', va='top', fontsize=8.5,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'rescuability.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")
