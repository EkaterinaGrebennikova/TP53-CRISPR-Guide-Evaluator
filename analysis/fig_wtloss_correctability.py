"""Standalone figure for the WT-loss correctability bridge (provisional;
NOT wired into the paper pipeline until the finding is committed).

Panel A: best-guide ML efficiency of real CN-LOH base-editable patients,
         ABE vs CBE, with the het / LOH thresholds (k=4) and the shaded
         WT-loss-attributable band. Shows the band is real and which
         editor populates it.
Panel B: WT-loss-attributable fraction across assembly exponents k=2..4,
         hard vs Phi-debiased, with bootstrap 95% CIs (the robustness
         gate, embedded so the figure shows uncertainty not a point).
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from wtloss_correctability_makebreak import collect, thresholds, K_VALUES

B = 2000
RNG = np.random.default_rng(42)


def band_fractions(e, sig, k):
    tr, tl = thresholds(k)
    hard = float(np.mean((e >= tr) & (e < tl))) * 100
    phi = float(np.mean(norm.cdf((e - tr) / sig)
                        - norm.cdf((e - tl) / sig))) * 100
    return hard, phi


def bootstrap_ci(e, sig, k, kind):
    n = len(e)
    vals = np.empty(B)
    for b in range(B):
        idx = RNG.integers(0, n, n)
        h, p = band_fractions(e[idx], sig[idx], k)
        vals[b] = h if kind == 'hard' else p
    return np.percentile(vals, [2.5, 97.5])


def main():
    d = collect()
    e, sig, mods = d['e'], d['sig'], d['mods']

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 5.5))

    # ---- Panel A: efficiency distribution + band ----
    tr4, tl4 = thresholds(4)
    bins = np.linspace(0, 1, 31)
    abe = e[mods == 'ABE']
    cbe = e[mods == 'CBE']
    axA.hist([cbe, abe], bins=bins, stacked=True,
             color=['#8E24AA', '#43A047'],
             label=[f'CBE (n={len(cbe)})', f'ABE (n={len(abe)})'],
             edgecolor='white', linewidth=0.3)
    axA.axvspan(tr4, tl4, color='#FF7043', alpha=0.18, zorder=0)
    axA.axvline(tr4, color='#1976D2', ls='--', lw=1.5)
    axA.axvline(tl4, color='#D32F2F', ls='--', lw=1.5)
    ymax = axA.get_ylim()[1]
    axA.text(tr4, ymax * 0.98, ' WT-retained\n bar (0.64)',
             color='#1976D2', fontsize=8.5, va='top', ha='left')
    axA.text(tl4, ymax * 0.98, ' WT-lost\n bar (0.82)',
             color='#D32F2F', fontsize=8.5, va='top', ha='left')
    axA.text((tr4 + tl4) / 2, ymax * 0.55,
             'WT-loss-\nattributable\nband',
             ha='center', va='center', fontsize=9, fontweight='bold',
             color='#BF360C')
    axA.set_xlabel('Best-guide ML-predicted correction efficiency',
                   fontsize=10)
    axA.set_ylabel('CN-LOH base-editable patients', fontsize=10)
    axA.set_title('A. Why the band exists (k=4): guides falling\n'
                  'between the two thresholds — CBE-populated',
                  fontsize=11, fontweight='bold')
    axA.legend(fontsize=9, loc='upper left')

    # ---- Panel B: fraction by k, hard vs Phi, bootstrap CI ----
    ks = K_VALUES
    x = np.arange(len(ks))
    w = 0.36
    hard_pts, hard_lo, hard_hi = [], [], []
    phi_pts, phi_lo, phi_hi = [], [], []
    for k in ks:
        h, p = band_fractions(e, sig, k)
        hlo, hhi = bootstrap_ci(e, sig, k, 'hard')
        plo, phi_ = bootstrap_ci(e, sig, k, 'phi')
        hard_pts.append(h); hard_lo.append(h - hlo); hard_hi.append(hhi - h)
        phi_pts.append(p); phi_lo.append(p - plo); phi_hi.append(phi_ - p)

    axB.bar(x - w / 2, hard_pts, w, yerr=[hard_lo, hard_hi],
            capsize=4, color='#90A4AE', edgecolor='black', linewidth=0.5,
            label='Hard count (optimistic; ignores model error)')
    axB.bar(x + w / 2, phi_pts, w, yerr=[phi_lo, phi_hi],
            capsize=4, color='#FF7043', edgecolor='black', linewidth=0.5,
            label='Phi-debiased (propagates model error) — primary')
    axB.axhline(10, color='red', ls=':', lw=1.2)
    axB.text(len(ks) - 0.5, 10.6, 'pre-registered 10% kill line',
             color='red', fontsize=8, ha='right')
    for i, (hp, pp) in enumerate(zip(hard_pts, phi_pts)):
        axB.text(i - w / 2, hp + hard_hi[i] + 0.6, f'{hp:.1f}',
                 ha='center', fontsize=8.5)
        axB.text(i + w / 2, pp + phi_hi[i] + 0.6, f'{pp:.1f}',
                 ha='center', fontsize=8.5, fontweight='bold')
    axB.set_xticks(x)
    axB.set_xticklabels([f'k={k}' for k in ks])
    axB.set_xlabel('Tetramer assembly exponent', fontsize=10)
    axB.set_ylabel('WT-loss-attributable non-correctable (%)',
                   fontsize=10)
    axB.set_title('B. Bridge magnitude with bootstrap 95% CI\n'
                  '(Phi-debiased robust across k=2-4; hard count clears '
                  'with k=3 lower CI borderline)',
                  fontsize=11, fontweight='bold')
    axB.legend(fontsize=8.5, loc='upper left')
    axB.set_ylim(0, max(phi_pts) + max(phi_hi) + 6)

    cap = (f"Denominator: {d['n_loh_total']} CN-LOH single-mutation "
           f"patients; {d['n_be']} base-editable (this analysis), "
           f"{d['n_pe_hdr']} PE/HDR-only "
           f"({100*d['n_pe_hdr']/d['n_loh_total']:.0f}% — separate, "
           f"harder correction-obligate stratum, not shown).")
    fig.text(0.5, -0.02, cap, ha='center', fontsize=8.5,
             style='italic', wrap=True)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), '..', 'figures',
                       'wtloss_correctability_bridge.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == '__main__':
    main()
