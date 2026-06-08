"""Verify the per-modality §3.9 claims:
   - "~50% of CBE-correctable LOH non-correctable due to WT loss"
   - "~0.8% of ABE-correctable LOH non-correctable due to WT loss"

Splits the make-or-break cohort by chosen editor and reports the
WT-loss-attributable band fraction within each.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from scipy.stats import norm

from wtloss_correctability_makebreak import collect, thresholds, K_VALUES


def main():
    d = collect()
    e, sig, mods = d['e'], d['sig'], d['mods']
    print(f"\nBase-editable cohort: ABE n={int((mods=='ABE').sum())}, "
          f"CBE n={int((mods=='CBE').sum())}")

    print("\n=== PER-MODALITY WT-loss-attributable fraction ===")
    print(f"{'k':>3} {'editor':>6}  {'n':>4}  {'clears LOH':>11}  "
          f"{'in band (hard)':>15}  {'Phi-debiased':>13}")
    for k in K_VALUES:
        tr, tl = thresholds(k)
        for ed in ('ABE', 'CBE'):
            mask = mods == ed
            ee = e[mask]; ss = sig[mask]
            if len(ee) == 0:
                continue
            clears = float(np.mean(ee >= tl)) * 100
            hard = float(np.mean((ee >= tr) & (ee < tl))) * 100
            phi = float(np.mean(norm.cdf((ee - tr) / ss)
                                - norm.cdf((ee - tl) / ss))) * 100
            print(f"{k:>3} {ed:>6}  {len(ee):>4}  "
                  f"{clears:>10.1f}%  {hard:>14.1f}%  {phi:>12.1f}%")


if __name__ == '__main__':
    main()
