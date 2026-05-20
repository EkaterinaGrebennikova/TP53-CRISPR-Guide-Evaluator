"""Retune CBE GradientBoostingRegressor for BE4-only (n=2,466).

Current hyperparameters were tuned on the 17,724-row pooled set and are
likely over-capacity for the smaller BE4-only set. Targeting the params
most likely wrong-sized:
  - max_depth      (over-capacity suspect)
  - min_samples_leaf  (key small-n regularizer; sklearn default=1 = overfit)
  - n_estimators x learning_rate  (tree-count / shrinkage trade-off)

NON-NEGOTIABLE: scoring is GroupKFold(5) by spacer. Random-fold CV would
reintroduce the exact leakage we just caught. Pooled-tuned params kept in
the grid as a baseline reference (n_est=400, depth=8, lr=0.05, leaf=1).

Bar to beat: BE4-only @ pooled-tuned params -> CV R^2 = 0.508, rho = 0.733.
"""
import os
import sys
import time
import itertools
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

from efficiencypredictorml import _collect_editor_rows, _df_to_Xy, CBE_CSV

MIN_READS = 100


def build_be4():
    df = pd.read_csv(CBE_CSV, low_memory=False)
    combined = _collect_editor_rows(df, ['BE4'])
    combined = combined[combined['reads'] >= MIN_READS].reset_index(drop=True)
    X, y = _df_to_Xy(combined, n_features=125)
    groups = combined['gRNA (20nt)'].to_numpy()
    return X, y, groups


def cv_score(X, y, groups, params, k=5):
    gkf = GroupKFold(n_splits=k)
    r2s, rhos = [], []
    for tr, te in gkf.split(X, y, groups):
        m = GradientBoostingRegressor(random_state=42, **params)
        m.fit(X[tr], y[tr])
        p = np.clip(m.predict(X[te]), 0.0, 1.0)
        r2s.append(r2_score(y[te], p))
        rhos.append(spearmanr(y[te], p)[0])
    return np.mean(r2s), np.std(r2s), np.mean(rhos)


def main():
    X, y, groups = build_be4()
    print(f"BE4-only: n={len(y)}  unique spacers={len(np.unique(groups))}")
    print(f"Baseline (pooled-tuned, n_est=400 depth=8 lr=0.05 leaf=1 sub=0.8):")
    base_params = dict(n_estimators=400, max_depth=8, learning_rate=0.05,
                       subsample=0.8, min_samples_leaf=1)
    r2_b, s_b, rho_b = cv_score(X, y, groups, base_params)
    print(f"  CV R2 = {r2_b:.3f} +/- {s_b:.3f}  CV rho = {rho_b:.3f}")

    grid = dict(
        n_estimators=[200, 400, 800],
        max_depth=[3, 4, 6, 8],
        learning_rate=[0.03, 0.05, 0.1],
        min_samples_leaf=[1, 5, 10],
        subsample=[0.8],
    )
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    print(f"\nGrid: {len(combos)} combinations x 5-fold GroupKFold "
          f"= {len(combos) * 5} fits\n")

    results = []
    t0 = time.time()
    for i, vals in enumerate(combos, 1):
        params = dict(zip(keys, vals))
        r2, sd, rho = cv_score(X, y, groups, params)
        results.append((r2, sd, rho, params))
        if i % 12 == 0 or i == len(combos):
            elapsed = time.time() - t0
            best_so_far = max(r['r2'] if isinstance(r, dict) else r[0]
                              for r in results)
            print(f"  {i:>3}/{len(combos)}  best CV R2 so far = "
                  f"{best_so_far:.3f}  elapsed {elapsed:.0f}s")

    results.sort(key=lambda r: -r[0])
    print("\n=== TOP 10 configurations (by GroupKFold CV R^2) ===")
    print(f"{'CV R2':>7}{'+/-':>7}{'CV rho':>8}  params")
    for r2, sd, rho, p in results[:10]:
        print(f"{r2:>7.3f}{sd:>7.3f}{rho:>8.3f}  {p}")

    best_r2, best_sd, best_rho, best_p = results[0]
    print("\n=== BEST ===")
    print(f"  params: {best_p}")
    print(f"  CV R^2 = {best_r2:.3f} +/- {best_sd:.3f}  "
          f"CV rho = {best_rho:.3f}")
    print(f"  vs baseline (pooled-tuned on BE4): R^2 {r2_b:.3f}->{best_r2:.3f} "
          f"({best_r2 - r2_b:+.3f})   rho {rho_b:.3f}->{best_rho:.3f} "
          f"({best_rho - rho_b:+.3f})")
    if best_r2 - r2_b > 0.02:
        print("  VERDICT: real gain from retuning.")
    elif best_r2 - r2_b > 0.005:
        print("  VERDICT: marginal gain.")
    else:
        print("  VERDICT: retuning did not help; pooled-tuned params were "
              "fine for BE4-only -- the limit is features/biology, not "
              "hyperparameters.")


if __name__ == '__main__':
    main()
