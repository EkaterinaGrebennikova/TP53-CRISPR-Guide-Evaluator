"""Lever 1: does training on the inference-appropriate editor beat the
pooled-6-editor model? All evaluated leak-free (group-by-spacer).

Inference always uses editor='BE4'. The deployed model trains on 6 pooled
CBE editors. This tests whether matching train to inference (BE4-only, or
BE4 + its CP1028 variant) raises the HONEST (group-split) R^2 vs the
pooled baseline -- the decision fork:

  BE4-only group R^2 >  pooled 0.45  -> pooling was hurting (improvement)
  BE4-only group R^2 ~= pooled 0.45  -> editor structure not the limiter
  BE4-only group R^2 <  pooled 0.45  -> n-loss dominates; keep pooling

Held fixed for apples-to-apples: MIN_READS=100 (proven best), 125 features,
deployed hyperparameters. Only the editor subset and the split change.
random-split R^2 shown only to expose the per-config leakage gap.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import (
    train_test_split, GroupShuffleSplit, GroupKFold,
)
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr

from efficiencypredictorml import _collect_editor_rows, _df_to_Xy, CBE_CSV

BEST = dict(n_estimators=400, max_depth=8, learning_rate=0.05,
            subsample=0.8, random_state=42)
MIN_READS = 100

CONFIGS = [
    ('pooled-6 (baseline)', ['BE4', 'BE4-CP1028', 'AID', 'CDA',
                             'eA3A', 'evoAPOBEC']),
    ('BE4 + BE4-CP1028',    ['BE4', 'BE4-CP1028']),
    ('BE4-only (inference)', ['BE4']),
]


def build(editors):
    df = pd.read_csv(CBE_CSV, low_memory=False)
    combined = _collect_editor_rows(df, editors)
    combined = combined[combined['reads'] >= MIN_READS].reset_index(drop=True)
    X, y = _df_to_Xy(combined, n_features=125)
    groups = combined['gRNA (20nt)'].to_numpy()
    return X, y, groups


def fit_eval(Xtr, ytr, Xte, yte):
    m = GradientBoostingRegressor(**BEST)
    m.fit(Xtr, ytr)
    p = np.clip(m.predict(Xte), 0.0, 1.0)
    return (r2_score(yte, p),
            np.sqrt(mean_squared_error(yte, p)),
            spearmanr(yte, p)[0])


def group_kfold(X, y, groups, k=5):
    gkf = GroupKFold(n_splits=k)
    r2s, rhos = [], []
    for tr, te in gkf.split(X, y, groups):
        r2, _, rho = fit_eval(X[tr], y[tr], X[te], y[te])
        r2s.append(r2); rhos.append(rho)
    return np.mean(r2s), np.std(r2s), np.mean(rhos)


def main():
    rows = []
    for label, editors in CONFIGS:
        X, y, groups = build(editors)
        nuq = len(np.unique(groups))

        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=42)
        r2_rand, _, _ = fit_eval(Xtr, ytr, Xte, yte)

        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        tr, te = next(gss.split(X, y, groups))
        r2_g, rmse_g, rho_g = fit_eval(X[tr], y[tr], X[te], y[te])

        cv_m, cv_s, cv_rho = group_kfold(X, y, groups)

        print(f"\n=== {label} ===")
        print(f"  n={len(y)}  unique spacers={nuq}")
        print(f"  random split R2     : {r2_rand:.3f}  "
              f"(leak gap vs group: {r2_rand - r2_g:+.3f})")
        print(f"  group split  R2     : {r2_g:.3f}  RMSE={rmse_g:.3f}  "
              f"rho={rho_g:.3f}")
        print(f"  GroupKFold CV R2    : {cv_m:.3f} +/- {cv_s:.3f}  "
              f"(CV rho={cv_rho:.3f})")
        rows.append((label, len(y), nuq, r2_rand, r2_g, rmse_g,
                     rho_g, cv_m, cv_s, cv_rho))

    print("\n" + "=" * 78)
    print("SUMMARY  (honest = GroupKFold CV; bar to beat = pooled CV)")
    print(f"{'config':<22}{'n':>6}{'rand':>7}{'grpR2':>7}"
          f"{'grpRMSE':>8}{'grpRho':>8}{'cvR2':>8}{'cvRho':>7}")
    for (lab, n, _, rr, rg, rm, rh, cm, cs, cr) in rows:
        print(f"{lab:<22}{n:>6}{rr:>7.3f}{rg:>7.3f}"
              f"{rm:>8.3f}{rh:>8.3f}{cm:>8.3f}{cr:>7.3f}")

    base_cv = rows[0][7]
    print("\n--- FORK VERDICT ---")
    for (lab, _, _, _, _, _, _, cm, _, _) in rows[1:]:
        d = cm - base_cv
        tag = ('IMPROVES' if d > 0.02 else
               'NO BETTER' if abs(d) <= 0.02 else 'WORSE')
        print(f"  {lab:<22} CV R2={cm:.3f}  vs pooled {base_cv:.3f}  "
              f"({d:+.3f})  -> {tag}")
    print("  (bar = pooled GroupKFold CV R2; >+0.02 = real structural gain)")


if __name__ == '__main__':
    main()
