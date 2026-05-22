"""ABE group-split honest evaluation -- analogous to the CBE leakage check.

Paper currently cites ABE R^2=0.841 (random split). ABE pools 2 editors x
2 cell types (up to 4 copies per spacer) -- fewer cross-editor duplicates
than the 6-editor CBE setup, but not zero. This script:
  1. Reproduces the inflated random-split number.
  2. Computes the leak-free group-by-spacer holdout + GroupKFold CV.
  3. Tests whether ABE-only (excluding ABE-CP1040) helps, analogous to
     BE4-only for CBE.

Apples-to-apples: 113 features (legacy ABE truncation, matching deployed
inference), MIN_READS=100, deployed default hyperparameters.
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

from efficiencypredictorml import _collect_editor_rows, _df_to_Xy, ABE_CSV

# deployed ABE defaults (efficiencypredictorml.train_model defaults)
BEST = dict(n_estimators=200, max_depth=8, learning_rate=0.05,
            subsample=0.8, random_state=42)
MIN_READS = 100

CONFIGS = [
    ('pooled (ABE + ABE-CP1040)', ['ABE', 'ABE-CP1040']),
    ('ABE-only (inference)',      ['ABE']),
]


def build(editors):
    df = pd.read_csv(ABE_CSV, low_memory=False)
    combined = _collect_editor_rows(df, editors)
    combined = combined[combined['reads'] >= MIN_READS].reset_index(drop=True)
    X, y = _df_to_Xy(combined, n_features=113)   # ABE legacy truncation
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
        r2_rand, _, rho_rand = fit_eval(Xtr, ytr, Xte, yte)

        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        tr, te = next(gss.split(X, y, groups))
        r2_g, rmse_g, rho_g = fit_eval(X[tr], y[tr], X[te], y[te])

        cv_m, cv_s, cv_rho = group_kfold(X, y, groups)

        print(f"\n=== {label} ===")
        print(f"  n={len(y)}  unique spacers={nuq}")
        print(f"  random split R2     : {r2_rand:.3f}  rho={rho_rand:.3f}  "
              f"(leak gap vs group R2: {r2_rand - r2_g:+.3f})")
        print(f"  group split  R2     : {r2_g:.3f}  RMSE={rmse_g:.3f}  "
              f"rho={rho_g:.3f}")
        print(f"  GroupKFold CV R2    : {cv_m:.3f} +/- {cv_s:.3f}  "
              f"(CV rho={cv_rho:.3f})")
        rows.append((label, len(y), nuq, r2_rand, rho_rand, r2_g, rmse_g,
                     rho_g, cv_m, cv_s, cv_rho))

    print("\n" + "=" * 84)
    print("SUMMARY  (honest = GroupKFold CV)")
    print(f"{'config':<28}{'n':>6}{'rand R2':>9}{'rand rho':>10}"
          f"{'grp R2':>8}{'grp rho':>9}{'cv R2':>8}{'cv rho':>8}")
    for (lab, n, _, rr, rrho, rg, rm, rh, cm, cs, cr) in rows:
        print(f"{lab:<28}{n:>6}{rr:>9.3f}{rrho:>10.3f}"
              f"{rg:>8.3f}{rh:>9.3f}{cm:>8.3f}{cr:>8.3f}")

    pooled_cv = rows[0][8]; pooled_cv_rho = rows[0][10]
    pooled_rand = rows[0][3]
    leak = pooled_rand - pooled_cv
    print(f"\n--- LEAKAGE on pooled ABE ---")
    print(f"  random R2={pooled_rand:.3f}  CV R2={pooled_cv:.3f}  "
          f"gap={leak:+.3f}")
    if leak > 0.15:
        sev = "LARGE -- the paper's 0.841 is materially inflated"
    elif leak > 0.05:
        sev = "MODERATE -- correction needed; smaller than CBE's +0.22"
    else:
        sev = "SMALL -- ABE largely escapes the leakage problem"
    print(f"  -> {sev}")

    abe_only_cv = rows[1][8]; abe_only_rho = rows[1][10]
    d = abe_only_cv - pooled_cv
    drho = abe_only_rho - pooled_cv_rho
    print(f"\n--- ABE-only fork (vs pooled honest baseline) ---")
    print(f"  pooled CV: R2={pooled_cv:.3f}  rho={pooled_cv_rho:.3f}")
    print(f"  ABE-only:  R2={abe_only_cv:.3f}  rho={abe_only_rho:.3f}  "
          f"(R2 {d:+.3f}, rho {drho:+.3f})")
    if d > 0.02:
        verdict = "IMPROVES -- pool was hurting, like CBE"
    elif abs(d) <= 0.02:
        verdict = "NO BETTER -- ABE pooling roughly neutral"
    else:
        verdict = "WORSE -- n-loss dominates; keep pooling"
    print(f"  -> {verdict}")


if __name__ == '__main__':
    main()
