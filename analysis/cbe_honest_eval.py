"""Lever 5 (group-by-spacer split) + Lever 2 (read-count weighting/threshold)
honest evaluation of the CBE model.

The production model uses a random train_test_split. Because the same 20nt
spacer is assayed across all 6 CBE editors in Arbab, a random split leaks
near-duplicate spacers between train and test, inflating held-out R^2. This
script measures the leakage-free generalization R^2 (GroupShuffleSplit /
GroupKFold grouped by spacer) and tests whether raising the read threshold
and/or weighting by read count improves the honest number.

Apples-to-apples: same features (125), same best hyperparameters as the
deployed model (n_est=400, depth=8, lr=0.05, sub=0.8). Only the split and
the read handling change.
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

from efficiencypredictorml import _collect_editor_rows, _df_to_Xy, CBE_EDITORS, CBE_CSV

BEST = dict(n_estimators=400, max_depth=8, learning_rate=0.05,
            subsample=0.8, random_state=42)


def build(min_reads):
    df = pd.read_csv(CBE_CSV, low_memory=False)
    combined = _collect_editor_rows(df, CBE_EDITORS)
    combined = combined[combined['reads'] >= min_reads].reset_index(drop=True)
    X, y = _df_to_Xy(combined, n_features=125)
    groups = combined['gRNA (20nt)'].to_numpy()
    reads = combined['reads'].to_numpy(dtype=float)
    return X, y, groups, reads


def fit_eval(Xtr, ytr, Xte, yte, w=None):
    m = GradientBoostingRegressor(**BEST)
    m.fit(Xtr, ytr, sample_weight=w)
    p = np.clip(m.predict(Xte), 0.0, 1.0)
    return (r2_score(yte, p),
            np.sqrt(mean_squared_error(yte, p)),
            spearmanr(yte, p)[0])


def group_kfold_r2(X, y, groups, w=None, k=5):
    gkf = GroupKFold(n_splits=k)
    scores = []
    for tr, te in gkf.split(X, y, groups):
        wt = w[tr] if w is not None else None
        scores.append(fit_eval(X[tr], y[tr], X[te], y[te], wt)[0])
    return np.mean(scores), np.std(scores)


def run(min_reads, label):
    X, y, groups, reads = build(min_reads)
    n_uniq = len(np.unique(groups))
    print(f"\n=== {label}  (MIN_READS={min_reads}) ===")
    print(f"n={len(y)}  unique spacers={n_uniq}")

    # Baseline: random split (reproduces the inflated production number)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    r2, rmse, rho = fit_eval(Xtr, ytr, Xte, yte)
    print(f"  random split        : R2={r2:.3f}  RMSE={rmse:.3f}  rho={rho:.3f}")

    # Lever 5: group-by-spacer holdout
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr, te = next(gss.split(X, y, groups))
    r2g, rmseg, rhog = fit_eval(X[tr], y[tr], X[te], y[te])
    print(f"  group split (L5)    : R2={r2g:.3f}  RMSE={rmseg:.3f}  rho={rhog:.3f}")

    # Lever 5 + Lever 2 weighting (sample_weight proportional to reads)
    r2w, rmsew, rhow = fit_eval(X[tr], y[tr], X[te], y[te], w=reads[tr])
    print(f"  group + read-weight : R2={r2w:.3f}  RMSE={rmsew:.3f}  rho={rhow:.3f}")

    # Honest CV (group k-fold)
    cv_m, cv_s = group_kfold_r2(X, y, groups)
    cv_mw, cv_sw = group_kfold_r2(X, y, groups, w=reads)
    print(f"  GroupKFold CV R2    : {cv_m:.3f} +/- {cv_s:.3f}")
    print(f"  GroupKFold CV (wt)  : {cv_mw:.3f} +/- {cv_sw:.3f}")
    return dict(label=label, min_reads=min_reads, n=len(y),
                random=r2, group=r2g, group_wt=r2w,
                cv=cv_m, cv_wt=cv_mw)


if __name__ == '__main__':
    results = []
    for mr, lbl in [(100, 'current threshold'),
                    (300, 'raised threshold'),
                    (500, 'high threshold')]:
        results.append(run(mr, lbl))

    print("\n" + "=" * 64)
    print("SUMMARY (Test R2)")
    print(f"{'config':<20}{'n':>7}{'random':>9}{'group':>8}"
          f"{'grp+wt':>8}{'CV':>8}{'CV+wt':>8}")
    for r in results:
        print(f"{r['label']:<20}{r['n']:>7}{r['random']:>9.3f}"
              f"{r['group']:>8.3f}{r['group_wt']:>8.3f}"
              f"{r['cv']:>8.3f}{r['cv_wt']:>8.3f}")
