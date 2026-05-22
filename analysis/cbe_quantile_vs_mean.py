"""Does quantile-regression LightGBM (median target) reduce the
regression-to-the-mean shrinkage seen in the predicted-vs-observed
scatter, without sacrificing rank performance?

Three models compared on BE4-only CBE, leak-free GroupKFold(5) by spacer:
  1. sklearn GBR (deployed)         -- mean,  MSE loss
  2. LightGBM regression            -- mean,  MSE loss   [separates model class
                                                          from loss]
  3. LightGBM quantile alpha=0.5    -- median, quantile loss

Metrics:
  R^2          : agreement
  Spearman rho : rank-correctness (operationally relevant for a guide-picker)
  slope        : OLS slope of predicted on observed -- the shrinkage diagnostic.
                 1.0 = no shrinkage; <1 = compression toward the mean.

Decoupling logic:
  M1 vs M2 isolates the model-class effect (sklearn GBR vs LightGBM tree
    booster), holding the loss fixed.
  M2 vs M3 isolates the loss effect (mean vs median), holding the model
    class fixed.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
import lightgbm as lgb

from efficiencypredictorml import _collect_editor_rows, _df_to_Xy, CBE_CSV

MIN_READS = 100

# Apples-to-apples hyperparameters. sklearn GBR uses these; LightGBM uses
# the closest analogs for tree complexity (num_leaves bounded by max_depth).
SKL_PARAMS = dict(n_estimators=200, max_depth=8, learning_rate=0.05,
                  min_samples_leaf=10, subsample=0.8, random_state=42)
LGB_COMMON = dict(n_estimators=200, max_depth=8, num_leaves=63,
                  learning_rate=0.05, min_child_samples=10,
                  subsample=0.8, subsample_freq=1, random_state=42,
                  verbosity=-1)


def build():
    df = pd.read_csv(CBE_CSV, low_memory=False)
    c = _collect_editor_rows(df, ['BE4'])
    c = c[c['reads'] >= MIN_READS].reset_index(drop=True)
    X, y = _df_to_Xy(c, n_features=125)
    groups = c['gRNA (20nt)'].to_numpy()
    return X, y, groups


def oof(X, y, groups, fit_predict):
    yhat = np.zeros_like(y)
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        m = fit_predict()
        m.fit(X[tr], y[tr])
        yhat[te] = np.clip(m.predict(X[te]), 0.0, 1.0)
    return yhat


def metrics(y, yhat):
    r2 = r2_score(y, yhat)
    rho, _ = spearmanr(y, yhat)
    # OLS slope of pred on obs: how much shrinkage. 1.0 = no shrinkage.
    a = np.polyfit(y, yhat, 1)
    return r2, rho, float(a[0])


def main():
    X, y, groups = build()
    print(f"BE4-only: n={len(y)}  unique spacers={len(np.unique(groups))}\n")

    models = [
        ('sklearn GBR  (mean, MSE)  [deployed]',
         lambda: GradientBoostingRegressor(**SKL_PARAMS)),
        ('LightGBM     (mean, MSE)',
         lambda: lgb.LGBMRegressor(objective='regression', **LGB_COMMON)),
        ('LightGBM     (median, quantile a=0.5)',
         lambda: lgb.LGBMRegressor(objective='quantile', alpha=0.5,
                                   **LGB_COMMON)),
    ]

    rows = []
    for label, factory in models:
        yhat = oof(X, y, groups, factory)
        r2, rho, slope = metrics(y, yhat)
        rows.append((label, r2, rho, slope))
        print(f"  {label}")
        print(f"    R2={r2:.3f}  rho={rho:.3f}  slope(pred~obs)={slope:.3f}\n")

    print("=" * 70)
    print(f"{'model':<42}{'R2':>7}{'rho':>8}{'slope':>8}")
    for label, r2, rho, slope in rows:
        print(f"{label:<42}{r2:>7.3f}{rho:>8.3f}{slope:>8.3f}")

    skl_r2, skl_rho, skl_sl = rows[0][1:]
    lgb_m_r2, lgb_m_rho, lgb_m_sl = rows[1][1:]
    lgb_q_r2, lgb_q_rho, lgb_q_sl = rows[2][1:]

    print("\n--- DECOMPOSITION ---")
    print(f"  Model-class effect (LGBM-mean vs sklearn GBR):")
    print(f"    R2 {lgb_m_r2-skl_r2:+.3f}  rho {lgb_m_rho-skl_rho:+.3f}  "
          f"slope {lgb_m_sl-skl_sl:+.3f}")
    print(f"  Loss effect (LGBM-quantile vs LGBM-mean):")
    print(f"    R2 {lgb_q_r2-lgb_m_r2:+.3f}  rho {lgb_q_rho-lgb_m_rho:+.3f}  "
          f"slope {lgb_q_sl-lgb_m_sl:+.3f}")

    print("\n--- INTERPRETATION GUIDE ---")
    print("  slope -> 1.0 = less shrinkage (extremes are not compressed).")
    print("  If quantile slope > mean slope AND quantile rho >= mean rho:")
    print("    -> quantile is a free win for the scatter without losing rank.")
    print("  If quantile slope > mean slope BUT quantile rho < mean rho:")
    print("    -> trade-off; pick by which metric matters in the paper "
          "(ranking favors mean; calibration favors quantile).")


if __name__ == '__main__':
    main()
