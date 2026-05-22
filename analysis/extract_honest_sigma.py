"""Extract honest sigma (group-CV RMSE) for the deployed CBE and ABE
models. The bridge module's Phi-debiasing previously used
{ABE: 0.156, CBE: 0.151} -- both derived from leaky random-split
residuals. The leak-free sigma is the group-CV RMSE from the deployed
configurations.

Reuses the deployed _preprocess functions + DEPLOY params so the
sigma exactly matches what the deployed pkl achieves on leak-free
data.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error

from efficiencypredictorml import (
    _collect_editor_rows, _df_to_Xy, CBE_CSV, ABE_CSV, MIN_READ_COUNT,
)

CONFIGS = [
    ('CBE', CBE_CSV, ['BE4'], 125,
     dict(n_estimators=200, max_depth=8, learning_rate=0.05,
          min_samples_leaf=10, subsample=0.8)),
    ('ABE', ABE_CSV, ['ABE'], 113,
     dict(n_estimators=400, max_depth=8, learning_rate=0.03,
          min_samples_leaf=10, subsample=0.8)),
]


def sigma_for(csv_path, editors, n_features, params):
    df = pd.read_csv(csv_path, low_memory=False)
    c = _collect_editor_rows(df, editors)
    c = c[c['reads'] >= MIN_READ_COUNT].reset_index(drop=True)
    X, y = _df_to_Xy(c, n_features=n_features)
    groups = c['gRNA (20nt)'].to_numpy()
    rmses, all_resid = [], []
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        m = GradientBoostingRegressor(random_state=42, **params)
        m.fit(X[tr], y[tr])
        p = np.clip(m.predict(X[te]), 0.0, 1.0)
        rmses.append(np.sqrt(mean_squared_error(y[te], p)))
        all_resid.extend((y[te] - p).tolist())
    fold_rmse_mean = float(np.mean(rmses))
    fold_rmse_sd = float(np.std(rmses))
    pooled_resid_sd = float(np.std(all_resid))
    return fold_rmse_mean, fold_rmse_sd, pooled_resid_sd, len(y)


def main():
    print(f"{'mod':>4}  {'n':>6}  {'fold RMSE mean':>16}  "
          f"{'fold RMSE SD':>13}  {'pooled resid SD':>16}")
    for mod, csv, editors, nf, params in CONFIGS:
        m, s, ps, n = sigma_for(csv, editors, nf, params)
        print(f"{mod:>4}  {n:>6d}  {m:>16.4f}  {s:>13.4f}  {ps:>16.4f}")
    print("\n-> use 'pooled resid SD' as the symmetric Phi-debiasing sigma "
          "per modality. (Equivalent to RMSE-of-aggregate residuals.)")


if __name__ == '__main__':
    main()
