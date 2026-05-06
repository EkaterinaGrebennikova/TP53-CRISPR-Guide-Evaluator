import os, pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr

DATA_DIR  = os.path.join(os.path.dirname(__file__), '..', 'data')
CBE_CSV = os.path.join(DATA_DIR, 'mmc2.csv')
ABE_CSV = os.path.join(DATA_DIR, 'mmc3.csv')
CBE_MODEL = os.path.join(DATA_DIR, 'be_model_cbe.pkl')
ABE_MODEL = os.path.join(DATA_DIR, 'be_model_abe.pkl')

# Target columns — observed amino acid correction precision
CBE_TARGET_HEK = 'Obs aa correction precision among edited reads_HEK293T_BE4'
CBE_TARGET_MES = 'Obs aa correction precision among edited reads_mES_BE4'
ABE_TARGET_HEK = 'Obs aa correction precision among edited reads_HEK293T_ABE'
ABE_TARGET_MES = 'Obs aa correction precision among edited reads_mES_ABE'

# Minimum read count to trust observed efficiency
MIN_READ_COUNT = 100

_cbe_model = None
_abe_model = None


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def one_hot(seq):
    """One-hot encode a nucleotide sequence. Returns list of length 4*len(seq)."""
    mapping = {'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,1]}
    out = []
    for s in seq.upper():
        out.extend(mapping.get(s, [0,0,0,0]))
    return out


def _dinucleotide_freqs(seq):
    """Count all 16 dinucleotide frequencies in a sequence."""
    dinucs = {}
    seq = seq.upper()
    for i in range(len(seq) - 1):
        dn = seq[i:i+2]
        dinucs[dn] = dinucs.get(dn, 0) + 1
    total = max(len(seq) - 1, 1)
    bases = 'ACGT'
    return [dinucs.get(a+b, 0) / total for a in bases for b in bases]


def _melting_temp(seq):
    """Rough Wallace-rule Tm for short oligos: 2*(A+T) + 4*(G+C)."""
    seq = seq.upper()
    at = sum(1 for s in seq if s in 'AT')
    gc = sum(1 for s in seq if s in 'GC')
    return 2 * at + 4 * gc


def extract_features(spacer, edit_position, cell_type='HEK293T'):
    """Build feature vector for a single guide.

    Features (95 total):
      - One-hot encoded 20bp spacer          (80)
      - GC content                            (1)
      - Normalised edit position              (1)
      - Local trinucleotide context one-hot   (12)
      - Dinucleotide frequencies              (16)
      - Melting temperature (normalised)      (1)
      - Number of substrate C/A in window 4-8 (1)
      - Cell type flag (0=mES, 1=HEK293T)    (1)
    """
    spacer = spacer.upper()
    feat = []

    # One-hot spacer (80)
    feat.extend(one_hot(spacer))

    # GC content (1)
    gc = sum(1 for s in spacer if s in 'GC') / len(spacer)
    feat.append(gc)

    # Normalised edit position (1)
    feat.append((edit_position - 1) / 19.0)

    # Local trinucleotide context (12)
    i = edit_position - 1
    left  = spacer[i - 1] if i > 0 else 'N'
    mid   = spacer[i]
    right = spacer[i + 1] if i < len(spacer) - 1 else 'N'
    feat.extend(one_hot(left))
    feat.extend(one_hot(mid))
    feat.extend(one_hot(right))

    # Dinucleotide frequencies (16)
    feat.extend(_dinucleotide_freqs(spacer))

    # Melting temperature, normalised to [0,1] (1)
    feat.append(_melting_temp(spacer) / 100.0)

    # Substrate nucleotides in editing window positions 4-8 (1)
    window = spacer[3:8]  # 0-indexed positions 3-7 = protospacer 4-8
    # For CBE: count C's; for ABE: count A's — caller doesn't know modality here
    # so count both and let the model learn which matters
    feat.append(sum(1 for s in window if s in 'CA') / len(window))

    # Cell type (1)
    feat.append(1.0 if cell_type == 'HEK293T' else 0.0)

    return feat


# ---------------------------------------------------------------------------
# Data preprocessing
# ---------------------------------------------------------------------------

def _preprocess_cbe():
    """Load and clean CBE (BE4) training data from Koblan et al. mmc2.csv.

    Combines HEK293T and mES observations, filters by read count,
    returns (X feature matrix, y target array).
    """
    df = pd.read_csv(CBE_CSV, low_memory=False)

    rows = []
    # HEK293T BE4
    hek = df[['gRNA (20nt)', 'Editing position',
              CBE_TARGET_HEK, 'Total obs read count_HEK293T_BE4']].copy()
    hek = hek.rename(columns={CBE_TARGET_HEK: 'target',
                               'Total obs read count_HEK293T_BE4': 'reads'})
    hek['cell_type'] = 'HEK293T'
    rows.append(hek)

    # mES BE4
    mes = df[['gRNA (20nt)', 'Editing position',
              CBE_TARGET_MES, 'Total obs read count_mES_BE4']].copy()
    mes = mes.rename(columns={CBE_TARGET_MES: 'target',
                               'Total obs read count_mES_BE4': 'reads'})
    mes['cell_type'] = 'mES'
    rows.append(mes)

    combined = pd.concat(rows, ignore_index=True)
    combined['reads'] = pd.to_numeric(combined['reads'], errors='coerce')
    combined = combined.dropna(subset=['target', 'gRNA (20nt)'])
    combined = combined[combined['reads'] >= MIN_READ_COUNT]
    combined = combined[combined['gRNA (20nt)'].str.len() == 20]

    X = np.array([
        extract_features(row['gRNA (20nt)'], int(row['Editing position']), row['cell_type'])
        for _, row in combined.iterrows()
    ])
    y = combined['target'].values.astype(float)
    return X, y


def _preprocess_abe():
    """Load and clean ABE training data from Koblan et al. mmc3.csv.

    Combines HEK293T and mES observations, filters by read count,
    returns (X feature matrix, y target array).
    """
    df = pd.read_csv(ABE_CSV, low_memory=False)

    rows = []
    # HEK293T ABE
    hek = df[['gRNA (20nt)', 'Editing position',
              ABE_TARGET_HEK, 'Total obs read count_HEK293T_ABE']].copy()
    hek = hek.rename(columns={ABE_TARGET_HEK: 'target',
                               'Total obs read count_HEK293T_ABE': 'reads'})
    hek['cell_type'] = 'HEK293T'
    rows.append(hek)

    # mES ABE
    mes = df[['gRNA (20nt)', 'Editing position',
              ABE_TARGET_MES, 'Total obs read count_mES_ABE']].copy()
    mes = mes.rename(columns={ABE_TARGET_MES: 'target',
                               'Total obs read count_mES_ABE': 'reads'})
    mes['cell_type'] = 'mES'
    rows.append(mes)

    combined = pd.concat(rows, ignore_index=True)
    combined['reads'] = pd.to_numeric(combined['reads'], errors='coerce')
    combined = combined.dropna(subset=['target', 'gRNA (20nt)'])
    combined = combined[combined['reads'] >= MIN_READ_COUNT]
    combined = combined[combined['gRNA (20nt)'].str.len() == 20]

    X = np.array([
        extract_features(row['gRNA (20nt)'], int(row['Editing position']), row['cell_type'])
        for _, row in combined.iterrows()
    ])
    y = combined['target'].values.astype(float)
    return X, y


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(modality='CBE', n_estimators=200, max_depth=8, learning_rate=0.05,
                subsample=0.8, random_state=42):
    """Train a GradientBoosting regressor for the given modality.

    Returns:
        dict with keys: model, r2, rmse, spearman, cv_r2_mean, cv_r2_std, n_train, n_test
    """
    preprocess = _preprocess_cbe if modality == 'CBE' else _preprocess_abe
    X, y = preprocess()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state,
    )

    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        random_state=random_state,
    )

    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 0.0, 1.0)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rho, _ = spearmanr(y_test, y_pred)

    return {
        'model': model,
        'r2': r2,
        'rmse': rmse,
        'spearman': rho,
        'cv_r2_mean': cv_scores.mean(),
        'cv_r2_std': cv_scores.std(),
        'n_train': len(X_train),
        'n_test': len(X_test),
    }


def tune_hyperparameters(modality='CBE', random_state=42):
    """Grid search over hyperparameter combinations, return best result.

    Tries ~108 combinations of n_estimators, max_depth, learning_rate, subsample.
    Selects the combination with the best 5-fold CV R² on the training set.
    """
    preprocess = _preprocess_cbe if modality == 'CBE' else _preprocess_abe
    X, y = preprocess()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state,
    )

    param_grid = {
        'n_estimators':  [100, 200, 400],
        'max_depth':     [4, 6, 8, 12],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample':     [0.7, 0.8, 1.0],
    }

    best_cv = -np.inf
    best_params = None
    best_model = None
    total = 1
    for v in param_grid.values():
        total *= len(v)

    print(f"  Testing {total} parameter combinations...")
    tried = 0
    for n_est in param_grid['n_estimators']:
        for depth in param_grid['max_depth']:
            for lr in param_grid['learning_rate']:
                for sub in param_grid['subsample']:
                    tried += 1
                    model = GradientBoostingRegressor(
                        n_estimators=n_est, max_depth=depth,
                        learning_rate=lr, subsample=sub,
                        random_state=random_state,
                    )
                    cv = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
                    mean_cv = cv.mean()
                    if mean_cv > best_cv:
                        best_cv = mean_cv
                        best_params = {
                            'n_estimators': n_est, 'max_depth': depth,
                            'learning_rate': lr, 'subsample': sub,
                        }
                        best_model = model
                    if tried % 18 == 0:
                        print(f"    {tried}/{total} done (best CV R² so far: {best_cv:.3f})")

    print(f"  Best params: {best_params}")
    print(f"  Best CV R²:  {best_cv:.3f}")

    best_model.fit(X_train, y_train)
    y_pred = np.clip(best_model.predict(X_test), 0.0, 1.0)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rho, _ = spearmanr(y_test, y_pred)

    return {
        'model': best_model,
        'params': best_params,
        'r2': r2,
        'rmse': rmse,
        'spearman': rho,
        'cv_r2': best_cv,
        'n_train': len(X_train),
        'n_test': len(X_test),
    }


def train_and_save_all(tune=False):
    """Train both CBE and ABE models, print metrics, save to disk.

    Args:
        tune: if True, run hyperparameter tuning (slower but better accuracy).
    """
    for modality, path in [('CBE', CBE_MODEL), ('ABE', ABE_MODEL)]:
        print(f"\n{'='*50}")
        print(f"Training {modality} model{'  (with tuning)' if tune else ''}...")
        print(f"{'='*50}")

        if tune:
            result = tune_hyperparameters(modality=modality)
            print(f"  Best params:   {result['params']}")
            print(f"  Best CV R²:    {result['cv_r2']:.3f}")
        else:
            result = train_model(modality=modality)
            print(f"  5-fold CV R²:  {result['cv_r2_mean']:.3f} +/- {result['cv_r2_std']:.3f}")

        print(f"  Train samples: {result['n_train']}")
        print(f"  Test samples:  {result['n_test']}")
        print(f"  Test R²:       {result['r2']:.3f}")
        print(f"  Test RMSE:     {result['rmse']:.3f}")
        print(f"  Test Spearman: {result['spearman']:.3f}")

        with open(path, 'wb') as f:
            pickle.dump(result['model'], f)
        print(f"  Saved model to {path}")


# ---------------------------------------------------------------------------
# Prediction (inference)
# ---------------------------------------------------------------------------

def _load_model(modality):
    """Load a trained model from disk. Returns the sklearn estimator."""
    global _cbe_model, _abe_model
    if modality == 'CBE':
        if _cbe_model is None:
            with open(CBE_MODEL, 'rb') as f:
                _cbe_model = pickle.load(f)
        return _cbe_model
    else:
        if _abe_model is None:
            with open(ABE_MODEL, 'rb') as f:
                _abe_model = pickle.load(f)
        return _abe_model


def predict_efficiency(spacer, edit_position, modality='CBE', cell_type='HEK293T'):
    """Predict AA-level correction efficiency for a single guide.

    Args:
        spacer: 20nt guide sequence
        edit_position: 1-based position of target edit in spacer
        modality: 'CBE' or 'ABE'
        cell_type: 'HEK293T' or 'mES'

    Returns:
        float in [0, 1] — predicted AA correction precision
    """
    model = _load_model(modality)
    features = np.array([extract_features(spacer, edit_position, cell_type)])
    pred = model.predict(features)[0]
    return float(np.clip(pred, 0.0, 1.0))


# ---------------------------------------------------------------------------
# CLI entry point for training
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    tune = '--tune' in sys.argv
    train_and_save_all(tune=tune)