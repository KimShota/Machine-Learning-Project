"""src/model.py — XGBoost + CatBoost ensemble with walk-forward CV and calibration."""
import pickle
from pathlib import Path

import catboost as cb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from features import FEATURE_COLS

# XGBoost requires OpenMP on macOS (`brew install libomp`). Defer import and fall back if unavailable.
try:
    import xgboost as xgb

    HAVE_XGBOOST = True
except Exception:
    xgb = None
    HAVE_XGBOOST = False


def ranked_probability_score(y_true_int, probs):
    """RPS for 3-class ordered outcome (H=0, D=1, A=2). Lower is better."""
    n = len(y_true_int)
    rps = 0.0
    for i in range(n):
        actual = np.zeros(3); actual[y_true_int[i]] = 1.0
        cum_pred = np.cumsum(probs[i])
        cum_true = np.cumsum(actual)
        rps += np.sum((cum_pred - cum_true)**2) / 2
    return rps / n


def walk_forward_cv(df, feature_cols, n_splits=5):
    """
    Temporal walk-forward cross-validation.
    Splits by year: trains on years 1..k, validates on year k+1.
    """
    df = df.copy().sort_values("date")
    years = sorted(df["date"].dt.year.unique())
    split_years = years[-(n_splits+1):]  # last n_splits+1 years

    results = []
    for i in range(len(split_years)-1):
        cutoff = pd.Timestamp(f"{split_years[i+1]}-01-01")
        train = df[df.date < cutoff]
        val   = df[df.date >= cutoff]
        if len(train) < 100 or len(val) < 20:
            continue

        X_tr = train[feature_cols].fillna(0)
        y_tr = train["outcome_int"]
        X_val = val[feature_cols].fillna(0)
        y_val = val["outcome_int"]

        model = build_ensemble(X_tr, y_tr, verbose=False)
        probs = predict_proba(model, X_val)
        rps  = ranked_probability_score(y_val.values, probs)
        acc  = (probs.argmax(axis=1) == y_val.values).mean()
        results.append(dict(fold=i+1, train_size=len(train), val_size=len(val),
                            val_year=split_years[i+1], rps=round(rps,4), accuracy=round(acc,4)))
        print(f"    Fold {i+1} | val_year={split_years[i+1]} | "
              f"n_train={len(train):,} n_val={len(val):,} | "
              f"RPS={rps:.4f} acc={acc:.3f}")
    return pd.DataFrame(results)


def _build_tree_model(verbose=True):
    """Gradient boosting tree model; XGBoost if OpenMP/lib loads, else sklearn HGB."""
    if HAVE_XGBOOST:
        if verbose:
            print("    Training XGBoost …")
        return xgb.XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=42,
            verbosity=0,
        )
    if verbose:
        print(
            "    Training sklearn HistGradientBoosting (XGBoost unavailable — "
            "on macOS install OpenMP: brew install libomp) …"
        )
    return HistGradientBoostingClassifier(
        max_iter=400,
        max_depth=5,
        learning_rate=0.05,
        l2_regularization=1.0,
        random_state=42,
    )


def build_ensemble(X_train, y_train, verbose=True):
    """Train tree booster + CatBoost ensemble."""
    tree_model = _build_tree_model(verbose=verbose)
    cb_model = cb.CatBoostClassifier(
        iterations=500,
        depth=5,
        learning_rate=0.05,
        l2_leaf_reg=3.0,
        random_seed=42,
        loss_function="MultiClass",
        eval_metric="Accuracy",
        verbose=False,
    )
    tree_model.fit(X_train, y_train)
    if verbose:
        print("    Training CatBoost …")
    cb_model.fit(X_train, y_train)
    return {"xgb": tree_model, "cb": cb_model}


def predict_proba(ensemble, X):
    """Ensemble prediction: average tree booster and CatBoost probabilities."""
    X = X.fillna(0)
    p_tree = ensemble["xgb"].predict_proba(X)
    p_cb = ensemble["cb"].predict_proba(X)
    return (p_tree + p_cb) / 2.0


def calibrate(ensemble, X_cal, y_cal):
    """
    Post-hoc calibration using isotonic regression per class.
    Returns calibration functions.
    """
    raw_probs = predict_proba(ensemble, X_cal)
    calibrators = []
    for cls in range(3):
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(raw_probs[:, cls], (y_cal == cls).astype(float))
        calibrators.append(iso)
    return calibrators


def predict_calibrated(ensemble, calibrators, X):
    """Predict with calibration + renormalise to sum to 1."""
    raw = predict_proba(ensemble, X)
    cal = np.column_stack([
        calibrators[i].predict(raw[:, i]) for i in range(3)
    ])
    row_sums = cal.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    return cal / row_sums


def train_final_model(df, feature_cols, models_dir):
    """Train on all data, calibrate, save to disk."""
    models_dir = Path(models_dir)
    models_dir.mkdir(exist_ok=True)

    df = df.sort_values("date")
    # Use last 15% as calibration set
    cal_start = int(len(df) * 0.85)
    train_df  = df.iloc[:cal_start]
    cal_df    = df.iloc[cal_start:]

    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df["outcome_int"]
    X_cal   = cal_df[feature_cols].fillna(0)
    y_cal   = cal_df["outcome_int"]

    print("  Training final ensemble on full dataset …")
    ensemble = build_ensemble(X_train, y_train, verbose=True)
    print("  Calibrating probabilities …")
    calibrators = calibrate(ensemble, X_cal, y_cal)

    # Save
    with open(models_dir / "ensemble.pkl", "wb") as f:
        pickle.dump(ensemble, f)
    with open(models_dir / "calibrators.pkl", "wb") as f:
        pickle.dump(calibrators, f)

    # Feature importance (XGBoost / HGB may omit importances in some sklearn builds)
    tree = ensemble["xgb"]
    if hasattr(tree, "feature_importances_"):
        fi = pd.Series(tree.feature_importances_, index=feature_cols)
    else:
        cb_imp = np.array(ensemble["cb"].get_feature_importance(), dtype=float)
        cb_imp = cb_imp / cb_imp.sum() if cb_imp.sum() else cb_imp
        fi = pd.Series(cb_imp, index=feature_cols)
    fi = fi.sort_values(ascending=False)
    fi.to_csv(models_dir / "feature_importance.csv")
    print(f"  Models saved to {models_dir}")
    return ensemble, calibrators, fi


def load_model(models_dir):
    models_dir = Path(models_dir)
    with open(models_dir / "ensemble.pkl","rb") as f:
        ensemble = pickle.load(f)
    with open(models_dir / "calibrators.pkl","rb") as f:
        calibrators = pickle.load(f)
    return ensemble, calibrators
