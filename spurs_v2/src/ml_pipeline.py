"""
ml_pipeline.py
==============
Full ML pipeline for Premier League relegation prediction.

Components:
  1. Feature engineering
  2. Walk-forward cross-validation (no future leakage)
  3. Ensemble of Logistic Regression, Random Forest, XGBoost
  4. Probability calibration (isotonic regression)
  5. SHAP explanations
  6. Brier score + calibration curve
"""

import pandas as pd
import numpy as np
import json, os, warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
import shap


# ── Feature engineering ──────────────────────────────────────────────────────

FEATURE_COLS = [
    "pts_snap",
    "gd_snap",
    "form",
    "ppg",
    "projected_pts",
    "safety_buffer",
    "gdpg",
    "gf_pg",
    "ga_pg",
    "win_rate",
    "home_ppg",
    "away_ppg",
    "elo",
    "pts_needed_for_safety",
    "max_pts_possible",
    "survival_feasibility",
    "defensive_fragility",
    "attacking_output",
    "home_away_balance",
]

FEATURE_LABELS = {
    "pts_snap":             "Points at GW34",
    "gd_snap":              "Goal difference",
    "form":                 "Weighted form (last 10)",
    "ppg":                  "Points per game",
    "projected_pts":        "Projected final pts",
    "safety_buffer":        "Buffer above danger line",
    "gdpg":                 "GD per game",
    "gf_pg":                "Goals scored per game",
    "ga_pg":                "Goals conceded per game",
    "win_rate":             "Win rate",
    "home_ppg":             "Home PPG",
    "away_ppg":             "Away PPG",
    "elo":                  "Elo rating",
    "pts_needed_for_safety":"Pts needed for safety",
    "max_pts_possible":     "Max points possible",
    "survival_feasibility": "Survival feasibility",
    "defensive_fragility":  "Defensive fragility",
    "attacking_output":     "Attacking output",
    "home_away_balance":    "Home/away balance",
}

SAFETY_LINE = 36  # historical average safety threshold


def engineer_features(df):
    df = df.copy()
    df["pts_needed_for_safety"] = (SAFETY_LINE - df["pts_snap"]).clip(lower=0)
    df["max_pts_possible"]      = df["pts_snap"] + df["games_remaining"] * 3
    df["survival_feasibility"]  = (df["max_pts_possible"] / SAFETY_LINE).clip(0, 2)
    df["defensive_fragility"]   = df["ga_pg"] / df["gf_pg"].replace(0, 0.01)
    df["attacking_output"]      = df["gf_pg"] * df["win_rate"]
    df["home_away_balance"]     = df["home_ppg"] - df["away_ppg"]
    return df


# ── Walk-forward validation ──────────────────────────────────────────────────

def walk_forward_cv(df, models_dict, min_train_seasons=5):
    """
    For each season (in chronological order), train on all prior seasons
    and predict on that season. Returns per-season predictions and metrics.
    """
    seasons = sorted(df["season"].unique())
    all_preds = []

    for i, test_season in enumerate(seasons):
        if i < min_train_seasons:
            continue  # not enough history yet

        train = df[df["season"].isin(seasons[:i])].copy()
        test  = df[df["season"] == test_season].copy()

        if train["relegated"].sum() < 3:
            continue  # too few positive examples

        train = engineer_features(train)
        test  = engineer_features(test)

        X_train = train[FEATURE_COLS].fillna(0)
        y_train = train["relegated"]
        X_test  = test[FEATURE_COLS].fillna(0)
        y_test  = test["relegated"]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        season_probs = {}
        for name, base_model in models_dict.items():
            clf = CalibratedClassifierCV(base_model, method="isotonic", cv=3)
            clf.fit(X_train_s, y_train)
            probs = clf.predict_proba(X_test_s)[:, 1]
            season_probs[name] = probs

        ensemble = np.mean([v for v in season_probs.values()], axis=0)

        for j, (_, row) in enumerate(test.iterrows()):
            all_preds.append({
                "season":     test_season,
                "team":       row["team"],
                "relegated":  int(row["relegated"]),
                "ensemble":   round(float(ensemble[j]), 4),
                **{name: round(float(season_probs[name][j]), 4) for name in models_dict},
            })

    return pd.DataFrame(all_preds)


# ── Full model training ──────────────────────────────────────────────────────

def train_final_models(df):
    """Train calibrated ensemble on full historical dataset."""
    df = engineer_features(df)

    # Exclude current season (no labels yet)
    train = df[df["relegated"].notna() & (df["season"] != "2025-26")].copy()

    X = train[FEATURE_COLS].fillna(0)
    y = train["relegated"].astype(int)

    base_models = {
        "logistic":  LogisticRegression(C=0.3, max_iter=2000, random_state=42),
        "rf":        RandomForestClassifier(n_estimators=300, max_depth=5,
                                            min_samples_leaf=3, random_state=42),
        "gbm":       GradientBoostingClassifier(n_estimators=150, max_depth=3,
                                                learning_rate=0.04, random_state=42),
    }

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    calibrated = {}
    for name, model in base_models.items():
        clf = CalibratedClassifierCV(model, method="isotonic", cv=5)
        clf.fit(X_scaled, y)
        calibrated[name] = clf

    return calibrated, scaler, X, y, train


def predict_team(calibrated_models, scaler, team_features_dict):
    """Predict relegation probability for a single team."""
    row = pd.DataFrame([team_features_dict])
    row = engineer_features(row)
    X = row[FEATURE_COLS].fillna(0)
    X_scaled = scaler.transform(X)

    probs = {}
    for name, clf in calibrated_models.items():
        probs[name] = float(clf.predict_proba(X_scaled)[0, 1])

    probs["ensemble"] = float(np.mean(list(probs.values())))
    return probs


# ── SHAP explanations ────────────────────────────────────────────────────────

def compute_shap(calibrated_models, scaler, df_train, team_features_dict):
    """
    Compute SHAP values for Spurs using the Random Forest model
    (tree-based SHAP is exact and fast).
    """
    df_train = engineer_features(df_train.copy())
    train_exc = df_train[df_train["relegated"].notna() & (df_train["season"] != "2025-26")]
    X_bg = train_exc[FEATURE_COLS].fillna(0)
    X_bg_s = scaler.transform(X_bg)

    # Extract the underlying RF from the calibrated wrapper
    rf_cal = calibrated_models["rf"]

    # For SHAP we use a background sample
    bg = shap.sample(X_bg_s, 50)

    explainer = shap.KernelExplainer(
        lambda x: rf_cal.predict_proba(x)[:, 1],
        bg
    )

    row = pd.DataFrame([team_features_dict])
    row = engineer_features(row)
    X_row = scaler.transform(row[FEATURE_COLS].fillna(0))

    shap_vals = explainer.shap_values(X_row, nsamples=200)
    base_val  = float(explainer.expected_value)

    result = []
    for i, feat in enumerate(FEATURE_COLS):
        result.append({
            "feature":      feat,
            "label":        FEATURE_LABELS.get(feat, feat),
            "value":        float(row[feat].iloc[0]),
            "shap_value":   float(shap_vals[0][i]),
            "direction":    "increases risk" if shap_vals[0][i] > 0 else "decreases risk",
        })

    result.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
    return result, base_val


# ── Validation metrics ────────────────────────────────────────────────────────

def compute_validation_metrics(preds_df):
    """Compute AUC, Brier score, calibration for walk-forward predictions."""
    y_true  = preds_df["relegated"].values
    y_pred  = preds_df["ensemble"].values

    auc    = roc_auc_score(y_true, y_pred)
    brier  = brier_score_loss(y_true, y_pred)

    # Calibration curve
    frac_pos, mean_pred = calibration_curve(y_true, y_pred, n_bins=8, strategy="quantile")

    # Per-season AUC
    per_season = {}
    for season, grp in preds_df.groupby("season"):
        if grp["relegated"].nunique() == 2:
            per_season[season] = roc_auc_score(grp["relegated"], grp["ensemble"])

    return {
        "auc":            round(float(auc), 4),
        "brier":          round(float(brier), 4),
        "calibration":    {"frac_pos": frac_pos.tolist(), "mean_pred": mean_pred.tolist()},
        "per_season_auc": {k: round(v, 4) for k, v in per_season.items()},
    }


# ── Scenario analysis ────────────────────────────────────────────────────────

SCENARIOS = [
    {"name": "Win all 4",                "W": 4, "D": 0, "L": 0},
    {"name": "Win 3, draw 1",            "W": 3, "D": 1, "L": 0},
    {"name": "Win 2, draw 2",            "W": 2, "D": 2, "L": 0},
    {"name": "Win 2, draw 1, lose 1",    "W": 2, "D": 1, "L": 1},
    {"name": "Win 2, lose 2",            "W": 2, "D": 0, "L": 2},
    {"name": "Win 1, draw 2, lose 1",    "W": 1, "D": 2, "L": 1},
    {"name": "Win 1, draw 1, lose 2",    "W": 1, "D": 1, "L": 2},
    {"name": "Draw 2, lose 2",           "W": 0, "D": 2, "L": 2},
    {"name": "Draw 1, lose 3",           "W": 0, "D": 1, "L": 3},
    {"name": "Lose all 4",               "W": 0, "D": 0, "L": 4},
]

def compute_scenarios(calibrated_models, scaler, base_features):
    results = []
    avg_gd_per_game = 1.5  # rough per-game GD impact
    for s in SCENARIOS:
        pts_gained = s["W"] * 3 + s["D"]
        gd_delta   = s["W"] * avg_gd_per_game - s["L"] * avg_gd_per_game
        form_score = (s["W"] * 3 + s["D"]) / (4 * 3)

        f = dict(base_features)
        f["pts_snap"]    += pts_gained
        f["gd_snap"]     += gd_delta
        f["form"]         = min(1.0, form_score)
        f["ppg"]          = f["pts_snap"] / f["games_played"]
        f["projected_pts"]= f["pts_snap"]  # no more games
        f["safety_buffer"]= f["pts_snap"] - SAFETY_LINE
        f["gdpg"]         = f["gd_snap"] / f["games_played"]

        probs = predict_team(calibrated_models, scaler, f)
        results.append({
            "name":          s["name"],
            "W": s["W"], "D": s["D"], "L": s["L"],
            "final_pts":     int(base_features["pts_snap"] + pts_gained),
            "final_gd":      round(base_features["gd_snap"] + gd_delta, 1),
            "probability":   round(probs["ensemble"], 4),
        })
    return results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from data_generator import build_full_dataset, get_current_spurs

    print("Loading data...")
    df = build_full_dataset()
    spurs = get_current_spurs()

    print("Training models...")
    calibrated, scaler, X, y, train = train_final_models(df)

    print("Walk-forward validation...")
    base_models = {
        "logistic": LogisticRegression(C=0.3, max_iter=2000, random_state=42),
        "rf":       RandomForestClassifier(n_estimators=300, max_depth=5, random_state=42),
        "gbm":      GradientBoostingClassifier(n_estimators=150, max_depth=3,
                                               learning_rate=0.04, random_state=42),
    }
    preds = walk_forward_cv(df, base_models)
    metrics = compute_validation_metrics(preds)
    print(f"  AUC:   {metrics['auc']:.4f}")
    print(f"  Brier: {metrics['brier']:.4f}")

    print("Predicting Spurs...")
    probs = predict_team(calibrated, scaler, spurs)
    for k, v in probs.items():
        print(f"  {k:<12} {v:.1%}")

    print("Scenario analysis...")
    scens = compute_scenarios(calibrated, scaler, spurs)
    for s in scens:
        print(f"  {s['name']:<30} → {s['final_pts']} pts → {s['probability']:.1%}")
