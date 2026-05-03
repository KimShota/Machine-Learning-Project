"""
Model Training — XGBoost + CatBoost ensemble with walk-forward CV.
Outputs calibrated 3-class probabilities (H / D / A) per match.
"""
import json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, accuracy_score
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import catboost as cb
import pickle, json

warnings.filterwarnings("ignore")
DATA    = Path("/home/claude/wc2026/data")
MODELS  = Path("/home/claude/wc2026/models")
OUTPUTS = Path("/home/claude/wc2026/outputs")
MODELS.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)

FEATURE_COLS = [
    "elo_diff", "elo_home_win_prob",
    "home_fifa_rank", "away_fifa_rank", "fifa_rank_diff",
    "home_form_5", "home_gf_5", "home_ga_5", "home_clean_5",
    "home_form_10", "home_gf_10", "home_ga_10", "home_clean_10",
    "away_form_5", "away_gf_5", "away_ga_5", "away_clean_5",
    "away_form_10", "away_gf_10", "away_ga_10", "away_clean_10",
    "h2h_home_win_rate", "h2h_n",
    "is_wc", "is_friendly", "is_neutral",
    "home_rest_days", "away_rest_days",
    "month",
]
TARGET = "outcome"
LABEL_MAP = {"H": 2, "D": 1, "A": 0}
LABEL_MAP_INV = {v: k for k, v in LABEL_MAP.items()}


def ranked_probability_score(y_true_int, probs):
    """
    Ranked Probability Score — the standard football prediction metric.
    Lower is better. Penalises confident wrong predictions.
    """
    n = len(y_true_int)
    rps_total = 0.0
    for i in range(n):
        true_label = y_true_int[i]
        p = probs[i]   # [p_A, p_D, p_H]
        # cumulative probabilities
        cp = np.cumsum(p)
        # one-hot cumulative
        oh = np.zeros(3)
        oh[true_label:] = 1.0
        rps_total += np.sum((cp - oh) ** 2) / 2
    return rps_total / n


def load_data():
    df = pd.read_csv(DATA / "merged_dataset.csv", parse_dates=["date"])
    df["outcome_int"] = df[TARGET].map(LABEL_MAP)
    # Fill missing features with median
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = 0.0
    return df.dropna(subset=["outcome_int"]).reset_index(drop=True)


def walk_forward_eval(df, n_splits=5):
    """
    Time-based walk-forward cross-validation.
    Train on years 1993-Y, validate on Y+1.
    Evaluates both XGBoost and CatBoost independently.
    """
    df = df.sort_values("date")
    years = sorted(df["year"].unique())
    split_years = years[-(n_splits+1):-1]   # last N+1 years

    results = []
    print(f"\n{'='*62}")
    print(f"  Walk-Forward Cross-Validation ({n_splits} folds)")
    print(f"{'='*62}")
    print(f"  {'Fold':<6} {'Train up to':<14} {'Val year':<10} {'XGB RPS':<10} {'CB RPS':<10} {'Ens RPS':<10} {'Acc'}")
    print(f"  {'-'*62}")

    for val_year in split_years:
        train = df[df["year"] < val_year]
        val   = df[df["year"] == val_year]
        if len(val) < 50:
            continue

        X_tr = train[FEATURE_COLS].values
        y_tr = train["outcome_int"].values.astype(int)
        X_val = val[FEATURE_COLS].values
        y_val = val["outcome_int"].values.astype(int)

        # XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="mlogloss",
            random_state=42, verbosity=0
        )
        xgb_model.fit(X_tr, y_tr)
        xgb_probs = xgb_model.predict_proba(X_val)

        # CatBoost
        cb_model = cb.CatBoostClassifier(
            iterations=400, depth=6, learning_rate=0.05,
            loss_function="MultiClass", random_seed=42, verbose=0
        )
        cb_model.fit(X_tr, y_tr)
        cb_probs = cb_model.predict_proba(X_val)

        # Ensemble (equal weight)
        ens_probs = (xgb_probs + cb_probs) / 2

        xgb_rps = ranked_probability_score(y_val, xgb_probs)
        cb_rps  = ranked_probability_score(y_val, cb_probs)
        ens_rps = ranked_probability_score(y_val, ens_probs)
        acc     = accuracy_score(y_val, ens_probs.argmax(axis=1))

        results.append({"year": val_year, "xgb_rps": xgb_rps,
                         "cb_rps": cb_rps, "ens_rps": ens_rps, "acc": acc,
                         "n_train": len(train), "n_val": len(val)})
        print(f"  {len(results):<6} {val_year-1!s:<14} {val_year!s:<10} "
              f"{xgb_rps:.4f}    {cb_rps:.4f}    {ens_rps:.4f}    {acc:.3f}")

    mean_rps = np.mean([r["ens_rps"] for r in results])
    mean_acc = np.mean([r["acc"] for r in results])
    print(f"  {'-'*62}")
    print(f"  {'MEAN':<6} {'':14} {'':10} {'':10} {'':10} {mean_rps:.4f}    {mean_acc:.3f}")
    print(f"{'='*62}\n")
    return results


def train_final_model(df):
    """Train final models on ALL historical data."""
    print("Training final ensemble on full dataset...")
    X = df[FEATURE_COLS].values
    y = df["outcome_int"].values.astype(int)

    # XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=600, max_depth=5, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric="mlogloss",
        random_state=42, verbosity=0
    )
    xgb_model.fit(X, y)

    # CatBoost
    cb_model = cb.CatBoostClassifier(
        iterations=600, depth=6, learning_rate=0.04,
        loss_function="MultiClass", random_seed=42, verbose=0
    )
    cb_model.fit(X, y)

    print(f"  XGBoost trained on {len(X):,} samples")
    print(f"  CatBoost trained on {len(X):,} samples")

    # Save models
    with open(MODELS / "xgb_model.pkl", "wb") as f:
        pickle.dump(xgb_model, f)
    with open(MODELS / "cb_model.pkl", "wb") as f:
        pickle.dump(cb_model, f)
    print(f"  Models saved → {MODELS}/")

    return xgb_model, cb_model


def predict_match(xgb_model, cb_model, features: dict) -> dict:
    """
    Predict a single match.
    features: dict with keys = FEATURE_COLS
    Returns: {"H": p_h, "D": p_d, "A": p_a, "predicted": "H"/"D"/"A"}
    """
    row = np.array([[features.get(c, 0.0) for c in FEATURE_COLS]])
    xgb_p = xgb_model.predict_proba(row)[0]
    cb_p  = cb_model.predict_proba(row)[0]
    ens   = (xgb_p + cb_p) / 2
    return {
        "A": round(ens[0], 4),
        "D": round(ens[1], 4),
        "H": round(ens[2], 4),
        "predicted": LABEL_MAP_INV[ens.argmax()],
        "confidence": round(ens.max(), 4),
    }


def plot_feature_importance(xgb_model, cb_model):
    """Bar chart of top feature importances (average of both models)."""
    xgb_imp = dict(zip(FEATURE_COLS, xgb_model.feature_importances_))
    cb_imp  = dict(zip(FEATURE_COLS, cb_model.get_feature_importance()))
    # normalise CB importance
    cb_total = sum(cb_imp.values())
    cb_imp   = {k: v/cb_total for k, v in cb_imp.items()}

    avg = {f: (xgb_imp.get(f,0) + cb_imp.get(f,0)) / 2 for f in FEATURE_COLS}
    avg = dict(sorted(avg.items(), key=lambda x: -x[1])[:15])

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#1A7A4A" if v > np.mean(list(avg.values())) else "#9ED5B5" for v in avg.values()]
    bars = ax.barh(list(avg.keys())[::-1], list(avg.values())[::-1], color=colors[::-1])
    ax.set_xlabel("Average feature importance", fontsize=11)
    ax.set_title("Top 15 features (XGBoost + CatBoost average)", fontsize=12, fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUTPUTS / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Feature importance chart → {OUTPUTS}/feature_importance.png")


def plot_cv_results(cv_results):
    """Line chart of RPS across CV folds."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    years = [r["year"] for r in cv_results]
    ax1.plot(years, [r["xgb_rps"] for r in cv_results], "o-", label="XGBoost", color="#185FA5")
    ax1.plot(years, [r["cb_rps"]  for r in cv_results], "s-", label="CatBoost", color="#854F0B")
    ax1.plot(years, [r["ens_rps"] for r in cv_results], "^-", label="Ensemble", color="#1A7A4A", lw=2)
    ax1.axhline(0.25, ls="--", color="gray", alpha=0.5, label="Random baseline")
    ax1.set_xlabel("Validation year"); ax1.set_ylabel("RPS (lower = better)")
    ax1.set_title("Ranked Probability Score by fold"); ax1.legend(); ax1.grid(alpha=0.3)
    ax1.spines[["top","right"]].set_visible(False)

    ax2.bar(years, [r["acc"] for r in cv_results], color="#1A7A4A", alpha=0.8)
    ax2.axhline(0.333, ls="--", color="gray", alpha=0.5, label="Random (33%)")
    ax2.axhline(0.45,  ls="--", color="orange", alpha=0.6, label="Naive baseline (45%)")
    ax2.set_xlabel("Validation year"); ax2.set_ylabel("3-way accuracy")
    ax2.set_title("Accuracy by fold"); ax2.legend(); ax2.set_ylim(0.2, 0.7); ax2.grid(alpha=0.3)
    ax2.spines[["top","right"]].set_visible(False)

    plt.suptitle("Walk-Forward Cross-Validation Results", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUTS / "cv_results.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  CV results chart → {OUTPUTS}/cv_results.png")


def main():
    print("\n" + "="*62)
    print("  WC2026 ML Prediction Model — Training Pipeline")
    print("="*62)

    df = load_data()
    print(f"Dataset: {len(df):,} matches | {len(FEATURE_COLS)} features | years {df['year'].min()}–{df['year'].max()}")
    print(f"Outcome distribution: H={( df['outcome']=='H').mean():.1%}  D={(df['outcome']=='D').mean():.1%}  A={(df['outcome']=='A').mean():.1%}")

    # Walk-forward CV
    cv_results = walk_forward_eval(df, n_splits=5)

    # Final model on full data
    xgb_model, cb_model = train_final_model(df)

    # Charts
    print("\nGenerating charts...")
    plot_feature_importance(xgb_model, cb_model)
    plot_cv_results(cv_results)

    # Save CV summary
    summary = {
        "mean_rps": round(np.mean([r["ens_rps"] for r in cv_results]), 4),
        "mean_accuracy": round(np.mean([r["acc"] for r in cv_results]), 4),
        "folds": cv_results,
        "feature_cols": FEATURE_COLS,
    }
    with open(MODELS / "cv_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Training complete.")
    print(f"  Mean RPS  : {summary['mean_rps']:.4f}  (target < 0.22)")
    print(f"  Mean acc  : {summary['mean_accuracy']:.1%}  (baseline 33%)")
    print(f"  Models    : {MODELS}/")
    print(f"  Charts    : {OUTPUTS}/")


if __name__ == "__main__":
    main()
