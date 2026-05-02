"""
run.py
======
Master orchestration script. Run this to execute the full pipeline:

  python run.py

Outputs:
  outputs/results.json  — all model outputs, metrics, SHAP, scenarios
"""

import sys, os, json, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from data_generator import build_full_dataset, get_current_spurs, get_current_rivals
from ml_pipeline import (
    train_final_models, walk_forward_cv, predict_team,
    compute_validation_metrics, compute_scenarios, engineer_features,
    FEATURE_COLS, FEATURE_LABELS, compute_shap
)
from monte_carlo import run_simulation, get_match_probabilities, CURRENT_STANDINGS

BANNER = "=" * 62

def section(title):
    print(f"\n{'─'*62}")
    print(f"  {title}")
    print(f"{'─'*62}")

def main():
    t0 = time.time()
    print(BANNER)
    print("  SPURS RELEGATION PREDICTION SYSTEM  v2.0")
    print("  2025-26 Premier League Season · GW34")
    print(BANNER)

    # ── 1. Data ──────────────────────────────────────────────────────────────
    section("1 / 6  Building dataset")
    df = build_full_dataset(out_path="data/pl_dataset.csv")
    spurs = get_current_spurs()
    rivals = get_current_rivals()
    print(f"  {len(df)} team-seasons | "
          f"{df['season'].nunique()} seasons | "
          f"{df['relegated'].sum()} relegated cases")

    # ── 2. Walk-forward validation ────────────────────────────────────────────
    section("2 / 6  Walk-forward validation (no data leakage)")
    base_models = {
        "logistic": LogisticRegression(C=0.3, max_iter=2000, random_state=42),
        "rf":       RandomForestClassifier(n_estimators=300, max_depth=5,
                                           min_samples_leaf=3, random_state=42),
        "gbm":      GradientBoostingClassifier(n_estimators=150, max_depth=3,
                                               learning_rate=0.04, random_state=42),
    }
    wf_preds = walk_forward_cv(df, base_models, min_train_seasons=5)
    metrics  = compute_validation_metrics(wf_preds)

    print(f"  AUC (walk-forward):   {metrics['auc']:.4f}")
    print(f"  Brier score:          {metrics['brier']:.4f}  (lower = better calibration)")
    print(f"  Per-season AUC range: "
          f"{min(metrics['per_season_auc'].values()):.3f} – "
          f"{max(metrics['per_season_auc'].values()):.3f}")

    # ── 3. Train final models ─────────────────────────────────────────────────
    section("3 / 6  Training calibrated ensemble")
    calibrated, scaler, X_train, y_train, train_df = train_final_models(df)
    print(f"  Training set: {len(X_train)} samples, {y_train.sum()} relegated")
    print(f"  Models: Logistic Regression | Random Forest | Gradient Boosting")
    print(f"  Calibration: Isotonic regression (5-fold)")

    # ── 4. Spurs prediction ───────────────────────────────────────────────────
    section("4 / 6  Predicting Tottenham Hotspur 2025-26")
    ml_probs = predict_team(calibrated, scaler, spurs)

    print(f"\n  {'MODEL':<20} {'PROB':>8}")
    print(f"  {'─'*30}")
    for name, prob in ml_probs.items():
        bar = "█" * int(prob * 20)
        marker = " ◄ ensemble" if name == "ensemble" else ""
        print(f"  {name:<20} {prob:>7.1%}  {bar}{marker}")

    # ── 5. Monte Carlo simulation ─────────────────────────────────────────────
    section("5 / 6  Monte Carlo season simulation (100,000 runs)")
    mc_results     = run_simulation(n_sims=100_000)
    match_probs    = get_match_probabilities()
    spurs_mc_prob  = mc_results["Tottenham"]["relegation_prob"]

    print(f"\n  Bottom-zone relegation probabilities:")
    for team, r in sorted(mc_results.items(),
                          key=lambda x: x[1]["relegation_prob"], reverse=True):
        bar = "█" * int(r["relegation_prob"] * 30)
        print(f"  {team:<16} {r['relegation_prob']:>6.1%}  {bar}")

    print(f"\n  Spurs remaining fixtures:")
    for m in match_probs:
        print(f"    {m['venue']:4} vs {(m['home'] if m['venue']=='Away' else m['away']):<18} "
              f"W:{m['spurs_win_prob']:.1%} D:{m['draw_prob']:.1%} L:{m['spurs_loss_prob']:.1%}  "
              f"xG: {m['lam_spurs']:.1f}–{m['lam_opp']:.1f}")

    # Final combined estimate
    final_prob = ml_probs["ensemble"] * 0.4 + spurs_mc_prob * 0.6
    print(f"\n  {'─'*40}")
    print(f"  ML Ensemble:          {ml_probs['ensemble']:>6.1%}")
    print(f"  Monte Carlo:          {spurs_mc_prob:>6.1%}")
    print(f"  COMBINED ESTIMATE:    {final_prob:>6.1%}  (40% ML + 60% MC)")
    print(f"  {'─'*40}")

    # ── 6. SHAP explanations ──────────────────────────────────────────────────
    section("6 / 6  SHAP feature explanations")
    print("  Computing SHAP values (this takes ~30s)...")
    try:
        shap_values, base_val = compute_shap(calibrated, scaler, df, spurs)
        print(f"\n  Base rate (avg relegation prob): {base_val:.1%}")
        print(f"  {'FEATURE':<30} {'SHAP':>8}  {'IMPACT'}")
        print(f"  {'─'*60}")
        for s in shap_values[:10]:
            arrow = "▲" if s["shap_value"] > 0 else "▼"
            print(f"  {s['label']:<30} {s['shap_value']:>+7.3f}  {arrow} {s['direction']}")
        shap_ok = True
    except Exception as e:
        print(f"  SHAP skipped ({e})")
        shap_values, base_val, shap_ok = [], 0.5, False

    # ── Scenarios ─────────────────────────────────────────────────────────────
    scenarios = compute_scenarios(calibrated, scaler, spurs)
    print(f"\n  Scenario analysis:")
    for s in scenarios:
        emoji = "🟢" if s["probability"] < 0.25 else ("🟡" if s["probability"] < 0.55 else "🔴")
        print(f"  {emoji} {s['name']:<30} → {s['final_pts']} pts → {s['probability']:.1%}")

    # ── Save all results ──────────────────────────────────────────────────────
    os.makedirs("outputs", exist_ok=True)
    output = {
        "meta": {
            "season": "2025-26",
            "gameweek": 34,
            "run_time_seconds": round(time.time() - t0, 1),
        },
        "validation": {
            **metrics,
            "walk_forward_predictions": wf_preds.to_dict(orient="records"),
        },
        "spurs_current": spurs,
        "ml_probabilities": ml_probs,
        "monte_carlo": {
            team: {k: v for k, v in r.items() if k != "pts_distribution"}
            for team, r in mc_results.items()
        },
        "match_probabilities": match_probs,
        "final_probability": round(float(final_prob), 4),
        "shap_values": shap_values if shap_ok else [],
        "shap_base_value": float(base_val),
        "scenarios": scenarios,
        "rivals": rivals,
        "current_standings": CURRENT_STANDINGS,
    }

    with open("outputs/results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\n{BANNER}")
    print(f"  DONE in {elapsed:.1f}s")
    print(f"  Final estimate: Spurs relegation probability = {final_prob:.1%}")
    if final_prob > 0.6:
        verdict = "VERY HIGH RISK — in the drop zone with difficult run-in"
    elif final_prob > 0.4:
        verdict = "HIGH RISK — survival possible but form must improve"
    elif final_prob > 0.2:
        verdict = "MODERATE RISK — likely to survive but not comfortable"
    else:
        verdict = "LOW RISK — survival expected"
    print(f"  Verdict: {verdict}")
    print(f"  Results saved → outputs/results.json")
    print(BANNER)

    return output

if __name__ == "__main__":
    main()
