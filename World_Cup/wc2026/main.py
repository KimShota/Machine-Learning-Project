"""
main.py — World Cup 2026 ML Prediction Pipeline
Run: python main.py
"""
import sys
sys.path.insert(0, "/home/claude/wc2026")

import time
import pandas as pd
from pathlib import Path

DATA_DIR    = Path("data")
MODELS_DIR  = Path("models")
OUTPUTS_DIR = Path("outputs")
for d in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR]: d.mkdir(exist_ok=True)

def banner(text):
    print(f"\n{'='*60}\n  {text}\n{'='*60}")

def main():
    t0 = time.time()
    banner("WC2026 ML Prediction — Full Pipeline")

    # ── PHASE 1: DATA ──────────────────────────────────────────────────────
    banner("Phase 1 — Data Generation")
    from src.generate_data import build_and_save_all
    data = build_and_save_all(DATA_DIR)
    matches = data["matches"]
    elo     = data["elo"]
    fifa    = data["fifa"]
    squad   = data["squad"]
    odds    = data["odds"]

    # ── PHASE 2: FEATURES ─────────────────────────────────────────────────
    banner("Phase 2 — Feature Engineering")
    from src.features import build_feature_matrix, FEATURE_COLS
    feature_df = build_feature_matrix(matches, elo, fifa, squad, odds)
    feature_df.to_csv(DATA_DIR / "merged_dataset.csv", index=False)
    print(f"  Saved → data/merged_dataset.csv  ({len(feature_df):,} rows, {len(FEATURE_COLS)} features)")

    # Filter to rows with enough feature data (drop first ~20 matches per team)
    model_df = feature_df.dropna(subset=["elo_diff","home_form_5","away_form_5"]).copy()
    print(f"  Training set: {len(model_df):,} matches (after warmup filter)")

    # ── PHASE 3: CROSS-VALIDATION ─────────────────────────────────────────
    banner("Phase 3 — Walk-Forward Cross-Validation")
    from src.model import walk_forward_cv
    cv_results = walk_forward_cv(model_df, FEATURE_COLS, n_splits=5)
    cv_results.to_csv(OUTPUTS_DIR / "cv_results.csv", index=False)
    print(f"\n  CV Summary:")
    print(f"  Mean RPS      : {cv_results.rps.mean():.4f}  (target < 0.20)")
    print(f"  Mean Accuracy : {cv_results.accuracy.mean()*100:.1f}%  (random baseline = 33%)")
    print(cv_results.to_string(index=False))

    # ── PHASE 4: TRAIN FINAL MODEL ────────────────────────────────────────
    banner("Phase 4 — Training Final Model")
    from src.model import train_final_model, predict_calibrated
    ensemble, calibrators, fi = train_final_model(model_df, FEATURE_COLS, MODELS_DIR)
    fi.to_csv(OUTPUTS_DIR / "feature_importance.csv")

    # ── PHASE 5: GENERATE WC2026 PREDICTIONS ─────────────────────────────
    banner("Phase 5 — WC2026 Match Predictions")
    from src.simulate import build_probs_lookup, monte_carlo, WC2026_GROUPS
    from src.report import (generate_all_group_matches, generate_group_predictions,
                             plot_winner_probabilities, plot_group_heatmap,
                             plot_feature_importance, plot_cv_results)

    print("  Building match probability lookup for all 48 teams …")
    probs_lookup = build_probs_lookup(feature_df, ensemble, calibrators, squad)

    # All group stage match predictions
    print("  Generating group stage match predictions …")
    group_matches = generate_all_group_matches(probs_lookup)
    group_matches.to_csv(OUTPUTS_DIR / "group_stage_predictions.csv", index=False)
    print(f"  Saved → outputs/group_stage_predictions.csv ({len(group_matches)} matches)")

    # Group qualification probabilities (20k simulations)
    print("  Simulating group qualification probabilities …")
    group_preds = generate_group_predictions(probs_lookup, n_sims=20000)
    group_preds.to_csv(OUTPUTS_DIR / "group_qualification_probs.csv", index=False)
    print(f"  Saved → outputs/group_qualification_probs.csv")

    # Monte Carlo tournament winner probabilities (50k simulations)
    banner("Phase 6 — Monte Carlo Tournament Simulation (50,000 runs)")
    mc_results = monte_carlo(probs_lookup, n_simulations=50000)
    mc_results.to_csv(OUTPUTS_DIR / "tournament_winner_probabilities.csv", index=False)
    print(f"\n  ╔══════════════════════════════════════════╗")
    print(f"  ║  Top 10 Tournament Winner Probabilities  ║")
    print(f"  ╠══════════════════════════════════════════╣")
    for _, row in mc_results.head(10).iterrows():
        bar = "█" * int(row.win_probability * 200)
        print(f"  ║  {row.team:<28s} {row.win_probability*100:>5.1f}%  ║")
    print(f"  ╚══════════════════════════════════════════╝")

    # ── PHASE 6: CHARTS ───────────────────────────────────────────────────
    banner("Phase 7 — Generating Charts")
    plot_winner_probabilities(mc_results, OUTPUTS_DIR / "01_winner_probabilities.png")
    plot_group_heatmap(group_preds, OUTPUTS_DIR / "02_group_advancement_heatmap.png")
    plot_feature_importance(fi, OUTPUTS_DIR / "03_feature_importance.png")
    plot_cv_results(cv_results, OUTPUTS_DIR / "04_cv_results.png")

    # ── SUMMARY ───────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    banner("Pipeline Complete")
    print(f"  Total runtime : {elapsed:.0f}s")
    print(f"\n  Output files:")
    for f in sorted(OUTPUTS_DIR.iterdir()):
        print(f"    outputs/{f.name}")
    print(f"\n  Data files:")
    for f in sorted(DATA_DIR.iterdir()):
        print(f"    data/{f.name}")
    print()

if __name__ == "__main__":
    main()
