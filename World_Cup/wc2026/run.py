"""
WC2026 ML Prediction — Main entry point
Run this file to execute the full pipeline end-to-end:
  1. Generate / load data
  2. Engineer features
  3. Train model + cross-validate
  4. Simulate tournament + generate predictions

Usage:
  python run.py                     # full pipeline
  python run.py --skip-data         # skip raw CSV generation (use existing data/*.csv)
  python run.py --skip-train        # skip training (needs models/ + merged_dataset.csv)
  python run.py --simulate-only     # only re-run tournament simulation (needs models/)
"""
import argparse
import sys
import time
from pathlib import Path

BASE = Path(__file__).parent.resolve()
sys.path.insert(0, str(BASE / "src"))

DATA_DIR = BASE / "data"
MODELS_DIR = BASE / "models"
OUTPUTS_DIR = BASE / "outputs"


def banner(msg):
    print(f"\n{'='*62}\n  {msg}\n{'='*62}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-data", action="store_true", help="Skip synthetic raw CSV generation")
    parser.add_argument("--skip-train", action="store_true", help="Skip model training & CV")
    parser.add_argument("--simulate-only", action="store_true", help="Only run WC2026 simulation outputs")
    args = parser.parse_args()

    t0 = time.time()
    print("\n" + "█" * 62)
    print("  ⚽  WC2026 ML Prediction Pipeline")
    print("  From raw data → trained model → tournament predictions")
    print("█" * 62)

    for d in (DATA_DIR, MODELS_DIR, OUTPUTS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    if args.simulate_only:
        banner("WC2026 Predictions (simulation only)")
        from simulate import main as sim_main

        sim_main()
    else:
        if not args.skip_data:
            banner("Step 1 / 4  —  Data Generation")
            from generate_data import build_and_save_all

            build_and_save_all(DATA_DIR)
            print("  ✓ Raw CSVs → data/")

        if not args.skip_train:
            banner("Step 2 / 4  —  Feature Engineering")
            from features import FEATURE_COLS, build_features

            build_features(save=True, data_dir=DATA_DIR)
            print("  ✓ merged_dataset.csv ready")

            banner("Step 3 / 4  —  Model Training (walk-forward CV + calibration)")
            import pandas as pd
            from model import train_final_model, walk_forward_cv
            from report import plot_cv_results, plot_feature_importance

            df = pd.read_csv(DATA_DIR / "merged_dataset.csv", parse_dates=["date"])
            cv_df = walk_forward_cv(df, FEATURE_COLS, n_splits=5)
            cv_df.to_csv(OUTPUTS_DIR / "cv_results.csv", index=False)
            plot_cv_results(cv_df, OUTPUTS_DIR / "cv_results.png")

            _, _, fi = train_final_model(df, FEATURE_COLS, MODELS_DIR)
            plot_feature_importance(fi, OUTPUTS_DIR / "feature_importance.png")
            print("  ✓ Models → models/")
        elif not args.skip_data:
            banner("Step 2 / 4  —  Feature Engineering (no training)")
            from features import build_features

            build_features(save=True, data_dir=DATA_DIR)

        banner("Step 4 / 4  —  WC2026 Match & Tournament Simulation")
        from simulate import main as sim_main

        sim_main()

    elapsed = time.time() - t0
    print(f"\n{'█'*62}")
    print(f"  ✓ Complete in {elapsed:.0f}s")
    print("  Key outputs:")
    print(f"    {OUTPUTS_DIR}/group_stage_predictions.csv")
    print(f"    {OUTPUTS_DIR}/group_qualification_probs.csv")
    print(f"    {OUTPUTS_DIR}/tournament_winner_probabilities.csv")
    print(f"    {OUTPUTS_DIR}/tournament_winner_odds.json")
    print(f"    {OUTPUTS_DIR}/winner_odds.png")
    print(f"    {OUTPUTS_DIR}/group_predictions.png")
    print(f"    {OUTPUTS_DIR}/feature_importance.png")
    print(f"    {OUTPUTS_DIR}/cv_results.png")
    print(f"{'█'*62}\n")


if __name__ == "__main__":
    main()
