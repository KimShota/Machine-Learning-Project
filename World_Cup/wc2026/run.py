"""
WC2026 ML Prediction — Main entry point
Run this file to execute the full pipeline end-to-end:
  1. Generate / load data
  2. Engineer features
  3. Train model + cross-validate
  4. Simulate tournament + generate predictions

Usage:
  python run.py                     # full pipeline
  python run.py --skip-data         # skip data generation (use existing CSVs)
  python run.py --simulate-only     # only re-run tournament simulation
"""
import sys, os, time, argparse
from pathlib import Path

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE / "src"))

def banner(msg):
    print(f"\n{'='*62}\n  {msg}\n{'='*62}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-data",      action="store_true", help="Skip data generation")
    parser.add_argument("--skip-train",     action="store_true", help="Skip model training")
    parser.add_argument("--simulate-only",  action="store_true", help="Only run simulation")
    args = parser.parse_args()

    t0 = time.time()
    print("\n" + "█"*62)
    print("  ⚽  WC2026 ML Prediction Pipeline")
    print("  From raw data → trained model → 104-match predictions")
    print("█"*62)

    # ── Step 1: Data ──────────────────────────────────────────────
    if not args.skip_data and not args.simulate_only:
        banner("Step 1 / 3  —  Data Generation")
        import generate_data   # src/generate_data.py
        # NOTE: On your machine, replace this with collect_wc2026_data.py
        # which downloads real data from Kaggle, eloratings.net, etc.
        print("  ✓ Data ready in data/")

    # ── Step 2: Features + Training ───────────────────────────────
    if not args.skip_train and not args.simulate_only:
        banner("Step 2 / 3  —  Feature Engineering")
        from features import build_features
        build_features(save=True)
        print("  ✓ Features engineered → data/merged_dataset.csv")

        banner("Step 3 / 3  —  Model Training (Walk-Forward CV)")
        from train import main as train_main
        train_main()
        print("  ✓ Models saved → models/")

    # ── Step 3: Simulate ──────────────────────────────────────────
    banner("Generating WC2026 Predictions")
    from simulate import main as sim_main
    sim_main()

    elapsed = time.time() - t0
    print(f"\n{'█'*62}")
    print(f"  ✓ Full pipeline complete in {elapsed:.0f}s")
    print(f"  Key outputs:")
    print(f"    outputs/group_stage_predictions.csv  — all 72 group matches")
    print(f"    outputs/tournament_winner_odds.json  — winner probabilities")
    print(f"    outputs/winner_odds.png              — top 20 chart")
    print(f"    outputs/group_predictions.png        — group heatmaps")
    print(f"    outputs/feature_importance.png       — model explainability")
    print(f"    outputs/cv_results.png               — model performance")
    print(f"{'█'*62}\n")


if __name__ == "__main__":
    main()
