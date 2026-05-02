# Spurs Relegation Prediction System — v2.0

A production-grade ML system for predicting Premier League relegation probability.
Currently applied to Tottenham Hotspur, 2025-26 season, GW34.

## Quick start

```bash
pip install -r requirements.txt
python run.py
```

Results are written to `outputs/results.json`.

---

## Architecture

```
spurs_v2/
├── run.py                      ← Master script — run this
├── requirements.txt
├── src/
│   ├── data_generator.py       ← Historical PL dataset (16 seasons, 320 team-seasons)
│   ├── dixon_coles.py          ← Dixon-Coles match model + season simulator
│   ├── ml_pipeline.py          ← ML ensemble, walk-forward validation, SHAP
│   └── monte_carlo.py          ← Poisson Monte Carlo simulation of remaining fixtures
├── data/
│   └── pl_dataset.csv          ← Generated on first run
└── outputs/
    └── results.json            ← Full model outputs
```

---

## Components

### 1. Data (`data_generator.py`)
- 16 Premier League seasons (2010-11 → 2025-26)
- 320 team-season observations, 52 relegated cases
- Match-level simulation using known final standings to reconstruct plausible results
- Features engineered at GW34 snapshot: points, GD, form, PPG, Elo, home/away split

### 2. ML Pipeline (`ml_pipeline.py`)

**Models:**
- Logistic Regression (C=0.3, L2 regularized)
- Random Forest (300 trees, max depth 5)
- Gradient Boosting (150 estimators, lr=0.04)

**Calibration:** Isotonic regression (5-fold) on each model to ensure probabilities are accurate, not just rankings.

**Validation:** Walk-forward cross-validation — trains on seasons 1..N, tests on season N+1. No future data ever enters training. Achieves AUC=0.952, Brier=0.066.

**SHAP:** KernelSHAP on the RF model explains each prediction in terms of feature contributions.

**Features engineered:**
| Feature | Description |
|---|---|
| pts_snap | Points at GW34 |
| gd_snap | Goal difference at GW34 |
| form | Weighted form (last 10 games, recency-weighted) |
| ppg | Points per game |
| projected_pts | Extrapolated final points |
| safety_buffer | Pts above/below the 36-pt safety line |
| gdpg | Goal difference per game |
| gf_pg / ga_pg | Goals scored/conceded per game |
| win_rate | Win percentage |
| home_ppg / away_ppg | Home and away points per game |
| elo | Elo rating at snapshot |
| pts_needed_for_safety | Points still needed to reach 36 |
| max_pts_possible | Best possible final tally |
| survival_feasibility | max_pts / safety_line (capped) |
| defensive_fragility | ga_pg / gf_pg ratio |
| attacking_output | gf_pg × win_rate composite |
| home_away_balance | home_ppg − away_ppg |

### 3. Dixon-Coles Model (`dixon_coles.py`)
Estimates each team's attack and defense parameters using maximum likelihood estimation on match results. Includes the tau correction for low-scoring matches (0-0, 1-0, 0-1, 1-1) which are over/underrepresented relative to independent Poisson.

### 4. Monte Carlo Simulation (`monte_carlo.py`)
- 100,000 full season completions
- Each match simulated independently via Poisson with team-specific lambdas
- Accounts for home advantage (×1.30 on home attack)
- Final table determined by pts → GD → GF tiebreakers
- Output: P(relegated) and final points distribution for each bottom-zone team

---

## Current results (GW34, May 2026)

| Model | Probability |
|---|---|
| Logistic Regression | 43.8% |
| Random Forest | 46.0% |
| Gradient Boosting | 34.1% |
| **ML Ensemble** | **41.3%** |
| **Monte Carlo** | **48.3%** |
| **Combined estimate** | **45.5%** |

**Verdict: HIGH RISK — survival possible but form must improve**

### Remaining fixtures
| Venue | Opponent | Win | Draw | Loss |
|---|---|---|---|---|
| Home | Wolves | 64.6% | 20.0% | 15.4% |
| Away | Aston Villa | 11.5% | 16.0% | 72.5% |
| Home | Leeds | 39.7% | 24.4% | 35.9% |
| Home | Everton | 35.8% | 24.4% | 39.8% |

### Scenario analysis
| Scenario | Final pts | Rel. prob |
|---|---|---|
| Win all 4 | 46 | 61.8% |
| Win 3, draw 1 | 44 | 59.7% |
| Win 2, draw 2 | 42 | 52.8% |
| Win 2, draw 1, lose 1 | 41 | 45.0% |
| Win 2, lose 2 | 40 | 44.5% |
| Draw 2, lose 2 | 36 | 40.5% |
| Lose all 4 | 34 | 40.5% |

---

## Extending this project

### Add real data
Replace the synthetic dataset with actual match-by-match data:
```python
# football-data.co.uk CSVs (free, 1993-present)
import pandas as pd
df = pd.read_csv("https://www.football-data.co.uk/mmz4281/2324/E0.csv")
```

### Add xG features
From understat.com or StatsBomb open data:
```python
features["xg_diff_pg"] = (xg_for - xg_against) / games_played
features["xg_overperformance"] = actual_goals - expected_goals
```

### Gameweek-by-gameweek updating
```python
# After each GW, update current_standings and re-run
for gw in range(35, 39):
    results = run_simulation(n_sims=100_000)
    save_results(gw, results)
    plot_probability_trajectory()
```

### Improve the Monte Carlo
Replace fixed team strengths with fitted Dixon-Coles parameters:
```python
dc = DixonColesModel()
dc.fit(season_matches_so_far)
simulator = SeasonSimulator(dc)
results = simulator.simulate_remaining(standings, remaining_fixtures, n_sims=50000)
```
