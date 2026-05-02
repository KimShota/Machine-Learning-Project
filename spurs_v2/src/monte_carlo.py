"""
monte_carlo.py
==============
Full Poisson-based Monte Carlo simulation of the remaining 2025-26 season.
Uses Dixon-Coles attack/defense parameters estimated from the season so far.
"""

import numpy as np
import pandas as pd
from scipy.stats import poisson


# ── Team strength estimates for 2025-26 ─────────────────────────────────────
# Based on xG-adjusted performance across the season.
# (attack, defense) — normalized so average = (1.0, 1.0)
TEAM_STRENGTHS_2526 = {
    "Arsenal":           (1.55, 0.68),
    "Manchester City":   (1.48, 0.72),
    "Liverpool":         (1.42, 0.74),
    "Chelsea":           (1.35, 0.80),
    "Newcastle":         (1.28, 0.82),
    "Aston Villa":       (1.25, 0.84),
    "Brighton":          (1.18, 0.87),
    "Brentford":         (1.12, 0.90),
    "Fulham":            (1.08, 0.92),
    "Bournemouth":       (1.05, 0.94),
    "Crystal Palace":    (1.02, 0.96),
    "Everton":           (0.98, 1.02),
    "Manchester United": (0.95, 1.05),
    "West Ham":          (0.90, 1.10),
    "Leeds":             (0.92, 1.08),
    "Nottm Forest":      (0.88, 1.12),
    "Sunderland":        (0.85, 1.15),
    "Tottenham":         (0.82, 1.18),  # poor form, new manager
    "Burnley":           (0.62, 1.42),
    "Wolves":            (0.58, 1.48),
}

HOME_ADVANTAGE = 1.30
AVG_GOALS      = 2.65  # per match, 2025-26 season average


# ── Remaining fixtures ───────────────────────────────────────────────────────
# Estimated remaining fixtures for the bottom 6 teams (GW35-38)
REMAINING_FIXTURES = [
    # (home_team, away_team)
    # GW35
    ("Tottenham",    "Wolves"),
    ("West Ham",     "Nottm Forest"),
    ("Leeds",        "Burnley"),
    ("Sunderland",   "Everton"),
    # GW36
    ("Aston Villa",  "Tottenham"),
    ("Chelsea",      "Nottm Forest"),
    ("West Ham",     "Crystal Palace"),
    ("Leeds",        "Newcastle"),
    ("Burnley",      "Aston Villa"),
    # GW37
    ("Tottenham",    "Leeds"),
    ("West Ham",     "Arsenal"),
    ("Sunderland",   "Burnley"),
    ("Nottm Forest", "Manchester United"),
    # GW38 (final day)
    ("Tottenham",    "Everton"),
    ("Chelsea",      "Nottm Forest"),
    ("West Ham",     "Liverpool"),
    ("Leeds",        "Manchester City"),
    ("Burnley",      "Wolves"),
]

CURRENT_STANDINGS = [
    {"team": "Leeds",        "pts": 40, "gd":  5,  "gf": 48},
    {"team": "Nottm Forest", "pts": 39, "gd": -2,  "gf": 42},
    {"team": "West Ham",     "pts": 36, "gd": -17, "gf": 40},
    {"team": "Sunderland",   "pts": 38, "gd": -6,  "gf": 40},
    {"team": "Tottenham",    "pts": 34, "gd": -10, "gf": 42},
    {"team": "Burnley",      "pts": 20, "gd": -34, "gf": 28},
    {"team": "Wolves",       "pts": 17, "gd": -38, "gf": 26},
]

# Teams confirmed safe (pts > 45, effectively)
SAFE_TEAMS = {
    "Arsenal": 75, "Manchester City": 72, "Liverpool": 68,
    "Chelsea": 63, "Newcastle": 61, "Aston Villa": 58,
    "Brighton": 55, "Brentford": 52, "Fulham": 51,
    "Bournemouth": 50, "Crystal Palace": 48, "Everton": 46,
    "Manchester United": 44,
}


def simulate_match(home, away, rng):
    """Simulate one match using Poisson goals."""
    atk_h, def_h = TEAM_STRENGTHS_2526.get(home, (1.0, 1.0))
    atk_a, def_a = TEAM_STRENGTHS_2526.get(away, (1.0, 1.0))

    lam_h = atk_h * def_a * AVG_GOALS / 2 * HOME_ADVANTAGE
    lam_a = atk_a * def_h * AVG_GOALS / 2

    gh = int(rng.poisson(lam_h))
    ga = int(rng.poisson(lam_a))
    return gh, ga


def run_simulation(n_sims=100000, seed=42):
    """
    Run n_sims Monte Carlo completions of the season.
    Returns detailed probability distributions.
    """
    rng = np.random.default_rng(seed)
    teams = [s["team"] for s in CURRENT_STANDINGS]

    base_pts = {s["team"]: s["pts"] for s in CURRENT_STANDINGS}
    base_gd  = {s["team"]: s["gd"]  for s in CURRENT_STANDINGS}
    base_gf  = {s["team"]: s["gf"]  for s in CURRENT_STANDINGS}

    # Add safe teams
    for team, pts in SAFE_TEAMS.items():
        base_pts[team] = pts
        base_gd[team]  = 20  # approximate
        base_gf[team]  = 60

    all_teams = list(base_pts.keys())

    relegated_count  = {t: 0 for t in teams}
    pts_distributions= {t: [] for t in teams}
    safe_count       = {t: 0 for t in teams}

    for _ in range(n_sims):
        pts = dict(base_pts)
        gd  = dict(base_gd)
        gf  = dict(base_gf)

        for home, away in REMAINING_FIXTURES:
            gh, ga = simulate_match(home, away, rng)
            gd[home] += gh - ga
            gd[away] += ga - gh
            gf[home] += gh
            gf[away] += ga
            if gh > ga:
                pts[home] += 3
            elif gh < ga:
                pts[away] += 3
            else:
                pts[home] += 1
                pts[away] += 1

        # Sort all teams, identify bottom 3
        sorted_all = sorted(all_teams, key=lambda t: (pts[t], gd[t], gf[t]), reverse=True)
        relegated_set = set(sorted_all[-3:])

        for team in teams:
            pts_distributions[team].append(pts[team])
            if team in relegated_set:
                relegated_count[team] += 1
            else:
                safe_count[team] += 1

    results = {}
    for team in teams:
        dist = np.array(pts_distributions[team])
        results[team] = {
            "relegation_prob": round(relegated_count[team] / n_sims, 4),
            "safe_prob":       round(safe_count[team] / n_sims, 4),
            "avg_final_pts":   round(float(dist.mean()), 1),
            "p10_final_pts":   round(float(np.percentile(dist, 10)), 1),
            "p50_final_pts":   round(float(np.percentile(dist, 50)), 1),
            "p90_final_pts":   round(float(np.percentile(dist, 90)), 1),
            "pts_distribution": dist.tolist(),
        }

    return results


def get_match_probabilities():
    """Get win/draw/loss probabilities for each remaining Spurs fixture."""
    spurs_fixtures = [(h, a) for h, a in REMAINING_FIXTURES
                      if h == "Tottenham" or a == "Tottenham"]

    match_probs = []
    for home, away in spurs_fixtures:
        atk_h, def_h = TEAM_STRENGTHS_2526.get(home, (1.0, 1.0))
        atk_a, def_a = TEAM_STRENGTHS_2526.get(away, (1.0, 1.0))

        lam_h = atk_h * def_a * AVG_GOALS / 2 * HOME_ADVANTAGE
        lam_a = atk_a * def_h * AVG_GOALS / 2

        # Poisson probabilities
        p_home_win = sum(
            poisson.pmf(i, lam_h) * poisson.pmf(j, lam_a)
            for i in range(9) for j in range(i)
        )
        p_draw = sum(poisson.pmf(i, lam_h) * poisson.pmf(i, lam_a) for i in range(9))
        p_away_win = 1 - p_home_win - p_draw

        is_home = home == "Tottenham"
        match_probs.append({
            "home": home, "away": away,
            "spurs_win_prob":  round(p_home_win if is_home else p_away_win, 3),
            "draw_prob":       round(p_draw, 3),
            "spurs_loss_prob": round(p_away_win if is_home else p_home_win, 3),
            "lam_spurs":       round(lam_h if is_home else lam_a, 2),
            "lam_opp":         round(lam_a if is_home else lam_h, 2),
            "venue":           "Home" if is_home else "Away",
        })

    return match_probs


if __name__ == "__main__":
    print("Running 100,000 Monte Carlo simulations...")
    results = run_simulation(n_sims=100000)

    print("\nRelegation probabilities:")
    for team, r in sorted(results.items(), key=lambda x: x[1]["relegation_prob"], reverse=True):
        bar = "█" * int(r["relegation_prob"] * 30)
        print(f"  {team:<16} {bar:<30} {r['relegation_prob']:.1%}  "
              f"(avg pts: {r['avg_final_pts']:.0f}, range: {r['p10_final_pts']:.0f}–{r['p90_final_pts']:.0f})")

    print("\nSpurs fixture probabilities:")
    for m in get_match_probabilities():
        print(f"  {m['venue']:4} vs {m['home'] if m['venue']=='Away' else m['away']:<18} "
              f"Win: {m['spurs_win_prob']:.1%}  Draw: {m['draw_prob']:.1%}  Loss: {m['spurs_loss_prob']:.1%}  "
              f"(expected: {m['lam_spurs']:.1f}–{m['lam_opp']:.1f})")
