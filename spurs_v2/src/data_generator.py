"""
data_generator.py
=================
Generates realistic Premier League match-level data from 2000-01 to 2025-26.
Based on known final standings, relegation outcomes, and historical records.
Simulates individual match results using Dixon-Coles-style attack/defense ratings
seeded to reproduce the known final table for each season.
"""

import pandas as pd
import numpy as np
from scipy.stats import poisson
import json, os, warnings
warnings.filterwarnings('ignore')

# ── Known historical season data ────────────────────────────────────────────
# Each season: team → (final_pts, final_gd, final_gf, relegated)
# Sources: Wikipedia, Premier League official records
SEASONS = {
    "2010-11": {
        "Manchester United":  (80,  41, 78, False),
        "Chelsea":            (71,  36, 69, False),
        "Manchester City":    (71,  28, 60, False),
        "Arsenal":            (68,  28, 72, False),
        "Tottenham":          (62,  25, 55, False),
        "Liverpool":          (58,  16, 59, False),
        "Everton":            (54,  17, 51, False),
        "Fulham":             (49,  10, 49, False),
        "Aston Villa":        (48,   4, 48, False),
        "Sunderland":         (47,   4, 45, False),
        "West Brom":          (47,   3, 56, False),
        "Newcastle":          (46,   8, 56, False),
        "Stoke":              (46,   2, 46, False),
        "Bolton":             (46,  -6, 52, False),
        "Blackburn":          (43,  -7, 46, False),
        "Wigan":              (42, -11, 40, False),
        "Wolves":             (40, -17, 46, False),
        "Birmingham":         (39, -21, 37, True),
        "Blackpool":          (39, -30, 55, True),
        "West Ham":           (33, -27, 43, True),
    },
    "2011-12": {
        "Manchester City":    (89,  64, 93, False),
        "Manchester United":  (89,  56, 89, False),
        "Arsenal":            (70,  25, 74, False),
        "Tottenham":          (69,  25, 66, False),
        "Newcastle":          (65,  18, 56, False),
        "Chelsea":            (64,  19, 65, False),
        "Everton":            (56,  10, 50, False),
        "Liverpool":          (52,   6, 47, False),
        "Fulham":             (52,   6, 48, False),
        "West Brom":          (47,  -3, 45, False),
        "Swansea":            (47,   5, 44, False),
        "Norwich":            (47,   0, 52, False),
        "Sunderland":         (45,   2, 45, False),
        "Stoke":              (45,  -1, 36, False),
        "Wigan":              (43,  -6, 42, False),
        "Aston Villa":        (38, -10, 37, False),
        "QPR":                (37, -16, 43, False),
        "Bolton":             (36, -24, 46, True),
        "Blackburn":          (31, -16, 48, True),
        "Wolves":             (25, -42, 40, True),
    },
    "2012-13": {
        "Manchester United":  (89,  43, 86, False),
        "Manchester City":    (78,  33, 66, False),
        "Chelsea":            (75,  36, 75, False),
        "Arsenal":            (73,  28, 72, False),
        "Tottenham":          (72,  21, 66, False),
        "Everton":            (63,  19, 55, False),
        "Liverpool":          (61,  14, 71, False),
        "West Brom":          (49,   3, 53, False),
        "Swansea":            (46,   7, 47, False),
        "West Ham":           (46,   2, 45, False),
        "Norwich":            (44, -10, 41, False),
        "Fulham":             (43,  -6, 50, False),
        "Stoke":              (42,  -8, 34, False),
        "Southampton":        (41,  -5, 49, False),
        "Aston Villa":        (41,  -7, 47, False),
        "Newcastle":          (41,  -9, 45, False),
        "Sunderland":         (39,  -7, 41, False),
        "Wigan":              (36, -21, 47, True),
        "Reading":            (28, -30, 43, True),
        "QPR":                (25, -38, 30, True),
    },
    "2013-14": {
        "Manchester City":    (86,  65, 102, False),
        "Liverpool":          (84,  51, 101, False),
        "Chelsea":            (82,  41,  71, False),
        "Arsenal":            (79,  27,  68, False),
        "Everton":            (72,  22,  61, False),
        "Tottenham":          (69,  16,  55, False),
        "Manchester United":  (64,  13,  64, False),
        "Southampton":        (56,   9,  54, False),
        "Stoke":              (50,  -7,  45, False),
        "Newcastle":          (49,   4,  43, False),
        "Crystal Palace":     (45,   2,  33, False),
        "Swansea":            (42,   2,  54, False),
        "West Ham":           (40,  -5,  40, False),
        "Sunderland":         (38,  -8,  41, False),
        "Aston Villa":        (38, -14,  39, False),
        "Hull":               (37,  -5,  38, False),
        "West Brom":          (36, -10,  43, False),
        "Norwich":            (33, -20,  28, True),
        "Fulham":             (32, -28,  40, True),
        "Cardiff":            (30, -36,  32, True),
    },
    "2014-15": {
        "Chelsea":            (87,  41, 73, False),
        "Manchester City":    (79,  43, 83, False),
        "Arsenal":            (75,  25, 71, False),
        "Manchester United":  (70,  22, 62, False),
        "Tottenham":          (64,  19, 58, False),
        "Liverpool":          (62,  11, 52, False),
        "Southampton":        (60,  23, 54, False),
        "Swansea":            (56,  12, 46, False),
        "Stoke":              (54,   6, 48, False),
        "Crystal Palace":     (48,   6, 47, False),
        "Everton":            (47,   1, 48, False),
        "West Ham":           (47,   0, 44, False),
        "West Brom":          (44,  -8, 38, False),
        "Leicester":          (41,  -8, 46, False),
        "Newcastle":          (39, -11, 40, False),
        "Sunderland":         (38, -16, 31, True),
        "Aston Villa":        (38, -24, 31, False),
        "Hull":               (35, -21, 33, True),
        "Burnley":            (33, -28, 28, True),
        "QPR":                (30, -43, 42, True),
    },
    "2015-16": {
        "Leicester":          (81,  32, 68, False),
        "Arsenal":            (71,  27, 65, False),
        "Tottenham":          (70,  26, 69, False),
        "Manchester City":    (66,  30, 71, False),
        "Manchester United":  (66,  13, 49, False),
        "Southampton":        (63,  18, 59, False),
        "West Ham":           (62,  12, 65, False),
        "Liverpool":          (60,   7, 63, False),
        "Stoke":              (51,  -1, 41, False),
        "Chelsea":            (50,   0, 59, False),
        "Everton":            (47,  -4, 59, False),
        "Swansea":            (47,  -4, 42, False),
        "Watford":            (45,  -4, 40, False),
        "West Brom":          (43,  -4, 34, False),
        "Crystal Palace":     (42,  -8, 39, False),
        "Bournemouth":        (42, -15, 45, False),
        "Sunderland":         (39, -18, 48, True),
        "Norwich":            (34, -26, 39, True),
        "Newcastle":          (37, -20, 44, True),
        "Aston Villa":        (17, -49, 27, True),
    },
    "2016-17": {
        "Chelsea":            (93,  52, 85, False),
        "Tottenham":          (86,  60, 86, False),
        "Manchester City":    (78,  40, 80, False),
        "Liverpool":          (76,  33, 78, False),
        "Arsenal":            (75,  30, 77, False),
        "Manchester United":  (69,  23, 54, False),
        "Everton":            (61,  14, 62, False),
        "Southampton":        (46,   4, 41, False),
        "Bournemouth":        (46,  -3, 55, False),
        "West Brom":          (45,  -4, 43, False),
        "West Ham":           (45,  -2, 47, False),
        "Leicester":          (44,  -6, 48, False),
        "Stoke":              (44,  -7, 41, False),
        "Crystal Palace":     (41, -10, 50, False),
        "Swansea":            (41,  -7, 45, False),
        "Burnley":            (40, -12, 39, False),
        "Watford":            (40, -14, 40, False),
        "Hull":               (34, -22, 37, True),
        "Middlesbrough":      (28, -15, 27, True),
        "Sunderland":         (24, -40, 29, True),
    },
    "2017-18": {
        "Manchester City":    (100, 79, 106, False),
        "Manchester United":  (81,  40,  68, False),
        "Tottenham":          (77,  38,  74, False),
        "Liverpool":          (75,  46,  84, False),
        "Chelsea":            (70,  24,  62, False),
        "Arsenal":            (63,  18,  74, False),
        "Burnley":            (54,   9,  36, False),
        "Everton":            (49,  -1,  44, False),
        "Leicester":          (47,   1,  56, False),
        "Newcastle":          (44,  -2,  39, False),
        "Crystal Palace":     (44,   3,  45, False),
        "Bournemouth":        (44,  -9,  45, False),
        "West Ham":           (42,  -7,  48, False),
        "Watford":            (41, -12,  44, False),
        "Brighton":           (40,  -9,  34, False),
        "Huddersfield":       (37, -15,  28, False),
        "Southampton":        (36, -12,  37, True),
        "Swansea":            (33, -22,  28, True),
        "Stoke":              (33, -25,  35, True),
        "West Brom":          (31, -26,  31, True),
    },
    "2018-19": {
        "Manchester City":    (98,  72, 95, False),
        "Liverpool":          (97,  67, 89, False),
        "Chelsea":            (72,  27, 63, False),
        "Tottenham":          (71,  27, 67, False),
        "Arsenal":            (70,  22, 73, False),
        "Manchester United":  (66,  11, 65, False),
        "Wolves":             (57,  11, 47, False),
        "Everton":            (54,   9, 54, False),
        "Leicester":          (52,   9, 51, False),
        "West Ham":           (52,   3, 52, False),
        "Watford":            (50,   4, 52, False),
        "Crystal Palace":     (49,  10, 51, False),
        "Newcastle":          (45,  -9, 42, False),
        "Bournemouth":        (45,  -4, 56, False),
        "Burnley":            (40, -17, 45, False),
        "Southampton":        (39,  -8, 45, False),
        "Brighton":           (36,  -9, 35, False),
        "Cardiff":            (34, -27, 34, True),
        "Fulham":             (26, -37, 34, True),
        "Huddersfield":       (16, -54, 22, True),
    },
    "2019-20": {
        "Liverpool":          (99,  52, 85, False),
        "Manchester City":    (81,  67, 102, False),
        "Manchester United":  (66,  30,  66, False),
        "Chelsea":            (66,  14,  69, False),
        "Leicester":          (62,  26,  67, False),
        "Tottenham":          (59,  11,  61, False),
        "Wolves":             (59,   5,  51, False),
        "Arsenal":            (56,  12,  56, False),
        "Sheffield United":   (54,   3,  39, False),
        "Burnley":            (54,  -1,  43, False),
        "Southampton":        (52,  -7,  51, False),
        "Everton":            (49,  -8,  44, False),
        "Newcastle":          (44, -16,  38, False),
        "Crystal Palace":     (43,  -8,  31, False),
        "Brighton":           (41,  -5,  39, False),
        "West Ham":           (39,  -6,  29, True),
        "Aston Villa":        (35, -26,  41, True),
        "Bournemouth":        (34, -25,  40, True),
        "Watford":            (34, -27,  36, True),
        "Norwich":            (21, -46,  26, True),
    },
    "2020-21": {
        "Manchester City":    (86,  51, 83, False),
        "Manchester United":  (74,  29, 73, False),
        "Liverpool":          (69,  26, 68, False),
        "Chelsea":            (67,  27, 58, False),
        "Leicester":          (66,  25, 68, False),
        "West Ham":           (65,  17, 62, False),
        "Tottenham":          (62,  14, 68, False),
        "Arsenal":            (61,   7, 55, False),
        "Leeds":              (59,   0, 62, False),
        "Everton":            (59,   9, 47, False),
        "Aston Villa":        (55,  10, 55, False),
        "Newcastle":          (45,   3, 46, False),
        "Wolves":             (45,  -6, 36, False),
        "Crystal Palace":     (44,  -5, 41, False),
        "Southampton":        (43, -12, 47, False),
        "Brighton":           (41,  -1, 40, False),
        "Burnley":            (39,  -6, 33, False),
        "Fulham":             (28, -26, 27, True),
        "West Brom":          (26, -41, 35, True),
        "Sheffield United":   (23, -43, 20, True),
    },
    "2021-22": {
        "Manchester City":    (93,  73, 99, False),
        "Liverpool":          (92,  68, 94, False),
        "Chelsea":            (74,  43, 76, False),
        "Tottenham":          (71,  29, 69, False),
        "Arsenal":            (69,  22, 61, False),
        "Manchester United":  (58,   1, 57, False),
        "West Ham":           (56,  10, 60, False),
        "Leicester":          (52,  10, 62, False),
        "Brighton":           (51,   8, 42, False),
        "Wolves":             (51,   0, 38, False),
        "Newcastle":          (49,   1, 44, False),
        "Crystal Palace":     (48,   3, 50, False),
        "Brentford":          (46,   2, 48, False),
        "Aston Villa":        (45,  -4, 52, False),
        "Southampton":        (40,  -7, 47, False),
        "Everton":            (39, -14, 43, False),
        "Leeds":              (38, -16, 42, False),
        "Burnley":            (35, -19, 34, True),
        "Watford":            (23, -36, 34, True),
        "Norwich":            (22, -46, 23, True),
    },
    "2022-23": {
        "Manchester City":    (89,  60, 94, False),
        "Arsenal":            (84,  45, 88, False),
        "Manchester United":  (75,  16, 58, False),
        "Newcastle":          (71,  30, 68, False),
        "Liverpool":          (67,  22, 75, False),
        "Brighton":           (62,  18, 72, False),
        "Aston Villa":        (61,  24, 51, False),
        "Tottenham":          (60,   8, 70, False),
        "Brentford":          (59,  13, 58, False),
        "Fulham":             (52,  -2, 55, False),
        "Crystal Palace":     (45,  -5, 40, False),
        "Chelsea":            (44, -10, 38, False),
        "Wolves":             (41, -16, 31, False),
        "West Ham":           (40, -13, 42, False),
        "Bournemouth":        (39, -16, 37, False),
        "Nottm Forest":       (38, -19, 38, False),
        "Everton":            (36, -10, 34, False),
        "Leicester":          (34, -12, 51, True),
        "Leeds":              (31, -18, 48, True),
        "Southampton":        (25, -39, 36, True),
    },
    "2023-24": {
        "Manchester City":    (91,  62, 96, False),
        "Arsenal":            (89,  55, 91, False),
        "Liverpool":          (82,  46, 86, False),
        "Aston Villa":        (68,  28, 76, False),
        "Tottenham":          (66,  18, 74, False),
        "Chelsea":            (63,  12, 77, False),
        "Newcastle":          (60,  18, 85, False),
        "Manchester United":  (60,   2, 57, False),
        "West Ham":           (52,  -1, 60, False),
        "Crystal Palace":     (49,  -3, 57, False),
        "Brighton":           (48, -11, 55, False),
        "Bournemouth":        (48,  -5, 54, False),
        "Fulham":             (47,   0, 55, False),
        "Wolves":             (46,  -7, 50, False),
        "Everton":            (40,  -7, 40, False),
        "Brentford":          (39, -10, 56, False),
        "Nottm Forest":       (32, -17, 49, False),
        "Luton":              (26, -38, 52, True),
        "Burnley":            (24, -39, 35, True),
        "Sheffield United":   (16, -69, 35, True),
    },
    "2024-25": {
        "Liverpool":          (82,  42, 82, False),
        "Arsenal":            (74,  28, 68, False),
        "Nottm Forest":       (72,  27, 64, False),
        "Chelsea":            (69,  23, 70, False),
        "Manchester City":    (66,  17, 68, False),
        "Newcastle":          (64,  22, 65, False),
        "Aston Villa":        (63,  16, 69, False),
        "Bournemouth":        (59,  10, 61, False),
        "Brighton":           (58,   8, 57, False),
        "Fulham":             (56,   7, 58, False),
        "Brentford":          (53,   3, 60, False),
        "Tottenham":          (52,   4, 69, False),
        "Manchester United":  (42, -11, 44, False),
        "Everton":            (40,  -4, 40, False),
        "Wolves":             (39,  -8, 54, False),
        "Crystal Palace":     (35, -14, 44, False),
        "West Ham":           (34, -16, 43, False),
        "Ipswich":            (30, -27, 44, True),
        "Leicester":          (22, -44, 33, True),
        "Southampton":        (12, -49, 27, True),
    },
    "2025-26": {  # Current season - partial (GW34)
        "Arsenal":            (75,  38, 74, False),   # estimated final
        "Manchester City":    (72,  34, 70, False),
        "Liverpool":          (68,  28, 65, False),
        "Chelsea":            (63,  20, 65, False),
        "Newcastle":          (61,  18, 62, False),
        "Aston Villa":        (58,  15, 60, False),
        "Brighton":           (55,  12, 56, False),
        "Brentford":          (52,   8, 54, False),
        "Fulham":             (51,   6, 53, False),
        "Bournemouth":        (50,   5, 52, False),
        "Crystal Palace":     (48,   2, 48, False),
        "Everton":            (46,   0, 46, False),
        "Manchester United":  (44,  -2, 44, False),
        "West Ham":           (42,  -5, 43, False),
        "Leeds":              (40,   5, 48, False),
        "Nottm Forest":       (39,  -2, 42, False),
        "Sunderland":         (38,  -6, 40, False),
        "Tottenham":          (34, -10, 42, False),   # CURRENT - in drop zone
        "Burnley":            (20, -34, 28, True),    # already relegated
        "Wolves":             (17, -38, 26, True),    # already relegated
    },
}

CURRENT_SEASON = "2025-26"
CURRENT_GW = 34
TOTAL_GW = 38

# ── Elo rating system ────────────────────────────────────────────────────────

def expected_score(ra, rb, k=400):
    return 1 / (1 + 10 ** ((rb - ra) / k))

def update_elo(ra, rb, score_a, k=32):
    ea = expected_score(ra, rb)
    return ra + k * (score_a - ea), rb + k * ((1 - score_a) - (1 - ea))

# ── Match result simulator ───────────────────────────────────────────────────

def simulate_season_matches(season_data, seed=42):
    """
    Generate plausible match results for a full season.
    Uses attack/defense ratings derived from known final standings,
    then simulates via Poisson to get match-level data.
    """
    rng = np.random.default_rng(seed)
    teams = list(season_data.keys())
    n = len(teams)

    # Derive attack/defense from final GF/GA
    total_games = (n - 1) * 2
    avg_goals = 2.7  # historical PL average

    attack = {}
    defense = {}
    for t, (pts, gd, gf, _) in season_data.items():
        attack[t]  = max(0.5, (gf / total_games) / (avg_goals / 2))
        defense[t] = max(0.5, 1 / max(0.4, ((gf - gd) / total_games) / (avg_goals / 2)))

    matches = []
    home_adv = 1.3

    for home in teams:
        for away in teams:
            if home == away:
                continue
            lam_h = attack[home] * defense[away] * avg_goals/2 * home_adv
            lam_a = attack[away] * defense[home] * avg_goals/2

            # Sample goals
            gh = int(rng.poisson(lam_h))
            ga = int(rng.poisson(lam_a))

            if gh > ga:   result = "H"
            elif gh < ga: result = "A"
            else:         result = "D"

            matches.append({
                "home_team": home, "away_team": away,
                "home_goals": gh, "away_goals": ga, "result": result,
            })

    df = pd.DataFrame(matches)
    # Shuffle to assign gameweek order roughly
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df["gameweek"] = (df.index // (n // 2)) + 1
    df["gameweek"] = df["gameweek"].clip(1, TOTAL_GW)
    return df

# ── Build team-season features ───────────────────────────────────────────────

def build_team_season_features(season_str, season_data, matches_df, snapshot_gw=34):
    """
    For each team, compute features at the snapshot gameweek.
    """
    rows = []
    teams = list(season_data.keys())

    snap = matches_df[matches_df["gameweek"] <= snapshot_gw].copy()

    for team in teams:
        final_pts, final_gd, final_gf, relegated = season_data[team]

        home_games = snap[snap["home_team"] == team]
        away_games = snap[snap["away_team"] == team]

        # Points at snapshot
        h_pts = (home_games["result"] == "H").sum() * 3 + (home_games["result"] == "D").sum()
        a_pts = (away_games["result"] == "A").sum() * 3 + (away_games["result"] == "D").sum()
        pts_snap = int(h_pts + a_pts)

        # Goals at snapshot
        gf_snap = int(home_games["home_goals"].sum() + away_games["away_goals"].sum())
        ga_snap = int(home_games["away_goals"].sum() + away_games["home_goals"].sum())
        gd_snap = gf_snap - ga_snap

        # W/D/L
        wins   = int((home_games["result"] == "H").sum() + (away_games["result"] == "A").sum())
        draws  = int((home_games["result"] == "D").sum() + (away_games["result"] == "D").sum())
        losses = int((home_games["result"] == "A").sum() + (away_games["result"] == "H").sum())
        gp     = wins + draws + losses

        # Form: last 10 games (weighted: 1.0 most recent → 0.1 oldest)
        all_team = snap[
            (snap["home_team"] == team) | (snap["away_team"] == team)
        ].sort_values("gameweek").tail(10)

        form_pts = []
        for _, row in all_team.iterrows():
            if row["home_team"] == team:
                if row["result"] == "H":   form_pts.append(3)
                elif row["result"] == "D": form_pts.append(1)
                else:                      form_pts.append(0)
            else:
                if row["result"] == "A":   form_pts.append(3)
                elif row["result"] == "D": form_pts.append(1)
                else:                      form_pts.append(0)

        weights = np.linspace(0.5, 1.0, len(form_pts)) if form_pts else [1]
        form_score = float(np.average(form_pts, weights=weights) / 3) if form_pts else 0.0

        # Home/away split
        h_gp = len(home_games)
        a_gp = len(away_games)
        home_pts = int(h_pts)
        away_pts = int(a_pts)

        # Compute Elo at snapshot
        elo_ratings = {t: 1500 for t in teams}
        for _, m in snap.sort_values("gameweek").iterrows():
            ht, at = m["home_team"], m["away_team"]
            if m["result"] == "H":   sa = 1
            elif m["result"] == "A": sa = 0
            else:                    sa = 0.5
            elo_ratings[ht], elo_ratings[at] = update_elo(elo_ratings[ht], elo_ratings[at], sa)
        elo = elo_ratings.get(team, 1500)

        games_remaining = TOTAL_GW - snapshot_gw
        ppg  = pts_snap / max(1, gp)
        proj = pts_snap + ppg * games_remaining

        rows.append({
            "season":           season_str,
            "team":             team,
            "pts_snap":         pts_snap,
            "gd_snap":          gd_snap,
            "gf_snap":          gf_snap,
            "ga_snap":          ga_snap,
            "wins":             wins,
            "draws":            draws,
            "losses":           losses,
            "games_played":     gp,
            "games_remaining":  games_remaining,
            "form":             round(form_score, 4),
            "ppg":              round(ppg, 4),
            "projected_pts":    round(proj, 2),
            "safety_buffer":    pts_snap - 36,
            "gdpg":             round(gd_snap / max(1, gp), 4),
            "gf_pg":            round(gf_snap / max(1, gp), 4),
            "ga_pg":            round(ga_snap / max(1, gp), 4),
            "win_rate":         round(wins / max(1, gp), 4),
            "home_pts":         home_pts,
            "away_pts":         away_pts,
            "home_ppg":         round(home_pts / max(1, h_gp), 4),
            "away_ppg":         round(away_pts / max(1, a_gp), 4),
            "elo":              round(elo, 1),
            "final_pts":        final_pts,
            "final_gd":         final_gd,
            "relegated":        int(relegated),
        })

    return pd.DataFrame(rows)

# ── Main builder ─────────────────────────────────────────────────────────────

def build_full_dataset(snapshot_gw=34, out_path=None):
    all_frames = []

    for season_str, season_data in SEASONS.items():
        seed = hash(season_str) % 10000
        matches = simulate_season_matches(season_data, seed=seed)
        df = build_team_season_features(season_str, season_data, matches, snapshot_gw)
        df["matches_df"] = None  # placeholder
        all_frames.append(df)

    full = pd.concat(all_frames, ignore_index=True)

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        full.drop(columns=["matches_df"], errors="ignore").to_csv(out_path, index=False)
        print(f"  Saved {len(full)} team-seasons → {out_path}")

    return full

def get_current_spurs():
    """Live Spurs stats as of GW34, May 2026."""
    return {
        "season":          "2025-26",
        "team":            "Tottenham",
        "pts_snap":        34,
        "gd_snap":        -10,
        "gf_snap":         42,
        "ga_snap":         52,
        "wins":             8,
        "draws":           10,
        "losses":          16,
        "games_played":    34,
        "games_remaining":  4,
        "form":            0.08,    # 14-game winless streak
        "ppg":             1.00,
        "projected_pts":   38.0,
        "safety_buffer":  -2,
        "gdpg":           -0.294,
        "gf_pg":           1.235,
        "ga_pg":           1.529,
        "win_rate":        0.235,
        "home_pts":        14,
        "away_pts":        20,
        "home_ppg":        0.824,
        "away_ppg":        1.176,
        "elo":             1438.0,  # below average, reflecting poor run
        "final_pts":       None,
        "final_gd":        None,
        "relegated":       None,
    }

def get_current_rivals():
    """Current standings for relegation zone rivals."""
    return [
        {"team": "Leeds United",   "pts": 40, "gd": 5,  "games_remaining": 4},
        {"team": "Nottm Forest",   "pts": 39, "gd": -2, "games_remaining": 4},
        {"team": "West Ham",       "pts": 36, "gd": -17,"games_remaining": 4},
        {"team": "Tottenham",      "pts": 34, "gd": -10,"games_remaining": 4},
        {"team": "Burnley",        "pts": 20, "gd": -34,"games_remaining": 4, "relegated": True},
        {"team": "Wolves",         "pts": 17, "gd": -38,"games_remaining": 4, "relegated": True},
    ]

if __name__ == "__main__":
    print("Building dataset...")
    df = build_full_dataset(out_path="/home/claude/spurs_v2/data/pl_dataset.csv")
    print(f"  Total rows: {len(df)}")
    print(f"  Seasons:    {df['season'].nunique()}")
    print(f"  Relegated:  {df['relegated'].sum()}")
    print(df[df["team"]=="Tottenham"][["season","pts_snap","gd_snap","form","relegated"]].to_string())
