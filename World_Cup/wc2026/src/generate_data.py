"""
src/generate_data.py
Generates realistic synthetic datasets mirroring real data structure.
Replace build_and_save_all() with real downloads when running locally.
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)

TEAMS = {
    "Argentina":             (2085, 3,  "CONMEBOL", 1.8),
    "France":                (2070, 1,  "UEFA",     1.9),
    "Spain":                 (2055, 2,  "UEFA",     1.9),
    "England":               (2020, 4,  "UEFA",     1.7),
    "Brazil":                (2000, 6,  "CONMEBOL", 1.7),
    "Portugal":              (1985, 5,  "UEFA",     1.8),
    "Netherlands":           (1975, 7,  "UEFA",     1.6),
    "Germany":               (1960, 10, "UEFA",     1.6),
    "Belgium":               (1940, 9,  "UEFA",     1.5),
    "Morocco":               (1920, 8,  "CAF",      1.3),
    "Croatia":               (1905, 11, "UEFA",     1.3),
    "Colombia":              (1890, 13, "CONMEBOL", 1.4),
    "Uruguay":               (1880, 17, "CONMEBOL", 1.3),
    "Japan":                 (1865, 18, "AFC",      1.4),
    "Mexico":                (1855, 15, "CONCACAF", 1.4),
    "United States":         (1840, 16, "CONCACAF", 1.3),
    "Senegal":               (1825, 14, "CAF",      1.2),
    "Switzerland":           (1810, 19, "UEFA",     1.3),
    "Ecuador":               (1795, 23, "CONMEBOL", 1.2),
    "South Korea":           (1780, 25, "AFC",      1.2),
    "Australia":             (1760, 27, "AFC",      1.2),
    "Turkey":                (1745, 22, "UEFA",     1.2),
    "Nigeria":               (1730, 26, "CAF",      1.2),
    "Iran":                  (1720, 21, "AFC",      1.1),
    "Canada":                (1700, 30, "CONCACAF", 1.2),
    "Egypt":                 (1690, 29, "CAF",      1.1),
    "Paraguay":              (1675, 32, "CONMEBOL", 1.1),
    "Peru":                  (1660, 33, "CONMEBOL", 1.1),
    "Ivory Coast":           (1645, 28, "CAF",      1.2),
    "Ghana":                 (1630, 35, "CAF",      1.1),
    "Saudi Arabia":          (1615, 36, "AFC",      1.1),
    "Slovakia":              (1600, 37, "UEFA",     1.1),
    "Scotland":              (1590, 38, "UEFA",     1.1),
    "Jordan":                (1570, 39, "AFC",      1.0),
    "Czechia":               (1560, 40, "UEFA",     1.1),
    "South Africa":          (1540, 41, "CAF",      1.0),
    "Venezuela":             (1515, 42, "CONMEBOL", 1.0),
    "Chile":                 (1510, 43, "CONMEBOL", 1.1),
    "Bolivia":               (1490, 44, "CONMEBOL", 0.9),
    "Indonesia":             (1470, 45, "AFC",      0.9),
    "New Zealand":           (1450, 46, "OFC",      0.9),
    "Costa Rica":            (1435, 47, "CONCACAF", 0.9),
    "Honduras":              (1420, 48, "CONCACAF", 0.9),
    "Kenya":                 (1400, 49, "CAF",      0.9),
    "Panama":                (1390, 50, "CONCACAF", 0.9),
    "Bosnia and Herzegovina":(1380, 51, "UEFA",     1.0),
    "Qatar":                 (1360, 52, "AFC",      0.8),
    "Haiti":                 (1340, 53, "CONCACAF", 0.8),
    "Cuba":                  (1310, 54, "CONCACAF", 0.8),
    "Trinidad and Tobago":   (1290, 55, "CONCACAF", 0.8),
    "Algeria":               (1530, 28, "CAF",      1.0),
}

WC2026_GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czechia"],
    "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Cuba", "Ivory Coast", "Ecuador"],
    "F": ["Portugal", "Indonesia", "Algeria", "Croatia"],
    "G": ["Belgium", "Egypt", "Venezuela", "Colombia"],
    "H": ["Spain", "Senegal", "Chile", "Japan"],
    "I": ["France", "Nigeria", "Bolivia", "Slovakia"],
    "J": ["Argentina", "Jordan", "Kenya", "New Zealand"],
    "K": ["Netherlands", "Costa Rica", "Iran", "Ghana"],
    "L": ["England", "Honduras", "Saudi Arabia", "Panama"],
}

def elo_win_prob(elo_a, elo_b, home_adv=0.0):
    return 1.0 / (1.0 + 10 ** (-(elo_a + home_adv - elo_b) / 400))

def simulate_match(elo_a, elo_b, gf_a, gf_b, home_adv=0.0):
    wp = elo_win_prob(elo_a, elo_b, home_adv)
    la = max(0.3, gf_a * (0.5 + wp))
    lb = max(0.3, gf_b * (1.5 - wp))
    return int(RNG.poisson(la)), int(RNG.poisson(lb))

def generate_historical_matches(n=12000):
    teams = list(TEAMS.keys())
    start, end = pd.Timestamp("2000-01-01"), pd.Timestamp("2025-12-31")
    dates = pd.to_datetime(RNG.integers(start.value, end.value, size=n)).sort_values()
    tournaments = (["Friendly"]*45 + ["FIFA World Cup"]*8 +
                   ["FIFA World Cup qualification"]*20 + ["UEFA Euro"]*4 +
                   ["Copa América"]*4 + ["AFC Asian Cup"]*3 +
                   ["Africa Cup of Nations"]*4 + ["CONCACAF Gold Cup"]*3 +
                   ["UEFA Nations League"]*5 + ["Confederations Cup"]*4)
    neutral_set = {"FIFA World Cup","UEFA Euro","Copa América",
                   "Confederations Cup","AFC Asian Cup",
                   "Africa Cup of Nations","CONCACAF Gold Cup"}
    rows = []
    for i, date in enumerate(dates):
        a, b = RNG.choice(len(teams), size=2, replace=False)
        ta, tb = teams[a], teams[b]
        elo_a, _, _, gf_a = TEAMS[ta]
        elo_b, _, _, gf_b = TEAMS[tb]
        yr = (date.year - 2000) / 25
        ea = elo_a - 100*(1-yr) + float(RNG.normal(0,40))
        eb = elo_b - 100*(1-yr) + float(RNG.normal(0,40))
        tourn = tournaments[i % len(tournaments)]
        neutral = tourn in neutral_set
        ga, gb = simulate_match(ea, eb, gf_a, gf_b, 0.0 if neutral else 60.0)
        rows.append(dict(date=date, home_team=ta, away_team=tb,
                         home_score=ga, away_score=gb, tournament=tourn,
                         neutral=neutral, home_elo=round(ea,1), away_elo=round(eb,1)))
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    # Add contextual
    df["venue_altitude_m"] = [int(RNG.integers(0,2300)) if r.neutral else 50 for _,r in df.iterrows()]
    df["home_travel_km"]   = [float(RNG.integers(500,12000)) if r.neutral else float(RNG.integers(0,500)) for _,r in df.iterrows()]
    df["away_travel_km"]   = [float(RNG.integers(500,12000)) for _ in range(len(df))]
    last = {}
    hr, ar = [], []
    for _,r in df.iterrows():
        hr.append((r.date - last[r.home_team]).days if r.home_team in last else 14)
        ar.append((r.date - last[r.away_team]).days if r.away_team in last else 14)
        last[r.home_team] = r.date; last[r.away_team] = r.date
    df["home_rest_days"] = hr; df["away_rest_days"] = ar
    return df

def generate_elo_series():
    rows = []
    months = pd.date_range("2000-01-01","2026-05-01",freq="MS")
    for team,(base,_,_,_) in TEAMS.items():
        for month in months:
            yr = (month.year-2000)/26
            elo = base - 100 + 100*yr + float(RNG.normal(0,8))
            rows.append(dict(date=month, team=team, elo=round(elo,1)))
    return pd.DataFrame(rows)

def generate_fifa_rankings():
    rows = []
    months = pd.date_range("2000-01-01","2026-04-01",freq="MS")
    teams_sorted = sorted(TEAMS.items(), key=lambda x: x[1][1])
    for month in months:
        for rank,(team,(_,base_rank,_,_)) in enumerate(teams_sorted,1):
            rows.append(dict(rank_date=month, country_full=team,
                             rank=max(1,base_rank+int(RNG.integers(-4,5))),
                             total_points=round(2000-base_rank*8+float(RNG.normal(0,30)),1)))
    return pd.DataFrame(rows)

def generate_squad_stats():
    rows = []
    for team,(elo,rank,conf,avg_gf) in TEAMS.items():
        s = (elo-1200)/1000
        rows.append(dict(team=team, season="2025-26", confederation=conf,
                         xg_per_game=round(avg_gf*0.95+float(RNG.normal(0,0.05)),2),
                         xga_per_game=round((2.8-avg_gf)*0.7+float(RNG.normal(0,0.05)),2),
                         possession_pct=round(48+s*12+float(RNG.normal(0,2)),1),
                         squad_avg_age=round(26.5+float(RNG.normal(0,1.2)),1),
                         squad_market_value_meur=round(500*s**2+float(RNG.uniform(5,50)),1),
                         avg_caps=round(35+s*20+float(RNG.normal(0,5)),1),
                         fifa_rank=rank, elo_rating=elo))
    return pd.DataFrame(rows)

def generate_betting_odds(matches):
    wc = matches[matches["tournament"]=="FIFA World Cup"].copy()
    rows = []
    for _,m in wc.iterrows():
        wp_h = elo_win_prob(m.home_elo, m.away_elo)
        wp_d, wp_a = 0.26, max(0.05, 1-wp_h-0.26)
        mg = 1.06
        oh = round(mg/max(0.05,wp_h),2); od = round(mg/0.26,2); oa = round(mg/wp_a,2)
        rows.append(dict(Date=m.date, HomeTeam=m.home_team, AwayTeam=m.away_team,
                         FTHG=m.home_score, FTAG=m.away_score,
                         B365H=oh, B365D=od, B365A=oa,
                         PSH=round(oh*float(RNG.uniform(0.97,1.03)),2),
                         PSD=round(od*float(RNG.uniform(0.97,1.03)),2),
                         PSA=round(oa*float(RNG.uniform(0.97,1.03)),2)))
    return pd.DataFrame(rows)

def build_and_save_all(data_dir):
    data_dir = Path(data_dir); data_dir.mkdir(exist_ok=True)
    print("  Generating match results …")
    matches = generate_historical_matches(12000)
    matches.to_csv(data_dir/"01_match_results.csv",index=False)
    print(f"    → {len(matches):,} rows")
    print("  Generating Elo ratings …")
    elo = generate_elo_series()
    elo.to_csv(data_dir/"02_elo_ratings.csv",index=False)
    print(f"    → {len(elo):,} rows")
    print("  Generating FIFA rankings …")
    fifa = generate_fifa_rankings()
    fifa.to_csv(data_dir/"03_fifa_rankings.csv",index=False)
    print(f"    → {len(fifa):,} rows")
    print("  Generating squad stats …")
    squad = generate_squad_stats()
    squad.to_csv(data_dir/"04_squad_stats.csv",index=False)
    print(f"    → {len(squad):,} rows")
    print("  Generating betting odds …")
    odds = generate_betting_odds(matches)
    odds.to_csv(data_dir/"05_betting_odds.csv",index=False)
    print(f"    → {len(odds):,} rows")
    return dict(matches=matches, elo=elo, fifa=fifa, squad=squad, odds=odds)
