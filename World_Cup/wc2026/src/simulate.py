"""src/simulate.py — WC2026 group stage + knockout bracket Monte Carlo simulation."""
import numpy as np
import pandas as pd
from itertools import combinations

RNG = np.random.default_rng(42)

WC2026_GROUPS = {
    "A": ["Mexico","South Africa","South Korea","Czechia"],
    "B": ["Canada","Bosnia and Herzegovina","Qatar","Switzerland"],
    "C": ["Brazil","Morocco","Haiti","Scotland"],
    "D": ["United States","Paraguay","Australia","Turkey"],
    "E": ["Germany","Cuba","Ivory Coast","Ecuador"],
    "F": ["Portugal","Indonesia","Algeria","Croatia"],
    "G": ["Belgium","Egypt","Venezuela","Colombia"],
    "H": ["Spain","Senegal","Chile","Japan"],
    "I": ["France","Nigeria","Bolivia","Slovakia"],
    "J": ["Argentina","Jordan","Kenya","New Zealand"],
    "K": ["Netherlands","Costa Rica","Iran","Ghana"],
    "L": ["England","Honduras","Saudi Arabia","Panama"],
}

# WC2026 Round of 32 bracket (simplified — winner/runner-up pairings)
# Format: (GroupX_1st, GroupY_2nd) — actual bracket depends on 3rd-place rankings
R32_BRACKET = [
    ("A1","B2"),("C1","D2"),("E1","F2"),("G1","H2"),
    ("I1","J2"),("K1","L2"),("B1","A2"),("D1","C2"),
    ("F1","E2"),("H1","G2"),("J1","I2"),("L1","K2"),
    # Best 8 third-place teams fill remaining 4 slots (simplified)
    ("A3","C3"),("B3","D3"),("E3","G3"),("F3","H3"),
]


def sample_match(p_home, p_draw, p_away):
    """Sample a single match outcome from probabilities."""
    r = RNG.random()
    if r < p_home: return "H"
    elif r < p_home + p_draw: return "D"
    else: return "A"


def sample_goals(p_home, p_draw, p_away, lam_home=1.4, lam_away=1.1):
    """Sample scoreline consistent with the outcome probabilities."""
    outcome = sample_match(p_home, p_draw, p_away)
    for _ in range(100):  # rejection sampling
        gh = int(RNG.poisson(lam_home * (1 + 0.3*(p_home-0.4))))
        ga = int(RNG.poisson(lam_away * (1 + 0.3*(p_away-0.3))))
        if outcome == "H" and gh > ga: return gh, ga
        if outcome == "D" and gh == ga: return gh, ga
        if outcome == "A" and ga > gh: return gh, ga
    # Fallback
    if outcome == "H": return 1, 0
    if outcome == "D": return 1, 1
    return 0, 1


def get_match_probs(team_a, team_b, probs_lookup):
    """
    Look up pre-computed match probabilities.
    probs_lookup: dict[(teamA, teamB)] -> (p_h, p_d, p_a)
    """
    key = (team_a, team_b)
    if key in probs_lookup:
        return probs_lookup[key]
    # Reverse
    rkey = (team_b, team_a)
    if rkey in probs_lookup:
        ph, pd_, pa = probs_lookup[rkey]
        return pa, pd_, ph
    return 0.38, 0.26, 0.36  # neutral default


def simulate_group(group_teams, probs_lookup):
    """Simulate one group stage. Returns sorted standings DataFrame."""
    teams = {t: dict(pts=0, gd=0, gf=0) for t in group_teams}
    for t1, t2 in combinations(group_teams, 2):
        ph, pd_, pa = get_match_probs(t1, t2, probs_lookup)
        gh, ga = sample_goals(ph, pd_, pa)
        if gh > ga:
            teams[t1]["pts"] += 3
        elif gh == ga:
            teams[t1]["pts"] += 1; teams[t2]["pts"] += 1
        else:
            teams[t2]["pts"] += 3
        teams[t1]["gf"] += gh; teams[t1]["gd"] += gh-ga
        teams[t2]["gf"] += ga; teams[t2]["gd"] += ga-gh

    standings = pd.DataFrame(teams).T.reset_index().rename(columns={"index":"team"})
    standings = standings.sort_values(["pts","gd","gf"], ascending=False).reset_index(drop=True)
    standings["pos"] = standings.index + 1
    return standings


def simulate_knockout_match(team_a, team_b, probs_lookup):
    """Single elimination — no draws allowed (ET/PKs modelled as 50/50 after draw."""
    ph, pd_, pa = get_match_probs(team_a, team_b, probs_lookup)
    outcome = sample_match(ph, pd_, pa)
    if outcome == "H": return team_a
    if outcome == "A": return team_b
    # Draw → 50/50 ET/penalties
    return team_a if RNG.random() < ph/(ph+pa) else team_b


def simulate_tournament(probs_lookup):
    """
    Full tournament simulation: group stage → R32 → R16 → QF → SF → Final.
    Returns the champion and path.
    """
    # Group stage
    group_results = {}
    all_third = []
    for grp, teams in WC2026_GROUPS.items():
        standings = simulate_group(teams, probs_lookup)
        group_results[grp] = standings
        all_third.append(standings.iloc[2].to_dict() | {"group": grp})

    # Best 8 third-place teams
    third_df = pd.DataFrame(all_third).sort_values(["pts","gd","gf"], ascending=False)
    best_third = third_df.head(8)["team"].tolist()

    # Build Round of 32 bracket
    def get_team(slot):
        grp, pos = slot[:-1], slot[-1]
        if pos in ("1","2"):
            idx = int(pos)-1
            return group_results[grp].iloc[idx]["team"]
        else:  # 3rd place
            return best_third.pop(0) if best_third else "TBD"

    r32_teams = []
    for a, b in R32_BRACKET:
        ta = get_team(a); tb = get_team(b)
        r32_teams.append((ta, tb))

    def play_round(matchups):
        winners = []
        for ta, tb in matchups:
            winners.append(simulate_knockout_match(ta, tb, probs_lookup))
        return [(winners[i], winners[i+1]) for i in range(0, len(winners), 2)]

    r16 = play_round(r32_teams)
    qf  = play_round(r16)
    sf  = play_round(qf)
    # Final
    finalist_a = simulate_knockout_match(*sf[0], probs_lookup)
    finalist_b = simulate_knockout_match(*sf[1], probs_lookup)
    champion   = simulate_knockout_match(finalist_a, finalist_b, probs_lookup)
    return champion


def monte_carlo(probs_lookup, n_simulations=50000):
    """
    Run n_simulations full tournament simulations.
    Returns championship probability per team.
    """
    from collections import Counter
    print(f"  Running {n_simulations:,} Monte Carlo simulations …")
    champions = Counter()
    for i in range(n_simulations):
        champion = simulate_tournament(probs_lookup)
        champions[champion] += 1
        if (i+1) % 10000 == 0:
            print(f"    {i+1:,}/{n_simulations:,} done")

    all_teams = [t for grp in WC2026_GROUPS.values() for t in grp]
    result = pd.DataFrame([
        {"team": t, "win_probability": round(champions[t]/n_simulations, 4),
         "simulated_wins": champions[t]}
        for t in all_teams
    ]).sort_values("win_probability", ascending=False).reset_index(drop=True)
    return result


def build_probs_lookup(feature_df, ensemble, calibrators, squad_stats):
    """
    Build a (teamA, teamB) -> (p_home, p_draw, p_away) lookup
    for all WC2026 match-ups using the trained model.
    """
    from src.features import FEATURE_COLS
    import pandas as pd

    all_teams = [t for grp in WC2026_GROUPS.values() for t in grp]
    sq = squad_stats.set_index("team") if not squad_stats.empty else pd.DataFrame()

    # Build feature rows for every possible match-up
    rows = []
    pairs = []
    for i, ta in enumerate(all_teams):
        for tb in all_teams:
            if ta == tb: continue
            pairs.append((ta, tb))

    # Use historical feature medians + team-specific overrides
    hist_medians = feature_df[FEATURE_COLS].median()

    for ta, tb in pairs:
        row = hist_medians.copy()
        # Override with team-specific values
        ta_info = squad_stats[squad_stats.team==ta].iloc[0] if not squad_stats.empty and ta in squad_stats.team.values else None
        tb_info = squad_stats[squad_stats.team==tb].iloc[0] if not squad_stats.empty and tb in squad_stats.team.values else None

        if ta_info is not None:
            row["home_elo_pre"]      = ta_info.elo_rating
            row["home_fifa_rank"]    = ta_info.fifa_rank
            row["home_xg_per_game"]  = ta_info.xg_per_game
            row["home_xga_per_game"] = ta_info.xga_per_game
            row["home_possession_pct"] = ta_info.possession_pct
        if tb_info is not None:
            row["away_elo_pre"]      = tb_info.elo_rating
            row["away_fifa_rank"]    = tb_info.fifa_rank
            row["away_xg_per_game"]  = tb_info.xg_per_game
            row["away_xga_per_game"] = tb_info.xga_per_game
            row["away_possession_pct"] = tb_info.possession_pct

        if ta_info is not None and tb_info is not None:
            row["elo_diff"]          = ta_info.elo_rating - tb_info.elo_rating
            row["elo_home_win_prob"] = 1/(1+10**(-(ta_info.elo_rating-tb_info.elo_rating)/400))
            row["fifa_rank_diff"]    = tb_info.fifa_rank - ta_info.fifa_rank
            row["diff_xg_per_game"]  = ta_info.xg_per_game - tb_info.xg_per_game

        row["is_world_cup"]       = 1
        row["is_knockout"]        = 0
        row["is_neutral"]         = 1
        row["tournament_weight"]  = 6
        row["mkt_home_prob"]      = row.get("elo_home_win_prob", 0.38)
        row["mkt_draw_prob"]      = 0.26
        row["mkt_away_prob"]      = 1 - row.get("elo_home_win_prob",0.38) - 0.26
        rows.append(row)

    X = pd.DataFrame(rows, columns=FEATURE_COLS).fillna(0)
    from src.model import predict_calibrated
    probs_arr = predict_calibrated(ensemble, calibrators, X)

    lookup = {}
    for i, (ta, tb) in enumerate(pairs):
        lookup[(ta, tb)] = tuple(probs_arr[i])
    return lookup
