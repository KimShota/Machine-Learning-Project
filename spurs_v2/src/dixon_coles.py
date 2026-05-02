"""
dixon_coles.py
==============
Simplified Dixon-Coles model for Premier League match simulation.
Estimates attack/defense parameters for each team, then simulates
remaining fixtures via Poisson distributions.

Dixon & Coles (1997): "Modelling Association Football Scores and Inefficiencies
in the Football Betting Market", Applied Statistics.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson


# ── Low-score correction (Dixon-Coles tau) ──────────────────────────────────

def tau(x, y, lam, mu, rho):
    """Correction factor for low-scoring outcomes."""
    if x == 0 and y == 0:
        return 1 - lam * mu * rho
    elif x == 0 and y == 1:
        return 1 + lam * rho
    elif x == 1 and y == 0:
        return 1 + mu * rho
    elif x == 1 and y == 1:
        return 1 - rho
    return 1.0


def dc_log_likelihood(params, teams, home_goals, away_goals, home_idx, away_idx):
    n = len(teams)
    attack  = np.exp(params[:n])
    defense = np.exp(params[n:2*n])
    home_adv = np.exp(params[2*n])
    rho      = params[2*n + 1]

    ll = 0.0
    for i in range(len(home_goals)):
        hi, ai = home_idx[i], away_idx[i]
        lam = attack[hi] * defense[ai] * home_adv
        mu  = attack[ai] * defense[hi]
        t   = tau(home_goals[i], away_goals[i], lam, mu, rho)
        if t <= 0:
            return 1e10
        ll += (np.log(t)
               + poisson.logpmf(home_goals[i], lam)
               + poisson.logpmf(away_goals[i], mu))
    return -ll


class DixonColesModel:
    def __init__(self):
        self.attack  = {}
        self.defense = {}
        self.home_adv = 1.35
        self.rho = -0.13
        self.fitted = False

    def fit(self, matches_df):
        """
        Fit model on historical match data.
        matches_df must have: home_team, away_team, home_goals, away_goals
        """
        teams = sorted(set(matches_df["home_team"]) | set(matches_df["away_team"]))
        team_idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)

        home_idx   = matches_df["home_team"].map(team_idx).values
        away_idx   = matches_df["away_team"].map(team_idx).values
        home_goals = matches_df["home_goals"].values.astype(int)
        away_goals = matches_df["away_goals"].values.astype(int)

        # Initial params: log(attack)=0, log(defense)=0, log(home_adv)=log(1.35), rho=-0.13
        x0 = np.zeros(2 * n + 2)
        x0[2*n]     = np.log(1.35)
        x0[2*n + 1] = -0.13

        res = minimize(
            dc_log_likelihood,
            x0,
            args=(teams, home_goals, away_goals, home_idx, away_idx),
            method="L-BFGS-B",
            options={"maxiter": 500, "ftol": 1e-8},
        )

        params = res.x
        attack_raw  = np.exp(params[:n])
        defense_raw = np.exp(params[n:2*n])

        # Normalize so mean attack = 1
        mean_atk = attack_raw.mean()
        for i, t in enumerate(teams):
            self.attack[t]  = attack_raw[i] / mean_atk
            self.defense[t] = defense_raw[i] / mean_atk

        self.home_adv = np.exp(params[2*n])
        self.rho      = params[2*n + 1]
        self.teams    = teams
        self.fitted   = True

        return self

    def predict_scoreline(self, home, away, max_goals=8):
        """Return matrix of scoreline probabilities."""
        lam = self.attack.get(home, 1.0) * self.defense.get(away, 1.0) * self.home_adv
        mu  = self.attack.get(away, 1.0) * self.defense.get(home, 1.0)

        matrix = np.zeros((max_goals + 1, max_goals + 1))
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                t = tau(i, j, lam, mu, self.rho)
                matrix[i, j] = t * poisson.pmf(i, lam) * poisson.pmf(j, mu)
        return matrix

    def predict_outcome_probs(self, home, away):
        """Return (p_home_win, p_draw, p_away_win)."""
        m = self.predict_scoreline(home, away)
        p_home = np.tril(m, -1).sum()
        p_draw = np.diag(m).sum()
        p_away = np.triu(m, 1).sum()
        return p_home, p_draw, p_away

    def simulate_match(self, home, away, rng):
        """Simulate one match, return (home_goals, away_goals)."""
        lam = self.attack.get(home, 1.0) * self.defense.get(away, 1.0) * self.home_adv
        mu  = self.attack.get(away, 1.0) * self.defense.get(home, 1.0)
        return int(rng.poisson(lam)), int(rng.poisson(mu))

    def default_team(self, team_name, strength="average"):
        """Register an unknown team with a default strength."""
        strength_map = {
            "strong":  {"attack": 1.4, "defense": 0.75},
            "average": {"attack": 1.0, "defense": 1.0},
            "weak":    {"attack": 0.7, "defense": 1.35},
        }
        s = strength_map.get(strength, strength_map["average"])
        self.attack[team_name]  = s["attack"]
        self.defense[team_name] = s["defense"]


# ── Season simulator ─────────────────────────────────────────────────────────

class SeasonSimulator:
    """
    Monte Carlo simulation of the remaining PL season.
    """

    def __init__(self, dc_model):
        self.dc = dc_model

    def simulate_remaining(self, current_standings, remaining_fixtures, n_sims=50000, seed=42):
        """
        Simulate n_sims completions of the season.

        current_standings: list of dicts {team, pts, gd, gf, ga}
        remaining_fixtures: list of (home_team, away_team)

        Returns dict: team → P(relegated), team → P(safe), distribution of final points
        """
        rng = np.random.default_rng(seed)
        teams = [s["team"] for s in current_standings]

        base_pts = {s["team"]: s["pts"] for s in current_standings}
        base_gd  = {s["team"]: s["gd"]  for s in current_standings}
        base_gf  = {s["team"]: s["gf"]  for s in current_standings}

        relegated_counts = {t: 0 for t in teams}
        final_pts_dist   = {t: [] for t in teams}

        for _ in range(n_sims):
            pts = dict(base_pts)
            gd  = dict(base_gd)
            gf  = dict(base_gf)

            for home, away in remaining_fixtures:
                gh, ga = self.dc.simulate_match(home, away, rng)
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

            # Sort: pts desc, gd desc, gf desc
            final_table = sorted(teams,
                key=lambda t: (pts[t], gd[t], gf[t]), reverse=True)

            for rank, team in enumerate(final_table):
                if rank >= len(teams) - 3:
                    relegated_counts[team] += 1
                final_pts_dist[team].append(pts[team])

        n = n_sims
        relegation_prob = {t: relegated_counts[t] / n for t in teams}
        avg_final_pts   = {t: np.mean(final_pts_dist[t]) for t in teams}
        p10_pts         = {t: np.percentile(final_pts_dist[t], 10) for t in teams}
        p90_pts         = {t: np.percentile(final_pts_dist[t], 90) for t in teams}

        return {
            "relegation_prob": relegation_prob,
            "avg_final_pts":   avg_final_pts,
            "p10_final_pts":   p10_pts,
            "p90_final_pts":   p90_pts,
            "final_pts_dist":  final_pts_dist,
        }


if __name__ == "__main__":
    import pandas as pd
    from data_generator import build_full_dataset

    print("Building dataset for DC fitting...")
    df = build_full_dataset()
    print(f"  {len(df)} team-seasons loaded")

    # The DC model needs match-level data; we'll test with dummy matches
    matches = pd.DataFrame({
        "home_team":   ["Arsenal", "Chelsea", "Liverpool", "Tottenham", "Arsenal"],
        "away_team":   ["Chelsea", "Liverpool", "Arsenal", "Arsenal", "Tottenham"],
        "home_goals":  [2, 1, 3, 0, 1],
        "away_goals":  [1, 1, 1, 2, 0],
    })
    model = DixonColesModel()
    model.fit(matches)
    print("  DC model fitted")

    ph, pd_, pa = model.predict_outcome_probs("Arsenal", "Tottenham")
    print(f"  Arsenal vs Tottenham: H={ph:.1%} D={pd_:.1%} A={pa:.1%}")
