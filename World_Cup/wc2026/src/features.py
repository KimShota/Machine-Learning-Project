"""src/features.py — Feature engineering pipeline."""
from pathlib import Path

import numpy as np
import pandas as pd

TOURNAMENT_WEIGHT = {
    "FIFA World Cup": 6, "UEFA Euro": 4, "Copa América": 4,
    "AFC Asian Cup": 4, "Africa Cup of Nations": 4, "CONCACAF Gold Cup": 4,
    "Confederations Cup": 3, "FIFA World Cup qualification": 3,
    "UEFA Nations League": 2, "Friendly": 1,
}

FEATURE_COLS = [
    "elo_diff","elo_home_win_prob",
    "home_form_5","home_gf_5","home_ga_5",
    "away_form_5","away_gf_5","away_ga_5",
    "home_form_10","home_gf_10","home_ga_10",
    "away_form_10","away_gf_10","away_ga_10",
    "home_h2h_wr_5","away_h2h_wr_5",
    "home_h2h_wr_10","away_h2h_wr_10",
    "fifa_rank_diff","home_fifa_rank","away_fifa_rank",
    "home_elo_pre","away_elo_pre",
    "tournament_weight","is_world_cup","is_knockout","is_neutral",
    "venue_altitude_m","home_travel_km","away_travel_km",
    "home_rest_days","away_rest_days",
    "mkt_home_prob","mkt_draw_prob","mkt_away_prob",
    "home_xg_per_game","away_xg_per_game",
    "home_xga_per_game","away_xga_per_game",
    "home_possession_pct","away_possession_pct",
    "diff_xg_per_game","diff_squad_market_value_meur","diff_avg_caps",
]

def _rolling_team_stats(df, window):
    df = df.sort_values("date").reset_index(drop=True)
    team_history = {t: [] for t in set(df.home_team)|set(df.away_team)}
    cols = [f"home_form_{window}",f"home_gf_{window}",f"home_ga_{window}",
            f"away_form_{window}",f"away_gf_{window}",f"away_ga_{window}",
            f"home_h2h_wr_{window}",f"away_h2h_wr_{window}"]
    feats = {c:[] for c in cols}
    for _,row in df.iterrows():
        ht,at,date = row.home_team,row.away_team,row.date
        def stats(team):
            h = [m for m in team_history[team] if m["date"]<date][-window:]
            if not h: return 0.5,1.2,1.2
            w=sum(1 for m in h if m["pts"]==3); d=sum(1 for m in h if m["pts"]==1)
            return (w*3+d)/(len(h)*3), np.mean([m["gf"] for m in h]), np.mean([m["ga"] for m in h])
        def h2h(team,opp):
            h=[m for m in team_history[team] if m["date"]<date and m["opp"]==opp][-window:]
            return sum(1 for m in h if m["pts"]==3)/len(h) if h else 0.33
        hf,hgf,hga=stats(ht); af,agf,aga=stats(at)
        feats[f"home_form_{window}"].append(hf); feats[f"home_gf_{window}"].append(hgf); feats[f"home_ga_{window}"].append(hga)
        feats[f"away_form_{window}"].append(af); feats[f"away_gf_{window}"].append(agf); feats[f"away_ga_{window}"].append(aga)
        feats[f"home_h2h_wr_{window}"].append(h2h(ht,at)); feats[f"away_h2h_wr_{window}"].append(h2h(at,ht))
        hpts=3 if row.home_score>row.away_score else (1 if row.home_score==row.away_score else 0)
        apts=3 if row.away_score>row.home_score else (1 if row.home_score==row.away_score else 0)
        team_history[ht].append(dict(date=date,gf=row.home_score,ga=row.away_score,pts=hpts,opp=at))
        team_history[at].append(dict(date=date,gf=row.away_score,ga=row.home_score,pts=apts,opp=ht))
    for c,v in feats.items(): df[c]=v
    return df

def _merge_elo(df, elo):
    elo=elo.sort_values("date")
    def get_elo(team,date):
        sub=elo[(elo.team==team)&(elo.date<date)]
        return float(sub.elo.iloc[-1]) if len(sub) else 1500.0
    df=df.copy()
    df["home_elo_pre"]=[row.home_elo if "home_elo" in df.columns and not np.isnan(row.home_elo)
                        else get_elo(row.home_team,row.date) for _,row in df.iterrows()]
    df["away_elo_pre"]=[row.away_elo if "away_elo" in df.columns and not np.isnan(row.away_elo)
                        else get_elo(row.away_team,row.date) for _,row in df.iterrows()]
    df["elo_diff"]=df.home_elo_pre-df.away_elo_pre
    df["elo_home_win_prob"]=1/(1+10**(-df.elo_diff/400))
    return df

def _merge_fifa(df, fifa):
    fifa=fifa.sort_values("rank_date")
    def get_rank(team,date):
        sub=fifa[(fifa.country_full==team)&(fifa.rank_date<date)]
        return int(sub["rank"].iloc[-1]) if len(sub) else 100
    df=df.copy()
    df["home_fifa_rank"]=[get_rank(r.home_team,r.date) for _,r in df.iterrows()]
    df["away_fifa_rank"]=[get_rank(r.away_team,r.date) for _,r in df.iterrows()]
    df["fifa_rank_diff"]=df.away_fifa_rank-df.home_fifa_rank
    return df

def _merge_odds(df, odds):
    if odds.empty:
        for c in ["mkt_home_prob","mkt_draw_prob","mkt_away_prob"]: df[c]=np.nan
        return df
    odds=odds.copy(); odds["Date"]=pd.to_datetime(odds["Date"])
    for pfx,h,d,a in [("b365","B365H","B365D","B365A"),("ps","PSH","PSD","PSA")]:
        if h in odds.columns:
            inv=1/odds[h]+1/odds[d]+1/odds[a]
            odds[f"{pfx}_ph"]=(1/odds[h])/inv; odds[f"{pfx}_pd"]=(1/odds[d])/inv; odds[f"{pfx}_pa"]=(1/odds[a])/inv
    odds["mkt_home_prob"]=odds[[c for c in odds.columns if c.endswith("_ph")]].mean(axis=1)
    odds["mkt_draw_prob"]=odds[[c for c in odds.columns if c.endswith("_pd")]].mean(axis=1)
    odds["mkt_away_prob"]=odds[[c for c in odds.columns if c.endswith("_pa")]].mean(axis=1)
    merged=df.merge(odds[["Date","HomeTeam","AwayTeam","mkt_home_prob","mkt_draw_prob","mkt_away_prob"]],
                    left_on=["date","home_team","away_team"],right_on=["Date","HomeTeam","AwayTeam"],how="left")
    merged=merged.drop(columns=["Date","HomeTeam","AwayTeam"],errors="ignore")
    return merged

def _merge_squad(df, squad):
    if squad.empty: return df
    sq=squad.set_index("team")
    for stat in ["xg_per_game","xga_per_game","possession_pct","squad_avg_age","squad_market_value_meur","avg_caps"]:
        if stat in sq.columns:
            df[f"home_{stat}"]=df.home_team.map(sq[stat])
            df[f"away_{stat}"]=df.away_team.map(sq[stat])
            if any(k in stat for k in ["xg","value","caps"]):
                df[f"diff_{stat}"]=df[f"home_{stat}"]-df[f"away_{stat}"]
    return df

def build_feature_matrix(matches, elo, fifa, squad, odds):
    print("  Engineering features …")
    df=matches.copy(); df["date"]=pd.to_datetime(df["date"])
    df["tournament_weight"]=df.tournament.map(TOURNAMENT_WEIGHT).fillna(1).astype(int)
    df["is_world_cup"]=(df.tournament=="FIFA World Cup").astype(int)
    df["is_knockout"]=df.tournament.isin({"FIFA World Cup","UEFA Euro","Copa América","AFC Asian Cup",
        "Africa Cup of Nations","CONCACAF Gold Cup","Confederations Cup"}).astype(int)
    df["is_neutral"]=df.neutral.astype(int)
    df["outcome"]=np.where(df.home_score>df.away_score,"H",np.where(df.home_score<df.away_score,"A","D"))
    df["outcome_int"]=df.outcome.map({"H":0,"D":1,"A":2})
    df=_rolling_team_stats(df,5)
    df=_rolling_team_stats(df,10)
    df=_merge_elo(df,elo)
    df=_merge_fifa(df,fifa)
    df=_merge_odds(df,odds)
    df=_merge_squad(df,squad)
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
    for col in FEATURE_COLS:
        med = df[col].median()
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(med if pd.notna(med) else 0.0)
    print(f"    → {len(df):,} rows, {sum(1 for c in FEATURE_COLS if c in df.columns)} features")
    return df


def build_features(save=True, data_dir=None):
    """
    Load raw CSVs from data/, build the match-level feature matrix, optionally save merged_dataset.csv.
    Expects: 01_match_results.csv … 05_betting_odds.csv from generate_data.build_and_save_all.
    """
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent.parent / "data"
    data_dir = Path(data_dir)

    matches = pd.read_csv(data_dir / "01_match_results.csv", parse_dates=["date"])
    elo = pd.read_csv(data_dir / "02_elo_ratings.csv", parse_dates=["date"])
    fifa = pd.read_csv(data_dir / "03_fifa_rankings.csv", parse_dates=["rank_date"])
    squad = pd.read_csv(data_dir / "04_squad_stats.csv")
    odds = pd.read_csv(data_dir / "05_betting_odds.csv")

    df = build_feature_matrix(matches, elo, fifa, squad, odds)
    df["year"] = df["date"].dt.year

    if save:
        out = data_dir / "merged_dataset.csv"
        df.to_csv(out, index=False)
        print(f"    Saved → {out} ({len(df):,} rows)")
    return df
