"""src/report.py — Generate all output CSVs and charts."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from itertools import combinations

WC2026_GROUPS = {
    "A":["Mexico","South Africa","South Korea","Czechia"],
    "B":["Canada","Bosnia and Herzegovina","Qatar","Switzerland"],
    "C":["Brazil","Morocco","Haiti","Scotland"],
    "D":["United States","Paraguay","Australia","Turkey"],
    "E":["Germany","Cuba","Ivory Coast","Ecuador"],
    "F":["Portugal","Indonesia","Algeria","Croatia"],
    "G":["Belgium","Egypt","Venezuela","Colombia"],
    "H":["Spain","Senegal","Chile","Japan"],
    "I":["France","Nigeria","Bolivia","Slovakia"],
    "J":["Argentina","Jordan","Kenya","New Zealand"],
    "K":["Netherlands","Costa Rica","Iran","Ghana"],
    "L":["England","Honduras","Saudi Arabia","Panama"],
}

CONF_COLORS = {
    "UEFA":"#3B82F6","CONMEBOL":"#10B981","CAF":"#F59E0B",
    "AFC":"#EF4444","CONCACAF":"#8B5CF6","OFC":"#6B7280",
}
TEAM_CONF = {
    "Argentina":"CONMEBOL","France":"UEFA","Spain":"UEFA","England":"UEFA",
    "Brazil":"CONMEBOL","Portugal":"UEFA","Netherlands":"UEFA","Germany":"UEFA",
    "Belgium":"UEFA","Morocco":"CAF","Croatia":"UEFA","Colombia":"CONMEBOL",
    "Uruguay":"CONMEBOL","Japan":"AFC","Mexico":"CONCACAF","United States":"CONCACAF",
    "Senegal":"CAF","Switzerland":"UEFA","Ecuador":"CONMEBOL","South Korea":"AFC",
    "Australia":"AFC","Turkey":"UEFA","Nigeria":"CAF","Iran":"AFC","Canada":"CONCACAF",
    "Egypt":"CAF","Paraguay":"CONMEBOL","Peru":"CONMEBOL","Ivory Coast":"CAF",
    "Ghana":"CAF","Saudi Arabia":"AFC","Slovakia":"UEFA","Scotland":"UEFA",
    "Jordan":"AFC","Czechia":"UEFA","South Africa":"CAF","Venezuela":"CONMEBOL",
    "Chile":"CONMEBOL","Bolivia":"CONMEBOL","Indonesia":"AFC","New Zealand":"OFC",
    "Costa Rica":"CONCACAF","Honduras":"CONCACAF","Kenya":"CAF","Panama":"CONCACAF",
    "Bosnia and Herzegovina":"UEFA","Qatar":"AFC","Haiti":"CONCACAF","Cuba":"CONCACAF",
    "Trinidad and Tobago":"CONCACAF","Algeria":"CAF",
}


def generate_group_predictions(probs_lookup, n_sims=20000):
    """Simulate group stage n_sims times to get qualification probabilities."""
    from src.simulate import simulate_group
    qual_counts = {t:{"1st":0,"2nd":0,"3rd":0,"4th":0} for grp in WC2026_GROUPS.values() for t in grp}
    for _ in range(n_sims):
        for grp, teams in WC2026_GROUPS.items():
            standings = simulate_group(teams, probs_lookup)
            for idx, row in standings.iterrows():
                pos_label = ["1st","2nd","3rd","4th"][min(idx,3)]
                qual_counts[row.team][pos_label] += 1
    rows = []
    for team, counts in qual_counts.items():
        total = sum(counts.values())
        grp = next(g for g,ts in WC2026_GROUPS.items() if team in ts)
        rows.append(dict(
            group=grp, team=team,
            p_1st=round(counts["1st"]/total,3),
            p_2nd=round(counts["2nd"]/total,3),
            p_3rd=round(counts["3rd"]/total,3),
            p_4th=round(counts["4th"]/total,3),
            p_advance=round((counts["1st"]+counts["2nd"])/total,3),
        ))
    return pd.DataFrame(rows).sort_values(["group","p_1st"],ascending=[True,False])


def generate_all_group_matches(probs_lookup):
    """Return a DataFrame of all 48 group stage matches with predicted probabilities."""
    rows = []
    match_num = 1
    for grp, teams in WC2026_GROUPS.items():
        for t1, t2 in combinations(teams, 2):
            ph, pd_, pa = probs_lookup.get((t1,t2), (0.38,0.26,0.36))
            rows.append(dict(
                match=match_num, group=grp,
                home_team=t1, away_team=t2,
                p_home_win=round(ph,3), p_draw=round(pd_,3), p_away_win=round(pa,3),
                predicted=("Home" if ph>pa and ph>pd_ else ("Draw" if pd_>pa else "Away")),
            ))
            match_num += 1
    return pd.DataFrame(rows)


def plot_winner_probabilities(mc_results, out_path):
    """Bar chart of top-20 tournament winner probabilities."""
    top = mc_results.head(20).copy()
    top["conf"] = top.team.map(TEAM_CONF).fillna("UEFA")
    colors = [CONF_COLORS.get(c,"#6B7280") for c in top.conf]

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.barh(top.team[::-1], top.win_probability[::-1]*100, color=colors[::-1], height=0.7)

    for bar, prob in zip(bars, top.win_probability[::-1]):
        ax.text(bar.get_width()+0.1, bar.get_y()+bar.get_height()/2,
                f"{prob*100:.1f}%", va="center", fontsize=9, fontweight="bold")

    ax.set_xlabel("Tournament Win Probability (%)", fontsize=11)
    ax.set_title("World Cup 2026 — Predicted Winner Probabilities\n(XGBoost + CatBoost Ensemble, Monte Carlo)", 
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_xlim(0, top.win_probability.max()*100 * 1.18)
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    legend = [mpatches.Patch(color=c, label=k) for k,c in CONF_COLORS.items()]
    ax.legend(handles=legend, loc="lower right", fontsize=9, framealpha=0.7)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Chart saved → {out_path}")


def plot_group_heatmap(group_preds, out_path):
    """Heatmap of advancement probability by group."""
    pivot = group_preds.pivot_table(index="team", columns="group", values="p_advance").fillna(0)
    grp_order = list(WC2026_GROUPS.keys())
    team_order = group_preds.sort_values(["group","p_advance"],ascending=[True,False]).team.tolist()
    pivot = pivot.reindex(index=team_order, columns=grp_order, fill_value=0)
    # Only show non-zero cells
    mask = pivot == 0

    fig, ax = plt.subplots(figsize=(16, 14))
    import matplotlib.colors as mcolors
    cmap = plt.cm.RdYlGn
    im = ax.imshow(pivot.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i,j]
            if val > 0:
                color = "black" if 0.3 < val < 0.7 else "white"
                ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                        fontsize=8, color=color, fontweight="bold")

    ax.set_xticks(range(len(grp_order))); ax.set_xticklabels([f"Group {g}" for g in grp_order], fontsize=10)
    ax.set_yticks(range(len(team_order))); ax.set_yticklabels(team_order, fontsize=9)
    ax.set_title("World Cup 2026 — Group Stage Advancement Probability\n(% chance of finishing 1st or 2nd)", 
                 fontsize=13, fontweight="bold", pad=15)
    plt.colorbar(im, ax=ax, fraction=0.02, label="Probability")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Chart saved → {out_path}")


def plot_feature_importance(fi_series, out_path, top_n=20):
    """Horizontal bar chart of top feature importances."""
    top = fi_series.head(top_n)
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(top.index[::-1], top.values[::-1], color="#3B82F6", height=0.6)
    ax.set_xlabel("Feature Importance (XGBoost)", fontsize=11)
    ax.set_title(f"Top {top_n} Most Important Features", fontsize=13, fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Chart saved → {out_path}")


def plot_cv_results(cv_df, out_path):
    """Line chart of walk-forward CV RPS and accuracy."""
    fig, (ax1,ax2) = plt.subplots(1,2, figsize=(12,4))
    ax1.plot(cv_df.val_year, cv_df.rps, "o-", color="#EF4444", linewidth=2, markersize=8)
    ax1.axhline(0.20, color="gray", linestyle="--", alpha=0.7, label="Target RPS < 0.20")
    ax1.set_title("RPS per CV Fold (lower = better)", fontweight="bold")
    ax1.set_xlabel("Validation Year"); ax1.set_ylabel("Ranked Probability Score")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(cv_df.val_year, cv_df.accuracy*100, "o-", color="#10B981", linewidth=2, markersize=8)
    ax2.axhline(54, color="gray", linestyle="--", alpha=0.7, label="Baseline ~54%")
    ax2.set_title("3-Way Accuracy per CV Fold", fontweight="bold")
    ax2.set_xlabel("Validation Year"); ax2.set_ylabel("Accuracy (%)")
    ax2.set_ylim(30,75); ax2.legend(); ax2.grid(alpha=0.3)

    fig.suptitle("Walk-Forward Cross-Validation Results", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Chart saved → {out_path}")
