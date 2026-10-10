"""Do LLM factors travel? Compare each factor's discovery-period score in China and India.

Needs both markets scored (score_factors.py, then score_factors.py --market india).
Writes results/india/transfer.csv, results/india/transfer.json, figures/india_transfer.png
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import markets
from style import BLUE, INK, INK2, MUTED, SURFACE, new_figure, save, tidy, titles

CHINA, INDIA = markets.get("china"), markets.get("india")
T_STRONG = 3.0


def main() -> None:
    fc, fi = CHINA.results / "phase2_factor_scores.csv", INDIA.results / "phase2_factor_scores.csv"
    if not (fc.exists() and fi.exists()):
        raise SystemExit("Score factors on both markets first (score_factors.py with and without --market india).")
    c = pd.read_csv(fc)
    i = pd.read_csv(fi)
    c, i = c[c["status"] == "scored"], i[i["status"] == "scored"]
    m = c[["id", "name", "model", "expression", "rank_ic", "ic_t"]].merge(
        i[["id", "rank_ic", "ic_t", "top_q_excess"]], on="id", suffixes=("_china", "_india"))
    m["same_sign"] = np.sign(m["rank_ic_china"]) == np.sign(m["rank_ic_india"])
    m["strong_china"] = m["ic_t_china"].abs() >= T_STRONG
    m["strong_india"] = m["ic_t_india"].abs() >= T_STRONG
    m["strong_both"] = m["strong_china"] & m["strong_india"] & m["same_sign"]
    m["min_abs_t"] = np.minimum(m["ic_t_china"].abs(), m["ic_t_india"].abs()).where(m["same_sign"], 0)
    m = m.sort_values("min_abs_t", ascending=False)
    INDIA.results.mkdir(parents=True, exist_ok=True)
    m.to_csv(INDIA.results / "transfer.csv", index=False)

    sc = m[m["strong_china"]]
    summary = {
        "factors_compared": int(len(m)),
        "ic_correlation": round(float(m["rank_ic_china"].corr(m["rank_ic_india"])), 3),
        "ic_rank_correlation": round(float(m["rank_ic_china"].corr(m["rank_ic_india"], method="spearman")), 3),
        "strong_in_china": int(len(sc)),
        "strong_in_china_same_sign_in_india": int(sc["same_sign"].sum()),
        "strong_in_china_also_strong_in_india": int(sc["strong_both"].sum()),
        "strong_in_india": int(m["strong_india"].sum()),
        "strong_in_both": int(m["strong_both"].sum()),
        "top_travellers": m.head(5)[["name", "rank_ic_china", "rank_ic_india"]].round(4).to_dict("records"),
    }
    (INDIA.results / "transfer.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    plot(m, summary)


def plot(m: pd.DataFrame, s: dict) -> None:
    fig, ax = new_figure(8.6, 7)
    lim = max(0.05, float(np.nanmax(np.abs(m[["rank_ic_china", "rank_ic_india"]].values))) * 1.12)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.plot([-lim, lim], [-lim, lim], color=MUTED, lw=0.9, ls=(0, (3, 3)))
    ax.text(lim * 0.97, lim * 0.9, "same strength\nin both", ha="right", va="top", fontsize=8, color=INK2)
    both, rest = m[m["strong_both"]], m[~m["strong_both"]]
    ax.scatter(rest["rank_ic_china"], rest["rank_ic_india"], s=36, color=MUTED, edgecolor=SURFACE, lw=1.3,
               label="Other factors", zorder=2)
    ax.scatter(both["rank_ic_china"], both["rank_ic_india"], s=56, color=BLUE, edgecolor=SURFACE, lw=1.3,
               label=f"Strong in both markets (|t| ≥ {T_STRONG:g}, same sign)", zorder=3)
    for _, r in both.head(4).iterrows():
        ax.annotate(r["name"], (r["rank_ic_china"], r["rank_ic_india"]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8.5, color=INK)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    tidy(ax, grid_axis="both")
    ax.set_xlabel("Rank IC in China (CSI300, 2014–2020)", color=INK2, fontsize=9)
    ax.set_ylabel("Rank IC in India (top-200 NSE, 2014–2020)", color=INK2, fontsize=9)
    n = s["strong_in_both"]
    titles(ax, f"{n} LLM factor{'s' if n != 1 else ''} work in both China and India",
           f"{s['factors_compared']} factors; correlation of their scores across markets: "
           f"{s['ic_correlation']:+.2f}")
    ax.legend(loc="lower right", frameon=False, fontsize=8.5, labelcolor=INK)
    save(fig, "india_transfer.png")


if __name__ == "__main__":
    main()
