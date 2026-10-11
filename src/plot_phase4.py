"""Phase 4 figure, computed from results/<market>/phase4_rounds.csv:

  figures/<prefix>phase4_rounds.png   share of factors that work weekly after costs, by round:
                                      each LLM with feedback (solid) vs without (dashed)
"""
import json
from pathlib import Path

import pandas as pd
from matplotlib.lines import Line2D

import markets
from style import INK, INK2, MUTED, SERIES, new_figure, save, tidy, titles

ROOT = Path(__file__).resolve().parents[1]
CFG = markets.from_argv() if __name__ == "__main__" else markets.get()
MAX_LINES = len(SERIES)


def short(label: str) -> str:
    name = label.split(": ", 1)[-1].split("/")[-1].replace(":free", "")
    return name if len(name) <= 22 else name[:21] + "…"


def rounds_chart(cfg=CFG) -> Path | None:
    f = cfg.results / "phase4_rounds.csv"
    if not f.exists():
        print("    no Phase 4 results yet")
        return None
    try:
        r = pd.read_csv(f)
    except pd.errors.EmptyDataError:
        r = pd.DataFrame()
    if r.empty:
        print("    no Phase 4 rounds yet (run with --providers)")
        return None
    s = json.loads((cfg.results / "phase4_summary.json").read_text())
    researchers = list(r["researcher"].drop_duplicates())[:MAX_LINES]
    fig, ax = new_figure(10, 5.6)
    rmax = int(r["round"].max())
    ytop = max(10.0, r["useful_pct"].max() * 1.45)   # room for the legend
    for i, lab in enumerate(researchers):
        c = SERIES[i]
        g = r[r["researcher"] == lab]
        r0 = g[g["round"] == 0]
        for arm, ls in (("feedback", "-"), ("control", (0, (4, 3)))):
            ga = g[g["arm"] == arm].sort_values("round")
            if not len(ga):
                continue
            ga = pd.concat([r0, ga])
            ax.plot(ga["round"], ga["useful_pct"], color=c, lw=2, ls=ls, marker="o" if arm == "feedback" else None,
                    ms=6, mec="#fcfcfb", mew=1.5, zorder=3)
        end = g[g["arm"] == "feedback"].sort_values("round")
        if len(end):
            ax.annotate(short(lab), (end["round"].iloc[-1], end["useful_pct"].iloc[-1]), xytext=(8, 0),
                        textcoords="offset points", va="center", fontsize=8.5, color=INK)
    ax.set_xticks(range(0, rmax + 1))
    ax.set_xticklabels(["Phase 2\nfactors"] + [f"round {k}" for k in range(1, rmax + 1)])
    ax.set_xlim(-0.2, rmax + 0.9)
    ax.set_ylim(0, ytop)
    ax.set_ylabel("Factors that work after costs (% of proposed)", color=INK2, fontsize=9)
    tidy(ax)
    handles = [Line2D([], [], color=SERIES[i], lw=2, label=short(l)) for i, l in enumerate(researchers)]
    handles += [Line2D([], [], color=MUTED, lw=2, label="with Critic feedback"),
                Line2D([], [], color=MUTED, lw=2, ls=(0, (4, 3)), label="without (control)")]
    ax.legend(handles=handles, frameon=False, fontsize=8.5, loc="upper left", ncol=2, labelcolor=INK2)
    fb, ct = s.get("useful_pct_feedback"), s.get("useful_pct_control")
    if fb is not None and ct is not None:
        title = f"With feedback {fb:.0f}% of new factors work after costs; without, {ct:.0f}%"
    elif fb is not None:
        title = f"With feedback {fb:.0f}% of new factors work after costs"
    else:
        title = "Phase 4: feedback rounds"
    titles(ax, title, f"{cfg.universe_label}, judged weekly: buy the top fifth, hold 5 days, after costs, "
                      f"vs holding all stocks equally ({s['period'][0][:4]}–{s['period'][1][:4]})")
    return save(fig, f"{cfg.fig_prefix}phase4_rounds.png")


if __name__ == "__main__":
    rounds_chart()
