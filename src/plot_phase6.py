"""Phase 6 figure, computed from results/<market>/phase6_weekly.csv:

  figures/<prefix>phase6_exam.png   both frozen versions vs the index on the held-out years, after costs
"""
import json

import pandas as pd

import markets
from style import BLUE, INK, INK2, MUTED, new_figure, save, tidy, titles

CFG = markets.from_argv() if __name__ == "__main__" else markets.get()
COLORS = {"Alpha158 + LLM factors": BLUE, "LLM factors only": BLUE, "Alpha158 only": INK}


def exam_chart(cfg=CFG):
    f = cfg.results / "phase6_weekly.csv"
    if not f.exists():
        print("    no Phase 6 exam yet")
        return None
    w = pd.read_csv(f, parse_dates=["date"])
    s = json.loads((cfg.results / "phase6_summary.json").read_text())
    fig, ax = new_figure(10, 5.6)
    ends, bench = [], None
    for name, g in w.groupby("variant", sort=False):
        g = g.set_index("date").sort_index()
        eq = (1 + g["net"]).cumprod()
        ls = "-" if "LLM" in name else (0, (4, 3))
        ax.plot(eq.index, eq, color=COLORS.get(name, BLUE), lw=2, ls=ls, zorder=3)
        ends.append((eq.iloc[-1], eq.index[-1], f"{name}  {eq.iloc[-1]:.2f}"))
        bench = (1 + g["bench"]).cumprod()
    ax.plot(bench.index, bench, color=MUTED, lw=2, zorder=2)
    ends.append((bench.iloc[-1], bench.index[-1], f"{cfg.bench_label} index  {bench.iloc[-1]:.2f}"))
    ax.axhline(1, color=INK2, lw=0.8, ls=(0, (3, 3)))
    lo, hi = ax.get_ylim()
    placed = []
    for y, x, text in sorted(ends):
        ly = y
        while any(abs(ly - p) < (hi - lo) * 0.05 for p in placed):
            ly += (hi - lo) * 0.05
        placed.append(ly)
        ax.annotate(text, (x, y), xytext=(x + pd.Timedelta(days=20), ly), textcoords="data", va="center",
                    fontsize=8.5, color=INK)
    ax.set_xlim(right=bench.index[-1] + pd.Timedelta(days=420))
    ax.set_ylabel("Growth of 1 (after costs)", color=INK2, fontsize=9)
    tidy(ax)
    verdicts = "; ".join(f"{k}: {v['verdict']['verdict'].lower()}" for k, v in s["variants"].items())
    titles(ax, f"Final exam on {s['years'][0]}–{s['years'][-1]}: {verdicts}",
           f"Settings frozen before the run; {cfg.universe_label}, {cfg.buy_cost:.2%} buy / {cfg.sell_cost:.2%} "
           "sell; each year predicted by a model trained only on earlier years")
    return save(fig, f"{cfg.fig_prefix}phase6_exam.png")


if __name__ == "__main__":
    exam_chart()
