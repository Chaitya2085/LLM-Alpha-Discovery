"""Phase 5 figures, computed from results/<market>/phase5_*.csv:

  figures/<prefix>phase5_growth.png        the Top 20 book after costs vs the index, test years
  figures/<prefix>phase5_calibration.png   does "x% confident" come true x% of the time?
"""
import json

import pandas as pd

import markets
from style import BLUE, GRID, INK, INK2, MUTED, ORANGE, new_figure, save, tidy, titles

CFG = markets.from_argv() if __name__ == "__main__" else markets.get()
COLORS = {"Alpha158 + LLM factors": BLUE, "LLM factors only": BLUE, "Alpha158 only": ORANGE}


def growth(cfg=CFG):
    f = cfg.results / "phase5_weekly.csv"
    if not f.exists():
        print("    no Phase 5 results yet")
        return None
    w = pd.read_csv(f, parse_dates=["date"])
    s = json.loads((cfg.results / "phase5_summary.json").read_text())
    w = w[w["book"] == "Top 20"]
    fig, ax = new_figure(10, 5.6)
    ends = []
    bench = None
    for name, g in w.groupby("variant", sort=False):
        g = g.set_index("date").sort_index()
        eq = (1 + g["net"]).cumprod()
        ax.plot(eq.index, eq, color=COLORS.get(name, BLUE), lw=2, zorder=3)
        ends.append((eq.iloc[-1], eq.index[-1], f"{name}  {eq.iloc[-1]:.2f}", COLORS.get(name, BLUE)))
        bench = (1 + g["bench"]).cumprod()
    ax.plot(bench.index, bench, color=MUTED, lw=2, zorder=2)
    ends.append((bench.iloc[-1], bench.index[-1], f"{cfg.bench_label} index  {bench.iloc[-1]:.2f}", MUTED))
    ax.axhline(1, color=INK2, lw=0.8, ls=(0, (3, 3)))
    ymin, ymax = ax.get_ylim()
    placed = []
    for y, x, text, c in sorted(ends):
        ly = y
        while any(abs(ly - p) < (ymax - ymin) * 0.05 for p in placed):
            ly += (ymax - ymin) * 0.05
        placed.append(ly)
        ax.annotate(text, (x, y), xytext=(8, (ly - y) / (ymax - ymin) * 300), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK)
    ax.set_xlim(right=bench.index[-1] + pd.Timedelta(days=330))
    ax.set_ylabel("Growth of 1 (after costs)", color=INK2, fontsize=9)
    tidy(ax)
    best = s["variants"].get("Alpha158 + LLM factors") or next(iter(s["variants"].values()))
    b = best["books"]["Top 20"]
    yrs = s["test_years"]
    titles(ax, f"Top 20 earns {b['ann_return']:.1%} a year after costs, vs {b['bench_ann_return']:.1%} for the index",
           f"Each week hold the 20 stocks the model is most confident about, {cfg.universe_label}, "
           f"held 5 days, {cfg.buy_cost:.2%} buy / {cfg.sell_cost:.2%} sell; walk-forward test {yrs[0]}–{yrs[-1]}")
    return save(fig, f"{cfg.fig_prefix}phase5_growth.png")


def calibration(cfg=CFG):
    f = cfg.results / "phase5_calibration.csv"
    if not f.exists():
        return None
    c = pd.read_csv(f)
    s = json.loads((cfg.results / "phase5_summary.json").read_text())
    fig, ax = new_figure(7.5, 6)
    lo = min(c["predicted"].min(), c["actual"].min()) - 0.03
    hi = max(c["predicted"].max(), c["actual"].max()) + 0.03
    ax.plot([lo, hi], [lo, hi], color=INK2, lw=0.9, ls=(0, (3, 3)), zorder=1)
    ax.text(hi - 0.005, hi - 0.012, "perfect calibration", ha="right", va="top", color=INK2, fontsize=8.5,
            rotation=0)
    for name, g in c.groupby("variant", sort=False):
        col = COLORS.get(name, BLUE)
        ax.plot(g["predicted"], g["actual"], color=col, lw=2, zorder=2)
        ax.scatter(g["predicted"], g["actual"], s=(g["n"] / g["n"].max() * 160).clip(lower=14), color=col,
                   edgecolor="#fcfcfb", lw=1.5, zorder=3, label=name)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Model's confidence that the stock beats the index next week", color=INK2, fontsize=9)
    ax.set_ylabel("How often it actually did", color=INK2, fontsize=9)
    from matplotlib.ticker import MultipleLocator
    step = 0.02 if hi - lo < 0.25 else 0.05
    ax.xaxis.set_major_locator(MultipleLocator(step))
    ax.yaxis.set_major_locator(MultipleLocator(step))
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    tidy(ax, grid_axis="both")
    ax.grid(color=GRID, lw=0.8)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=INK2)
    v = s["variants"].get("Alpha158 + LLM factors") or next(iter(s["variants"].values()))
    cs = v["calibration"]
    titles(ax, f"Confidence vs outcome: {cs['ece']:.1%} apart on average (target: under 5%)",
           f"Each dot is a group of stock-weeks (bigger = more); base rate {cs['base_rate']:.0%} of stocks beat "
           f"the index in a week")
    return save(fig, f"{cfg.fig_prefix}phase5_calibration.png")


if __name__ == "__main__":
    growth()
    calibration()
