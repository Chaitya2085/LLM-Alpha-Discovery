"""Phase 1 figure: growth of 1 unit, CSI300 vs the LightGBM model before and after
trading costs (Top-50 / Dropout-5). Title is computed from the results.

Output: figures/phase1_growth.png
"""
from pathlib import Path

import pandas as pd

from backtest import topk_dropout
from data import load_market
from metrics import perf
from style import AQUA, BLUE, INK, INK2, ORANGE, new_figure, save, tidy, titles

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    m = load_market()
    s = pd.read_parquet(ROOT / "results" / "lgb_alpha158_scores.parquet")["score"] \
        .unstack("code").reindex(index=m.dates, columns=m.member.columns)
    d = topk_dropout(s, m, topk=50, n_drop=5, start="2017-01-01").daily
    g, n, b = (perf(d[c])["ann_return"] for c in ("gross", "net", "bench"))
    series = [
        ("Model, before costs", (1 + d["gross"]).cumprod(), BLUE),
        ("Model, after costs", (1 + d["net"]).cumprod(), ORANGE),
        ("CSI300 index", (1 + d["bench"]).cumprod(), AQUA),
    ]

    fig, ax = new_figure(10, 5.4)
    # end labels, nudged apart when two series finish close together
    ends = sorted(((eq.iloc[-1], label) for label, eq, _ in series), reverse=True)
    yr = max(eq.max() for _, eq, _ in series) - min(eq.min() for _, eq, _ in series)
    offset, prev = {}, None
    for y, label in ends:
        offset[label] = 0 if prev is None or prev - y > 0.05 * yr else -10
        if offset[label] and prev is not None:
            offset[[l for v, l in ends if v == prev][0]] = 6
        prev = y
    for label, eq, col in series:
        ax.plot(eq.index, eq.values, color=col, lw=1.6, solid_capstyle="round", label=label)
        ax.annotate(f"{label}  {eq.iloc[-1]:.2f}", xy=(eq.index[-1], eq.iloc[-1]), xytext=(8, offset[label]),
                    textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.axhline(1, color=INK2, lw=0.8, ls=(0, (3, 3)))
    tidy(ax)
    ax.set_ylabel(f"Growth of 1 (start {d.index[0]:%b %Y})", color=INK2, fontsize=9)
    titles(ax, f"Before costs the model earns {g:.1%} a year; after costs {n:.1%}, vs {b:.1%} for the index",
           "Alpha158 + LightGBM, CSI300 stocks, top 50 held, 5 swapped daily, 0.05% buy / 0.15% sell costs, "
           f"{d.index[0]:%b %Y} to {d.index[-1]:%b %Y}")
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK)
    ax.set_xlim(d.index[0], d.index[-1] + pd.Timedelta(days=720))
    years = range(d.index[0].year + 1, d.index[-1].year + 1)
    ax.set_xticks([pd.Timestamp(f"{y}-01-01") for y in years])
    ax.set_xticklabels([str(y) for y in years])
    save(fig, "phase1_growth.png")


if __name__ == "__main__":
    main()
