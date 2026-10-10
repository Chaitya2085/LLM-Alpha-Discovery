"""Phase 2 figures, computed from results/ (titles included):

  figures/phase2_strength_vs_novelty.png  every factor: predictive strength vs overlap with the baseline
  figures/phase2_models.png               each LLM side by side: valid formulas, strong and shortlisted factors
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from style import BLUE, INK, INK2, MUTED, NOVEL_ZONE, SURFACE, new_figure, save, tidy, titles

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
MAX_LABELS = 4  # the README shortlist table lists every shortlisted factor


def strength_vs_novelty() -> None:
    s = pd.read_csv(RES / "phase2_factor_scores.csv")
    s = s[s["status"] == "scored"].copy()
    s["abs_ic"] = s["rank_ic"].abs()
    summ = json.loads((RES / "phase2_summary.json").read_text())
    ref = summ["best_alpha158_single_feature_rank_ic"]
    short = s[s["shortlist"]].sort_values("abs_ic", ascending=False)
    rest = s[~s["shortlist"]]
    n_beat = int((short["abs_ic"] > ref["rank_ic"]).sum())
    ymax = max(0.055, s["abs_ic"].max() * 1.15)

    fig, ax = new_figure(10, 6)
    ax.axvspan(0, 0.7, color=NOVEL_ZONE, zorder=0)
    ax.text(0.015, ymax * 0.94, "novel zone (corr < 0.7 with every Alpha158 feature)", color=INK2, fontsize=8.5)
    ax.scatter(rest["max_corr_a158"], rest["abs_ic"], s=40, color=MUTED, edgecolor=SURFACE, lw=1.5,
               label="Other factors", zorder=2)
    ax.scatter(short["max_corr_a158"], short["abs_ic"], s=60, color=BLUE, edgecolor=SURFACE, lw=1.5,
               label="Shortlist: strong, novel, not redundant", zorder=3)
    # label the strongest shortlisted factors, nudging labels down so they never overlap
    placed = []  # (x, y) of label anchors in axes fraction
    for _, r in short.head(MAX_LABELS).iterrows():
        flip = " (flipped)" if r["direction"] < 0 else ""
        x, y = r["max_corr_a158"], r["abs_ic"] / ymax
        ly = y
        while any(abs(ly - py) < 0.032 and abs(x - px) < 0.42 for px, py in placed):
            ly -= 0.034
        placed.append((x, ly))
        ax.annotate(r["name"] + flip, (r["max_corr_a158"], r["abs_ic"]),
                    xytext=(x + 0.012, ly * ymax), textcoords="data", va="center", fontsize=8.5,
                    color=INK, arrowprops=None if abs(ly - y) < 1e-9 else
                    dict(arrowstyle="-", color=MUTED, lw=0.7, shrinkA=0, shrinkB=3))
    ax.axhline(ref["rank_ic"], color=INK2, lw=0.9, ls=(0, (3, 3)))
    ax.text(0.015, ref["rank_ic"] - ymax * 0.033,
            f"best single Alpha158 feature ({ref['feature']}): {ref['rank_ic']:.3f}", color=INK2, fontsize=8.5)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, ymax)
    tidy(ax, grid_axis="both")
    ax.set_xlabel("Max |rank correlation| with any of the 158 baseline features   (lower = more novel)",
                  color=INK2, fontsize=9)
    ax.set_ylabel("|Rank IC| vs next-day return   (higher = more predictive)", color=INK2, fontsize=9)
    if n_beat:
        title = (f"{n_beat} LLM factor{'s' if n_beat > 1 else ''} beat{'' if n_beat > 1 else 's'} "
                 "every baseline feature while staying novel")
    elif len(short):
        title = f"{len(short)} novel, strong LLM factors so far; none beats the best baseline feature yet"
    else:
        title = "No LLM factor is yet both strong and novel"
    p = summ["discovery_period"]
    titles(ax, title, f"{len(s)} LLM-written factors from {summ.get('models', 1)} model(s), scored on "
                      f"{p[0][:4]}–{p[1][:4]} only (2021 onward held out); top {MAX_LABELS} labelled")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.93), frameon=False, fontsize=9, labelcolor=INK)
    save(fig, "phase2_strength_vs_novelty.png")


def model_comparison() -> None:
    m = pd.read_csv(RES / "phase2_model_summary.csv")
    # rates per 20 proposed, so a model that proposed more isn't favoured
    m["strong_per20"] = 20 * m["strong"] / m["proposed"]
    m["short_per20"] = 20 * m["shortlist"] / m["proposed"]
    m = m.sort_values(["short_per20", "strong_per20"], ascending=True)
    labels = [f"{r.model}  ({r.proposed} proposed)" for r in m.itertuples()]
    panels = [("valid_pct", "Valid formulas (%)", "{:.0f}%"),
              ("strong_per20", "Strong factors per 20 proposed", "{:.1f}"),
              ("short_per20", "Shortlisted per 20 proposed", "{:.1f}")]
    h = max(2.6, 0.55 * len(m) + 1.6)
    fig, axes = new_figure(11, h)
    fig.clf()
    axes = fig.subplots(1, 3, sharey=True)
    y = np.arange(len(m))
    for ax, (col, name, fmt) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        vals = m[col].fillna(0).to_numpy()
        ax.barh(y, vals, color=BLUE, height=0.55)
        top = max(vals.max(), 1)
        for yi, v in zip(y, vals):
            ax.text(v + top * 0.02, yi, fmt.format(v), va="center", fontsize=8.5, color=INK)
        ax.set_xlim(0, top * 1.22)
        ax.set_ylim(-0.7, max(len(m), 2.5) - 0.3)  # keep bars thin when few models
        ax.set_title(name, loc="left", fontsize=9.5, color=INK)
        tidy(ax, grid_axis="x")
        ax.set_xticks([])
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=8.5, color=INK)
    best = m.iloc[-1]
    title = (f"Which LLM writes the most useful factors? So far: {best.model}" if len(m) > 1
             else f"Factor quality for {best.model}; add more models to compare")
    fig.suptitle(title, x=0.01, y=0.985, ha="left", fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "phase2_models.png", tight=False)


if __name__ == "__main__":
    strength_vs_novelty()
    model_comparison()
