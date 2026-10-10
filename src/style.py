"""Shared chart style so every figure in the project looks like one set.

Colours come from a palette checked for colour-blind separation and contrast.
All figures are written to figures/ by code; the README embeds them from there.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
MUTED = "#b4b2a9"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
NOVEL_ZONE = "#eef4fc"


def new_figure(w: float = 10, h: float = 5.6):
    fig, ax = plt.subplots(figsize=(w, h), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    return fig, ax


def tidy(ax, grid_axis: str = "y") -> None:
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, lw=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.set_axisbelow(True)


def titles(ax, title: str, subtitle: str = "") -> None:
    ax.set_title(title, loc="left", color=INK, fontsize=13, pad=26 if subtitle else 12)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=INK2, fontsize=8.5)


def save(fig, name: str, tight: bool = True) -> Path:
    FIG.mkdir(exist_ok=True)
    out = FIG / name
    if tight:
        fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    print(f"    wrote {out.relative_to(ROOT)}")
    return out
