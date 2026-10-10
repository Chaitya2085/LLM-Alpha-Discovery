"""Health check of the Indian dataset (run after build_india.py; takes seconds).

    python src/india_diagnose.py

Prints a short report and saves results/india/diagnostics.json:
  1. Splits and bonuses the builder adjusted for, per year, and the ones worth a second look
     (price move not exactly a standard ratio).
  2. Big price moves it left alone because they don't look like a split or bonus (real news,
     or a corporate action it couldn't recognise).
  3. The biggest one-day moves left in the top-200 after adjustment, and VWAP sanity.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import build_india  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default=None)
    ap.add_argument("--results", default=str(ROOT / "results" / "india"))
    ap.add_argument("--big", type=float, default=0.25, help="what counts as a big one-day move")
    a = ap.parse_args()
    proc = Path(a.processed) if a.processed else build_india.OUT
    px = pd.read_parquet(proc / "prices.parquet")
    ev = pd.read_csv(proc / "corporate_actions.csv", parse_dates=["date"])
    out: dict = {}

    done = ev[ev["adjusted"]]
    out["adjusted"] = {k: int(v) for k, v in done["kind"].value_counts().items()}
    out["adjusted_per_year"] = {int(k): int(v) for k, v in done.groupby(done["date"].dt.year).size().items()}
    print(f"1) Adjusted: {out['adjusted'].get('split', 0)} splits/consolidations (ISIN changed), "
          f"{out['adjusted'].get('bonus', 0)} bonuses")
    print("   per year: " + ", ".join(f"{k}:{v}" for k, v in out["adjusted_per_year"].items()))
    unsure = done[done["match"] > 0.05].sort_values("match", ascending=False)
    out["adjusted_check"] = [f"{r.date.date()} {r.symbol} {r.kind}, shares x{r.ratio:g} (price x{r.price_move})"
                             for r in unsure.itertuples()]
    print(f"   worth a check (price move not exactly the ratio): {len(unsure)}")
    for line in out["adjusted_check"][:12]:
        print("     " + line)

    left = ev[~ev["adjusted"]].reindex(ev[~ev["adjusted"]]["price_move"].sort_values().index)
    out["left_alone"] = [f"{r.date.date()} {r.symbol} x{r.price_move} ({r.kind} candidate; nearest shares x{r.ratio:g}, "
                         f"traded value x{r.value_x})" for r in left.itertuples()]
    print(f"\n2) Big drops left as real moves (not a clean split/bonus): {len(left)}")
    for line in out["left_alone"][:15]:
        print("     " + line)

    in_u = px[px["in_universe"]]
    big = in_u["ret_1d"].dropna()
    big = big[big.abs() > a.big]
    big = big.reindex(big.abs().sort_values(ascending=False).index)
    out["big_moves_total"] = int(len(big))
    out["big_moves_per_year"] = {int(k): int(v) for k, v in big.groupby(big.index.get_level_values("date").year).size().items()}
    out["big_moves_top"] = [f"{d.date()} {c} {r:+.1%}" for (d, c), r in big.head(20).items()]
    print(f"\n3) One-day moves over {a.big:.0%} on top-200 days after adjustment: {len(big)}")
    print("   per year: " + ", ".join(f"{k}:{v}" for k, v in out["big_moves_per_year"].items()))
    for line in out["big_moves_top"]:
        print("     " + line)

    ok = in_u["vwap"].notna() & in_u["low"].notna()
    inside = ((in_u["vwap"] >= in_u["low"] * 0.999) & (in_u["vwap"] <= in_u["high"] * 1.001))[ok]
    out["vwap_inside_low_high_pct"] = round(100 * float(inside.mean()), 2)
    print(f"\n   VWAP inside the day's low-high range: {out['vwap_inside_low_high_pct']}% of top-200 days")

    res = Path(a.results)
    res.mkdir(parents=True, exist_ok=True)
    (res / "diagnostics.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved {res / 'diagnostics.json'}")


if __name__ == "__main__":
    main()
