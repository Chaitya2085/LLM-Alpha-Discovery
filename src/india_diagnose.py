"""Health check of the Indian dataset (run after build_india.py; takes seconds).

    python src/india_diagnose.py

Prints a short report and saves results/india/diagnostics.json:
  1. Splits, bonuses and 'other' events (demergers etc.) the builder adjusted for. Check the
     'other' list: a real crash in it belongs in config/india_overrides.csv as 'keep'.
  2. Big drops it kept as real price moves (a bonus among them goes in the overrides as 'adjust').
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
    print(f"1) Adjusted: {out['adjusted'].get('split', 0)} splits (ISIN changed), "
          f"{out['adjusted'].get('bonus', 0)} bonuses, {out['adjusted'].get('other', 0)} other "
          "(demergers etc., treated as return-neutral)")
    print("   per year: " + ", ".join(f"{k}:{v}" for k, v in out["adjusted_per_year"].items()))
    other = done[done["kind"] == "other"].sort_values("price_move")
    out["other"] = [f"{r.date.date()} {r.symbol} price x{r.price_move}" for r in other.itertuples()]
    print(f"   'other' events (if one is a REAL crash, add it to config/india_overrides.csv as keep): {len(other)}")
    for line in out["other"][:40]:
        print("     " + line)

    kept = ev[~ev["adjusted"]].sort_values("price_move")
    out["kept"] = [f"{r.date.date()} {r.symbol} price x{r.price_move} open x{r.open_gap} {r.note}".rstrip()
                   for r in kept.itertuples()]
    print(f"\n2) Big drops kept as real moves: {len(kept)} (if one is a bonus, add it to the overrides as "
          "adjust with its ratio)")
    for line in out["kept"][:40]:
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
