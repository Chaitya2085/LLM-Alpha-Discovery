"""Leakage test: did the LLMs invent ideas, or remember what happened?

    python leakage_test.py            # India (default)

Every LLM learned from text up to its knowledge cutoff (config/llm_cutoffs.csv; fill in the ones
marked unknown from each model's published model card). If a model "remembered" which kinds of stocks
did well, its factors would look good in years it read about and fade straight after its cutoff.

For each model with a known cutoff, every factor it wrote is scored weekly (rank IC, in the direction
it worked in 2014-2020) in two windows of the held-out years:
  before   2021 up to the model's cutoff (years it could have read about)
  after    the cutoff to the latest data (truly unseen)
Ideas also fade with time for ordinary reasons, so the same calendar split is applied to every OTHER
model's factors as a placebo. The memory signal = this model's drop minus the placebo drop.
A drop more than 2 standard errors larger than the placebo is flagged.

Writes results/<market>/leakage.csv (per factor) and leakage_summary.json (per model).
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

import markets
from metrics import ic_series

ROOT = Path(__file__).resolve().parents[1]
CUTOFFS = ROOT / "config" / "llm_cutoffs.csv"
DISC = ("2014-07-01", "2020-12-31")
HOLD_START = "2021-01-01"
MIN_WEEKS = 26


def model_label(r: dict) -> str:
    if r.get("provider") == "claude-in-chat":
        return "seed: claude (chat)"
    return f"{r.get('provider')}: {r.get('model')}"


def load_cutoffs(path: Path = CUTOFFS) -> dict[str, pd.Timestamp]:
    if not path.exists():
        return {}
    c = pd.read_csv(path, comment="#", dtype=str).fillna("")
    out = {}
    for r in c.itertuples():
        if r.cutoff.strip() and r.cutoff.strip().lower() != "unknown":
            out[r.model.strip()] = pd.Timestamp(r.cutoff.strip())
    return out


def factor_ic_by_week(m, exprs: dict[str, str], hold: int = 5) -> pd.DataFrame:
    """Weekly rank IC (5-day forward returns, every 5th day) for each factor id."""
    from dsl import Evaluator
    o = m.open
    fwd = o.shift(-(hold + 1)) / o.shift(-1) - 1
    days = m.dates[::hold][:-2]
    ev = Evaluator(m)
    cols = {}
    for fid, ex in exprs.items():
        try:
            f = ev(ex)
        except Exception:  # noqa: BLE001
            continue
        cols[fid] = ic_series(f.loc[days], fwd.loc[days], m.member.loc[days], "spearman")
        ev.clear_cache()
    return pd.DataFrame(cols, index=days)


def run(cfg, cutoffs: dict | None = None, log=print) -> dict:
    from data import load_market
    from generate_factors import load_library
    cutoffs = load_cutoffs() if cutoffs is None else cutoffs
    lib = [r for r in load_library() if r.get("status") == "valid"]
    owner = {r["id"]: model_label(r) for r in lib}
    m = load_market(cfg.name)
    log(f"  scoring {len(lib)} factors weekly, {m.dates[0].date()} to {m.dates[-1].date()} ...")
    ic = factor_ic_by_week(m, {r["id"]: r["expression"] for r in lib})
    disc = ic.loc[DISC[0]:DISC[1]].mean()
    sign = np.sign(disc).replace(0, 1)
    ic = ic * sign                      # each factor in the direction it worked in 2014-2020
    end = ic.index[-1]
    rows = []
    for fid in ic.columns:
        rows.append({"id": fid, "model": owner.get(fid, "?"), "ic_discovery": float(disc[fid] * sign[fid]),
                     "ic_holdout": float(ic.loc[HOLD_START:, fid].mean())})
    per_factor = pd.DataFrame(rows)
    models = {}
    for label, cut in cutoffs.items():
        mine = [c for c in ic.columns if owner.get(c) == label]
        others = [c for c in ic.columns if owner.get(c) != label]
        if not mine or cut <= pd.Timestamp(HOLD_START) or cut >= end:
            models[label] = {"cutoff": str(cut.date()), "status": "cutoff outside the held-out years"}
            continue
        b = ic.loc[HOLD_START:cut]
        a = ic.loc[cut + pd.Timedelta(days=1):]
        if len(b) < MIN_WEEKS or len(a) < MIN_WEEKS:
            models[label] = {"cutoff": str(cut.date()), "status": f"too few weeks ({len(b)} before, {len(a)} after)"}
            continue
        drop_mine = (b[mine].mean() - a[mine].mean()).dropna()
        drop_other = (b[others].mean() - a[others].mean()).dropna()
        diff = float(drop_mine.mean() - drop_other.mean())
        se = math.sqrt(drop_mine.var() / len(drop_mine) + drop_other.var() / len(drop_other))
        z = diff / se if se > 0 else float("nan")
        models[label] = {
            "cutoff": str(cut.date()), "factors": len(drop_mine), "weeks_before": len(b), "weeks_after": len(a),
            "ic_before": float(b[mine].mean().mean()), "ic_after": float(a[mine].mean().mean()),
            "placebo_ic_before": float(b[others].mean().mean()), "placebo_ic_after": float(a[others].mean().mean()),
            "extra_drop": diff, "z": z,
            "status": "MEMORY SIGNAL" if z > 2 else "no memory signal",
        }
        log(f"    {label} (cutoff {cut.date()}): IC {models[label]['ic_before']:+.4f} before, "
            f"{models[label]['ic_after']:+.4f} after; extra drop vs others {diff:+.4f} (z {z:+.1f}) -> "
            f"{models[label]['status']}")
    unknown = sorted({owner[c] for c in ic.columns} - set(cutoffs))
    out = {"market": cfg.name, "holdout_start": HOLD_START, "latest": str(end.date()), "models": models,
           "models_without_cutoff": unknown,
           "all_factors": {"ic_discovery": float(per_factor["ic_discovery"].mean()),
                           "ic_holdout": float(per_factor["ic_holdout"].mean()),
                           "n": int(len(per_factor))}}
    cfg.results.mkdir(parents=True, exist_ok=True)
    per_factor.to_csv(cfg.results / "leakage.csv", index=False)
    (cfg.results / "leakage_summary.json").write_text(json.dumps(out, indent=2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--market", default=os.environ.get("LLM_ALPHA_MARKET", "india"))
    a = ap.parse_args()
    cfg = markets.get(a.market)
    s = run(cfg)
    print(json.dumps({k: v.get("status") for k, v in s["models"].items()}, indent=2))
    if s["models_without_cutoff"]:
        print("  no cutoff yet for: " + ", ".join(s["models_without_cutoff"]) + " (add them to config/llm_cutoffs.csv)")


if __name__ == "__main__":
    main()
