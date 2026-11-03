"""Phase 6: the final exam. One run on the years nothing has seen (2021 onward), with every setting
frozen beforehand.

    python phase6_exam.py --freeze     # 1. write config/phase6_frozen.json from Phase 5 (then commit it!)
    python phase6_exam.py --run        # 2. the exam itself, once
    python phase6_exam.py --report     # rebuild tables from the saved exam (no new run)

How it stays honest
  * Frozen first: the trading design, the model settings, the exact LLM factor list, the two versions
    (Alpha158 only, and with LLM factors) and the PASS MARKS are written to config/phase6_frozen.json
    before the exam. Committing that file to git timestamps the promise.
  * One shot: the first run writes results/<market>/phase6_runlog.json. Any later run must give a
    reason (--rerun-reason "...") and is recorded there, so the number of attempts is public.
  * Same machinery: each test year is predicted by a model trained only on earlier years, exactly as
    in Phase 5, with the factor list and design fixed.
  * Luck counted: the result is judged with the Deflated Sharpe Ratio, which asks how likely the edge
    is real after counting the trading designs tried in Phase 5.

Pass marks (pre-registered in the frozen file):
  REAL EDGE        after-cost excess over the index > 0 and Deflated Sharpe probability >= 0.95
  PROMISING        excess > 0 and plain probability (PSR) >= 0.95, but not after counting the tries
  NO PROVEN EDGE   anything else
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import honest_stats as hs
import markets
import phase5_model as p5

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "config" / "phase6_frozen.json"
HOLDOUT_START = "2021-01-01"
PASS_MARKS = {"real_edge_dsr": 0.95, "promising_psr": 0.95, "ic_t_min": 2.0, "calibration_gap_max": 0.05}


def _hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def freeze(cfg) -> dict:
    s = json.loads((cfg.results / "phase5_summary.json").read_text())
    g = pd.read_csv(cfg.results / "phase5_grid.csv")
    variants = list(s["variants"])
    designs = {v: s["variants"][v]["chosen_design"] for v in variants}
    weekly_sr = (g["info_ratio"] / np.sqrt(p5.WEEKS)).dropna()
    frozen = {
        "market": cfg.name, "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "holdout_start": HOLDOUT_START, "variants": variants, "designs": designs,
        "llm_factors": [{"id": f["id"], "name": f["name"], "expression": f["expression"]}
                        for f in s["llm_factors"]],
        "model_params": p5.PARAMS, "train_start": p5.TRAIN_START, "hold_days": p5.HOLD,
        "trials": {"count": int(len(g)), "weekly_sharpe_variance": float(weekly_sr.var()),
                   "what": "trading designs x model versions tried in Phase 5 (2017-2020)"},
        "pass_marks": PASS_MARKS,
        "phase5_reference": {v: s["variants"][v]["books"] for v in variants},
    }
    frozen["fingerprint"] = _hash({k: v for k, v in frozen.items() if k not in ("frozen_at", "phase5_reference")})
    FROZEN.parent.mkdir(parents=True, exist_ok=True)
    FROZEN.write_text(json.dumps(frozen, indent=2))
    return frozen


def verdict(excess: list[float], frozen: dict) -> dict:
    pm = frozen["pass_marks"]
    tr = frozen["trials"]
    mean_ex = float(np.mean(excess)) if excess else float("nan")
    p_plain = hs.psr(excess, 0.0)
    p_defl = hs.dsr(excess, tr["count"], tr["weekly_sharpe_variance"])
    if mean_ex > 0 and p_defl >= pm["real_edge_dsr"]:
        word = "REAL EDGE"
    elif mean_ex > 0 and p_plain >= pm["promising_psr"]:
        word = "PROMISING"
    else:
        word = "NO PROVEN EDGE"
    return {"verdict": word, "psr": p_plain, "dsr": p_defl,
            "luck_benchmark_weekly_sharpe": hs.expected_max_sr(tr["count"], tr["weekly_sharpe_variance"]),
            "weekly_sharpe": hs.sharpe(excess), "excess_t": hs.t_stat(excess), "weeks": len(excess)}


def _runlog(cfg) -> Path:
    return cfg.results / "phase6_runlog.json"


def check_lock(cfg, frozen: dict, reason: str | None) -> list:
    f = _runlog(cfg)
    log = json.loads(f.read_text()) if f.exists() else []
    if log and not reason:
        raise SystemExit(f"The exam has already run {len(log)} time(s) (first {log[0]['ran_at']}). "
                         "Running it again must be recorded: add --rerun-reason \"why\". "
                         "Use --report to rebuild the tables from the saved run.")
    if log and log[-1]["fingerprint"] != frozen["fingerprint"]:
        print("    note: the frozen settings changed since the last run; this is recorded")
    return log


def run(cfg, reason: str | None = None, threads: int = 2, log=print, cutoff: str = "2262-01-01") -> dict:
    if not FROZEN.exists():
        raise SystemExit("Freeze the settings first:  python phase6_exam.py --freeze  (then commit the file)")
    frozen = json.loads(FROZEN.read_text())
    runs = check_lock(cfg, frozen, reason)
    t0 = time.time()
    m = p5.load_cut(cfg, cutoff)
    fwd5, bench5, y, y_rank, yex_long, label_end = p5.targets(m)
    last_label = label_end[label_end < pd.Timestamp("2262-01-01")]
    last_year = int(m.dates[-1].year)
    years = list(range(int(frozen["holdout_start"][:4]), last_year + 1))
    log(f"  exam years {years[0]}-{years[-1]} (last day with a full week after it: "
        f"{last_label.index[-1].date()}); {len(frozen['llm_factors'])} frozen LLM factors")
    X_all = p5.build_features(m, cfg, frozen["llm_factors"], use_a158=True)
    a158_cols = [c for c in X_all.columns if not c.startswith("llm_")]
    data = {"Alpha158 only": X_all[a158_cols], "Alpha158 + LLM factors": X_all}
    res = cfg.results
    out = {"market": cfg.name, "fingerprint": frozen["fingerprint"], "years": years, "variants": {}}
    weekly_rows = []
    for name in frozen["variants"]:
        X = data.get(name, X_all)
        d = frozen["designs"][name]
        log(f"  {name}: training year by year ...")
        pred, info, _ = p5.walk_forward(X, y, label_end, years, threads, log, y_train=y_rank)
        pred = pred[pred.index.get_level_values("date") >= pd.Timestamp(frozen["holdout_start"])]
        w = p5.portfolio(pred, m, fwd5, bench5, d["stocks"], None, d["keep_zone"], d["trade_every_weeks"])
        st = p5.book_stats(w)
        ex = (w["net"] - w["bench"]).tolist()
        cal_t, cal_s = p5.calibration(pred, y)
        v = {"design": d, "training": info, "book": st, **p5.weekly_ic(pred, yex_long),
             "calibration": cal_s, "verdict": verdict(ex, frozen),
             "by_year": {int(k): float(g.mean() * p5.WEEKS)
                         for k, g in (w["net"] - w["bench"]).groupby(w.index.year)}}
        out["variants"][name] = v
        weekly_rows.append(w.assign(variant=name).reset_index())
        cal_t.assign(variant=name).to_csv(res / f"phase6_calibration_{'llm' if 'LLM' in name else 'a158'}.csv",
                                          index=False)
        log(f"    {name}: {st['ann_return']:+.1%}/yr vs index {st['bench_ann_return']:+.1%}; "
            f"excess {st['excess_ann']:+.1%}; verdict {v['verdict']['verdict']}")
    out["seconds"] = round(time.time() - t0)
    pd.concat(weekly_rows).to_csv(res / "phase6_weekly.csv", index=False)
    (res / "phase6_summary.json").write_text(json.dumps(out, indent=2, default=str))
    runs.append({"ran_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "fingerprint": frozen["fingerprint"], "reason": reason or "first run",
                 "verdicts": {k: v["verdict"]["verdict"] for k, v in out["variants"].items()}})
    _runlog(cfg).write_text(json.dumps(runs, indent=2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--market", default=os.environ.get("LLM_ALPHA_MARKET", "india"))
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--freeze", action="store_true")
    g.add_argument("--run", action="store_true")
    g.add_argument("--report", action="store_true")
    ap.add_argument("--rerun-reason", default=None)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    cfg = markets.get(a.market)
    os.environ["LLM_ALPHA_MARKET"] = cfg.name
    if a.freeze:
        if FROZEN.exists() and _runlog(cfg).exists():
            raise SystemExit("The exam has already run on these frozen settings; refreezing would hide that.")
        f = freeze(cfg)
        print(f"  frozen: {', '.join(f'{v}: {d}' for v, d in f['designs'].items())}")
        print(f"  {len(f['llm_factors'])} LLM factors, {f['trials']['count']} trials counted; "
              f"fingerprint {f['fingerprint'][:12]}")
        print("  Now commit it before running the exam:\n"
              "    git add config/phase6_frozen.json && git commit -m \"Phase 6: freeze settings\" && git push")
    elif a.run:
        s = run(cfg, a.rerun_reason, a.threads)
        print(json.dumps({k: v["verdict"] for k, v in s["variants"].items()}, indent=2))
    else:
        if not (cfg.results / "phase6_summary.json").exists():
            raise SystemExit("No exam has run yet.")
        print((cfg.results / "phase6_summary.json").read_text()[:2000])


if __name__ == "__main__":
    main()
