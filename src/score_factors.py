"""Score every valid factor in the library on the DISCOVERY period only.

Discovery: 2014-07-01 -> end of 2020. Everything from 2021 onward is the
holdout for Phase 5 and is never touched here, so choices made while looking at
these scores can't leak into the final test.

Per factor:
  coverage        share of index-member days with a value
  rank_ic         mean daily Spearman corr between factor and next-day return
  rank_icir       rank_ic / its std  (consistency)
  ic_t            t-statistic of rank_ic
  years_same_sign share of discovery years whose IC has the overall sign
  autocorr        day-to-day rank correlation of the factor (high = slow = cheap to trade)
  q5_q1_ann       annualised gross return of top-fifth minus bottom-fifth (sign-adjusted)
  top_q_excess    annualised gross excess of the top fifth over the equal-weight universe
  max_corr_a158   largest |rank corr| with any Alpha158 feature, and which one (novelty)
  max_corr_llm    largest |rank corr| with a stronger LLM factor (redundancy)

Also summarises each LLM (valid rate, how many strong/novel factors it found)
in results/phase2_model_summary.csv, so different models can be compared.

Writes results/phase2_factor_scores.csv, phase2_model_summary.csv, phase2_summary.json.
Use --no-seed to leave out the seed batch written in the chat session.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from data import load_market
from dsl import Evaluator, FactorError
from generate_factors import load_library
from metrics import ic_series

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
DISC_START, DISC_END = "2014-07-01", "2020-12-31"
N_SAMPLE_DATES = 80


def _cs_rank_centered(mat: np.ndarray) -> np.ndarray:
    """Rank each row (date) of a [dates x stocks] array, centre, NaN -> 0."""
    r = pd.DataFrame(mat).rank(axis=1, pct=True).to_numpy()
    r = r - np.nanmean(r, axis=1, keepdims=True)
    return np.nan_to_num(r)


def _avg_rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Average over dates of the cross-sectional rank correlation of a and b ([dates x stocks])."""
    num = (a * b).sum(1)
    den = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    ok = den > 0
    return float(np.mean(num[ok] / den[ok])) if ok.any() else np.nan


SEED_PROVIDER = "claude-in-chat"


def model_label(r: dict) -> str:
    if r.get("provider") == SEED_PROVIDER:
        return "seed: claude (chat)"
    return f"{r.get('provider')}: {r.get('model')}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-seed", action="store_true", help="exclude the seed batch written in the chat session")
    args = ap.parse_args()
    t0 = time.time()
    m = load_market()
    dates = m.dates[(m.dates >= DISC_START) & (m.dates <= DISC_END)]
    dates = dates[:-3]  # forward return of the last days would use 2021 prices
    assert dates[-1] < pd.Timestamp("2021-01-01")
    member = m.member.loc[dates]
    fwd = m.fwd_ret.loc[dates]
    sample = dates[np.linspace(0, len(dates) - 1, N_SAMPLE_DATES).astype(int)]

    # Alpha158 ranks on the sample dates (for the novelty check)
    X = pd.read_parquet(ROOT / "data" / "processed" / "alpha158.parquet")
    X = X[X.index.get_level_values("date").isin(sample)]
    a158 = {}
    for col in X.columns:
        w = X[col].unstack("code").reindex(index=sample, columns=m.member.columns)
        a158[col] = _cs_rank_centered(w.where(m.member.loc[sample]).to_numpy())
    del X

    full_lib = load_library()
    if args.no_seed:
        full_lib = [r for r in full_lib if r.get("provider") != SEED_PROVIDER]
    lib = [r for r in full_lib if r["status"] == "valid"]
    if not lib:
        raise SystemExit("No valid factors in the library yet. Generate some first (see README).")
    ev = Evaluator(m)
    rows, ranks = [], {}
    for r in lib:
        try:
            f = ev(r["expression"])
        except (FactorError, Exception) as e:  # noqa: BLE001  (runtime failure = reject, keep going)
            rows.append({"id": r["id"], "name": r["name"], "status": "eval_error", "error": str(e)[:200],
                         "model": model_label(r)})
            continue
        fd = f.loc[dates]
        cov = float(fd.notna().where(member).stack().mean())
        ic = ic_series(fd, fwd, member, "spearman").dropna()
        if len(ic) < 250 or ic.std() == 0:
            rows.append({"id": r["id"], "name": r["name"], "status": "too_sparse", "coverage": cov,
                         "model": model_label(r)})
            continue
        sign = 1.0 if ic.mean() >= 0 else -1.0
        by_year = ic.groupby(ic.index.year).mean()
        ac = ic_series(fd, fd.shift(1), member, "spearman").mean()

        q = fd.where(member).rank(axis=1, pct=True) * sign + (0 if sign > 0 else 1)
        top, bot = q > 0.8, q <= 0.2
        univ = fwd.where(member).mean(axis=1)
        r_top = fwd.where(top).mean(axis=1)
        r_bot = fwd.where(bot).mean(axis=1)

        fr = _cs_rank_centered(f.loc[sample].where(m.member.loc[sample]).to_numpy())
        ranks[r["id"]] = fr * sign
        corrs = {k: _avg_rank_corr(fr, v) for k, v in a158.items()}
        best = max(corrs, key=lambda k: abs(corrs[k]))

        rows.append({
            "id": r["id"], "name": r["name"], "category": r["category"], "expression": r["expression"],
            "status": "scored", "coverage": cov, "direction": int(sign),
            "rank_ic": ic.mean(), "rank_icir": ic.mean() / ic.std(),
            "ic_t": ic.mean() / ic.std() * np.sqrt(len(ic)),
            "years_same_sign": float((np.sign(by_year) == sign).mean()),
            "autocorr": ac,
            "q5_q1_ann": (r_top - r_bot).mean() * 252,
            "top_q_excess": (r_top - univ).mean() * 252,
            "max_corr_a158": abs(corrs[best]), "closest_a158": best,
            "n_nodes": r.get("n_nodes"), "lookback": r.get("lookback"),
            "hypothesis": r["hypothesis"], "model": model_label(r),
        })
        print(f"  {r['id']} {r['name'][:32]:32s} RankIC {ic.mean():+.4f}  t {rows[-1]['ic_t']:+6.2f}  "
              f"autocorr {ac:.2f}  max|corr| A158 {abs(corrs[best]):.2f} ({best})", flush=True)
        ev.clear_cache()

    df = pd.DataFrame(rows)
    sc = df[df["status"] == "scored"].copy()
    sc["abs_ic"] = sc["rank_ic"].abs()
    sc = sc.sort_values("abs_ic", ascending=False)
    # redundancy among LLM factors: corr with any STRONGER factor
    order = list(sc["id"])
    red, red_with = [], []
    for i, fid in enumerate(order):
        best_c, best_id = 0.0, ""
        for other in order[:i]:
            c = abs(_avg_rank_corr(ranks[fid], ranks[other]))
            if c > best_c:
                best_c, best_id = c, other
        red.append(best_c)
        red_with.append(best_id)
    sc["max_corr_llm"], sc["closest_llm"] = red, red_with

    # preliminary flags only; Phase 3 makes the real decisions
    sc["flag_strong"] = (sc["ic_t"].abs() >= 3) & (sc["abs_ic"] >= 0.015) & (sc["years_same_sign"] >= 0.7)
    sc["flag_novel"] = sc["max_corr_a158"] < 0.7
    sc["flag_slow"] = sc["autocorr"] >= 0.8
    sc["flag_unique"] = sc["max_corr_llm"] < 0.7
    sc["shortlist"] = sc[["flag_strong", "flag_novel", "flag_unique"]].all(axis=1)

    out = pd.concat([sc, df[df["status"] != "scored"]], ignore_index=True)
    RES.mkdir(exist_ok=True)
    out.drop(columns=["abs_ic"]).to_csv(RES / "phase2_factor_scores.csv", index=False)

    # ---- per-model comparison ----
    models = []
    for label in sorted({model_label(r) for r in full_lib}):
        recs = [r for r in full_lib if model_label(r) == label]
        s_m = sc[sc["model"] == label]
        best = s_m.iloc[0] if len(s_m) else None
        models.append({
            "model": label, "proposed": len(recs),
            "valid_pct": 100 * sum(r["status"] == "valid" for r in recs) / len(recs),
            "invalid_pct": 100 * sum(r["status"] == "invalid" for r in recs) / len(recs),
            "duplicate_pct": 100 * sum(r["status"] == "duplicate" for r in recs) / len(recs),
            "scored": len(s_m),
            "median_abs_ic": float(s_m["abs_ic"].median()) if len(s_m) else np.nan,
            "strong": int(s_m["flag_strong"].sum()), "novel": int(s_m["flag_novel"].sum()),
            "shortlist": int(s_m["shortlist"].sum()),
            "wrong_sign_pct": 100 * float((s_m["direction"] < 0).mean()) if len(s_m) else np.nan,
            "best_factor": best["name"] if best is not None else "",
            "best_rank_ic": float(best["rank_ic"]) if best is not None else np.nan,
        })
    pd.DataFrame(models).to_csv(RES / "phase2_model_summary.csv", index=False)

    summary = {
        "discovery_period": [str(dates[0].date()), str(dates[-1].date())],
        "factors_in_library": len(full_lib), "models": len(models), "seed_included": not args.no_seed,
        "valid": len(lib), "scored": int(len(sc)),
        "strong": int(sc["flag_strong"].sum()), "novel": int(sc["flag_novel"].sum()),
        "slow": int(sc["flag_slow"].sum()), "shortlist": int(sc["shortlist"].sum()),
        "best_alpha158_single_feature_rank_ic": None,
        "seconds": round(time.time() - t0),
    }
    # reference point: the strongest single Alpha158 feature on the same period
    best_a = _best_alpha158(m, dates, member, fwd)
    summary["best_alpha158_single_feature_rank_ic"] = best_a
    (RES / "phase2_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def _best_alpha158(m, dates, member, fwd) -> dict:
    """Rank IC of each Alpha158 feature on the same discovery days (reference point)."""
    X = pd.read_parquet(ROOT / "data" / "processed" / "alpha158.parquet")
    d = X.index.get_level_values("date")
    X = X[(d >= dates[0]) & (d <= dates[-1])]
    out = {}
    for col in X.columns:
        w = X[col].unstack("code").reindex(index=dates, columns=m.member.columns).astype("float64")
        ic = ic_series(w, fwd, member, "spearman").dropna()
        out[col] = (ic.mean(), ic.mean() / ic.std() * np.sqrt(len(ic)))
    best = max(out, key=lambda k: abs(out[k][0]))
    top10 = sorted(out, key=lambda k: abs(out[k][0]), reverse=True)[:10]
    pd.Series({k: v[0] for k, v in out.items()}).rename("rank_ic").to_csv(RES / "phase2_alpha158_feature_ics.csv")
    return {"feature": best, "rank_ic": round(out[best][0], 4), "t": round(out[best][1], 2),
            "median_abs_rank_ic": round(float(np.median([abs(v[0]) for v in out.values()])), 4),
            "top10_abs": round(float(np.mean([abs(out[k][0]) for k in top10])), 4)}


if __name__ == "__main__":
    main()
