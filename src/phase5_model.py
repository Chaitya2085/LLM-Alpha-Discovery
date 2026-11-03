"""Phase 5: the prediction model. Combines the factors that work into one weekly ranking, with a
confidence score that means what it says.

    python phase5_model.py                      # India (default), test years 2017-2020
    python phase5_model.py --market china

What it predicts: for every stock in the universe, each week, the probability that it beats
the benchmark index over the next 5 trading days (buy at the next open, sell 5 days later).

How:
  * Features: the 158 standard Alpha158 features, the LLM factors that passed (Phase 2
    shortlist + Phase 4 'works after costs'), and the market regime (index trend, volatility,
    how spread out stocks are). Every feature is ranked across stocks each day.
  * Model: LightGBM classifier learning which stocks beat the median stock that week (a ranking
    task that doesn't depend on whether the market rose), walk-forward: for each test year it
    trains on all earlier years, early-stops on the year before (by ranking accuracy, AUC), and
    never sees the test year.
  * Confidence: each week's model scores are turned into ranks, and the rank is calibrated on
    the previous year (a two-number logistic fit) into the probability of beating the index, so
    "60%" should come true about 60% of the time. Calibration is checked on the test years.
  * Two versions are trained, Alpha158 only and Alpha158 + LLM factors, so we can see whether
    the LLM factors add anything.

How it's judged (weekly, after Indian costs, vs the Nifty 200):
  * Top 20: each week hold the 20 stocks with the highest confidence, equal weight (a stock
    already held stays while it's in the best 60, to save trading costs)
  * Confident only: hold only stocks with confidence >= --min-prob (up to 20); the rest of the
    money sits in the index. This is "abstain when unsure".
  * Weekly hit rate, return after costs, information ratio, drawdown, calibration (reliability
    table, Brier score), hit rate of high-confidence calls, results by market regime.

Held-out years: nothing after 2020 is loaded at all (the data is cut at --cutoff before anything
is computed), so the Phase 6 final test on 2021 onward stays untouched.

Writes results/<market>/phase5_summary.json, phase5_weekly.csv, phase5_calibration.csv,
phase5_regimes.csv, phase5_importance.csv, phase5_predictions.parquet.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

import markets

ROOT = Path(__file__).resolve().parents[1]
CUTOFF = "2020-12-31"
TRAIN_START = "2014-04-01"
HOLD = 5
WEEKS = 52
PARAMS = dict(objective="binary", metric="auc", learning_rate=0.05, num_leaves=31, max_depth=6, min_data_in_leaf=300,
              feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0,
              verbose=-1, seed=7, num_threads=2)
PROB_BINS = [0.0, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 1.0]


# ---------------------------------------------------------------- data

def load_cut(cfg, cutoff: str):
    """Load the market and drop every day after `cutoff`, before anything is computed."""
    from data import load_market
    m = load_market(cfg.name)
    keep = m.dates <= pd.Timestamp(cutoff)
    fields = {}
    for f in dataclasses.fields(m):
        v = getattr(m, f.name)
        if isinstance(v, (pd.DataFrame, pd.Series)):
            v = v.loc[keep]
        fields[f.name] = v
    m = dataclasses.replace(m, **fields)
    # forward returns must be recomputed on the cut data so the last days have no label
    o = m.open
    m = dataclasses.replace(m, fwd_ret=o.shift(-2) / o.shift(-1) - 1)
    return m


def labels(m) -> tuple[pd.DataFrame, pd.Series]:
    """5-day forward return of each stock (open t+1 -> open t+6) and of the index, same window."""
    o = m.open
    fwd5 = o.shift(-(HOLD + 1)) / o.shift(-1) - 1
    li = np.log1p(m.index_fwd_ret)
    bench5 = np.expm1(li[::-1].rolling(HOLD, min_periods=HOLD).sum()[::-1])
    return fwd5, bench5


def targets(m):
    """Everything the model learns from and is judged on, from one market:
    fwd5/bench5 (5-day returns), y (beats the index), y_rank (beats the median stock),
    yex_long (excess over the index) and label_end (the day each label is known)."""
    fwd5, bench5 = labels(m)
    yex = (fwd5.sub(bench5, axis=0)).where(m.member)
    y = _long((yex > 0).astype(float).where(yex.notna())).rename("y")
    # the model learns to rank: "beats the median stock this week" (always ~50%, whatever the market
    # does); its scores are then calibrated to the question users care about, "beats the index"
    rel = fwd5.where(m.member)
    rel = rel.sub(rel.median(axis=1), axis=0)
    y_rank = _long((rel > 0).astype(float).where(rel.notna())).rename("y_rank")
    yex_long = _long(yex).rename("yex")
    pos = pd.Series(np.arange(len(m.dates)), index=m.dates)
    end_pos = (pos + HOLD + 1).clip(upper=len(m.dates) - 1)
    label_end = pd.Series(m.dates[end_pos.to_numpy()], index=m.dates)
    label_end[pos + HOLD + 1 >= len(m.dates)] = pd.Timestamp("2262-01-01")   # no label: never used
    return fwd5, bench5, y, y_rank, yex_long, label_end


def regimes(m) -> pd.DataFrame:
    """Market-wide state each day, from past data only."""
    lvl = m.index_close.ffill()
    r = lvl.pct_change()
    vol20 = r.rolling(20).std() * np.sqrt(252)
    vol_pct = vol20.expanding(min_periods=120).rank(pct=True)
    ma200 = lvl.rolling(200, min_periods=150).mean()
    stock_r20 = m.close / m.close.shift(20) - 1
    disp = stock_r20.where(m.member).std(axis=1)
    out = pd.DataFrame({
        "mkt_ret20": lvl / lvl.shift(20) - 1, "mkt_ret60": lvl / lvl.shift(60) - 1,
        "mkt_vol20": vol20, "mkt_vol_pct": vol_pct, "mkt_above_ma200": (lvl > ma200).astype(float),
        "mkt_dispersion": disp,
    })
    label = np.where(vol_pct > 0.8, "turbulent", np.where(lvl > ma200, "calm, rising", "calm, falling"))
    out["regime"] = pd.Series(label, index=lvl.index).where(ma200.notna() & vol_pct.notna())
    return out


def select_llm_factors(cfg) -> list[dict]:
    """Factors that passed: Phase 2 shortlist and Phase 4 'works after costs' (deduplicated)."""
    from generate_factors import load_library
    lib = {r["id"]: r for r in load_library() if r.get("status") == "valid"}
    chosen: dict[str, dict] = {}
    p2 = cfg.results / "phase2_factor_scores.csv"
    if p2.exists():
        s = pd.read_csv(p2)
        s = s[s["shortlist"].fillna(False).astype(bool)] if "shortlist" in s.columns else s.iloc[0:0]
        for r in s.itertuples():
            if r.id in lib:
                chosen[lib[r.id]["expression"]] = {"id": r.id, "name": r.name, "source": "phase 2 shortlist",
                                                   "expression": lib[r.id]["expression"]}
    p4 = cfg.results / "phase4_factors.csv"
    if p4.exists():
        try:
            f = pd.read_csv(p4)
        except pd.errors.EmptyDataError:
            f = pd.DataFrame()
        if len(f):
            for r in f[f["verdict"].isin(["keep", "star"])].itertuples():
                if r.id in lib:
                    chosen.setdefault(lib[r.id]["expression"], {"id": r.id, "name": r.name,
                                                                "source": "phase 4 works after costs",
                                                                "expression": lib[r.id]["expression"]})
    return list(chosen.values())


def _long(df: pd.DataFrame) -> pd.Series:
    """Wide (date x code) -> long Series indexed (date, code), NaN dropped."""
    out = df.stack()
    out.index = out.index.set_names(["date", "code"])
    return out


def cs_rank(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(level="date").rank(pct=True) - 0.5


def build_features(m, cfg, factors: list[dict], use_a158: bool = True) -> pd.DataFrame:
    """Long (date, code) table of cross-sectionally ranked features for universe members."""
    from dsl import Evaluator
    member_idx = _long(m.member)
    member_idx = member_idx[member_idx.astype(bool)].index
    parts = []
    a158 = cfg.proc / "alpha158.parquet"
    if use_a158 and a158.exists():
        try:
            X = pd.read_parquet(a158, filters=[("date", "<=", m.dates[-1])])
        except Exception:  # noqa: BLE001 - older pyarrow
            X = pd.read_parquet(a158)
            X = X[X.index.get_level_values("date") <= m.dates[-1]]
        X = X.reindex(member_idx).astype("float32")
        parts.append(cs_rank(X).astype("float32").add_prefix("a158_"))
    if factors:
        ev = Evaluator(m)
        cols = {}
        for f in factors:
            try:
                w = ev(f["expression"])
            except Exception as e:  # noqa: BLE001
                print(f"    skipped {f['name']}: {e}")
                continue
            cols[f"llm_{f['id']}"] = _long(w.where(m.member)).reindex(member_idx)
            ev.clear_cache()
        if cols:
            L = pd.DataFrame(cols, index=member_idx).astype("float32")
            parts.append(cs_rank(L).astype("float32"))
    if not parts:
        raise SystemExit("no features: build Alpha158 first or pass factors")
    X = pd.concat(parts, axis=1)
    reg = regimes(m).drop(columns="regime")
    X = X.join(reg.astype("float32"), on="date")
    return X


# ---------------------------------------------------------------- calibration (no sklearn needed)

def platt_fit(x: np.ndarray, y: np.ndarray, iters: int = 50):
    """Logistic fit p = 1 / (1 + exp(-(a + b*x))) by Newton's method; returns the mapping.
    Two numbers only, so a year of data can't overfit it."""
    X = np.column_stack([np.ones_like(x), x])
    w = np.zeros(2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (y - p)
        H = (X * (p * (1 - p))[:, None]).T @ X + 1e-9 * np.eye(2)
        step = np.linalg.solve(H, g)
        w += step
        if np.abs(step).max() < 1e-8:
            break
    return lambda s: 1 / (1 + np.exp(-(w[0] + w[1] * s)))


def week_pct(scores: np.ndarray, dates: pd.Index) -> np.ndarray:
    """Rank of each score among the stocks on the same day, centred: -0.5 (worst) .. +0.5 (best)."""
    return (pd.Series(scores, index=dates).groupby(level=0).rank(pct=True) - 0.5).to_numpy()

def isotonic_fit(x: np.ndarray, y: np.ndarray):
    """Pool-adjacent-violators; returns a function mapping scores to calibrated probabilities."""
    order = np.argsort(x)
    xs, ys = x[order], y[order].astype(float)
    # pre-bin to at most 200 blocks so the fit is fast and smooth
    nb = min(200, len(xs))
    edges = np.linspace(0, len(xs), nb + 1).astype(int)
    bx = np.array([xs[a:b].mean() for a, b in zip(edges[:-1], edges[1:]) if b > a])
    by = np.array([ys[a:b].mean() for a, b in zip(edges[:-1], edges[1:]) if b > a])
    bw = np.array([b - a for a, b in zip(edges[:-1], edges[1:]) if b > a], float)
    vals, wts, xs_out = [], [], []
    for xv, yv, wv in zip(bx, by, bw):
        vals.append(yv), wts.append(wv), xs_out.append([xv])
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2, x2 = vals.pop(), wts.pop(), xs_out.pop()
            v1, w1, x1 = vals.pop(), wts.pop(), xs_out.pop()
            vals.append((v1 * w1 + v2 * w2) / (w1 + w2)), wts.append(w1 + w2), xs_out.append(x1 + x2)
    cx = np.array([np.mean(g) for g in xs_out])
    cy = np.array(vals)
    return lambda s: np.interp(s, cx, cy)


# ---------------------------------------------------------------- walk-forward

def walk_forward(X: pd.DataFrame, y: pd.Series, label_end: pd.Series, years: list[int],
                 threads: int = 2, log=print, y_train: pd.Series | None = None
                 ) -> tuple[pd.DataFrame, dict, lgb.Booster | None]:
    """Train on y_train (default y), calibrate the scores to y on the validation year."""
    d = X.index.get_level_values("date")
    le = label_end.reindex(d).to_numpy()
    yv = y.reindex(X.index)
    yt = (y_train if y_train is not None else y).reindex(X.index)
    ok = (yv.notna() & yt.notna()).to_numpy()
    params = {**PARAMS, "num_threads": threads}
    preds, info, last_model = [], {}, None
    for Y in years:
        v0, t0, t1 = pd.Timestamp(f"{Y - 1}-01-01"), pd.Timestamp(f"{Y}-01-01"), pd.Timestamp(f"{Y + 1}-01-01")
        tr = ok & (d >= pd.Timestamp(TRAIN_START)) & (le < v0)          # whole label window before validation
        va = ok & (d >= v0) & (le < t0)                                  # ... and before the test year
        te = ok & (d >= t0) & (d < t1)
        if tr.sum() < 5000 or va.sum() < 1000 or te.sum() == 0:
            log(f"    {Y}: not enough data (train {tr.sum()}, valid {va.sum()}), skipped")
            continue
        t = time.time()
        model = lgb.train(params, lgb.Dataset(X.values[tr], yt.values[tr].astype(float)), num_boost_round=800,
                          valid_sets=[lgb.Dataset(X.values[va], yt.values[va].astype(float))],
                          callbacks=[lgb.early_stopping(50, verbose=False)])
        raw_va = model.predict(X.values[va], num_iteration=model.best_iteration)
        cal = platt_fit(week_pct(raw_va, d[va]), yv.values[va].astype(float))
        raw_te = model.predict(X.values[te], num_iteration=model.best_iteration)
        pct_te = week_pct(raw_te, d[te])
        preds.append(pd.DataFrame({"p_raw": raw_te, "rank": pct_te, "p": cal(pct_te)}, index=X.index[te]))
        info[Y] = {"train_rows": int(tr.sum()), "valid_rows": int(va.sum()), "test_rows": int(te.sum()),
                   "best_iter": int(model.best_iteration), "secs": round(time.time() - t, 1)}
        log(f"    {Y}: {info[Y]}")
        last_model = model
    if not preds:
        raise SystemExit("walk-forward produced no predictions (not enough history)")
    return pd.concat(preds), info, last_model


# ---------------------------------------------------------------- judging

def rebalance_days(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return dates[::HOLD]


def portfolio(pred: pd.DataFrame, m, fwd5: pd.DataFrame, bench5: pd.Series, top: int,
              min_prob: float | None, buffer: int | None = None, every: int = 1) -> pd.DataFrame:
    """Weekly book: the `top` most confident buyable stocks (only those >= min_prob if given);
    unused slots sit in the index. A stock already held is kept while it stays in the best
    `buffer` (default 3 x top), which cuts trading a lot. Trades happen every `every` weeks
    (in between, the book is simply held). Costs on the stocks that change."""
    buffer = buffer or 3 * top
    p = pred["p"].unstack("code")
    days = rebalance_days(p.index)
    pos = {d: i for i, d in enumerate(m.dates)}
    rows, held, k = [], set(), -1
    for d in days:
        i = pos[d]
        if i + HOLD + 1 >= len(m.dates) or pd.isna(bench5.get(d)):
            continue
        k += 1
        if held and k % every:
            picks = sorted(held)
            r = fwd5.loc[d].reindex(picks).fillna(0.0)
            gross = (r.sum() / top) + (1 - len(picks) / top) * bench5[d]
            univ = fwd5.loc[d].where(m.member.loc[d]).mean()
            rows.append({"date": d, "gross": gross, "cost": 0.0, "net": gross, "bench": bench5[d],
                         "universe": univ, "n_stocks": len(picks), "turnover": 0.0})
            continue
        buyable = m.can_buy.iloc[i + 1]
        allp = p.loc[d].dropna().sort_values(ascending=False)
        keep_zone = set(allp.index[:buffer])
        s = allp[buyable.reindex(allp.index).fillna(False).astype(bool)]
        if min_prob is not None:
            s = s[s >= min_prob]
            keep_zone &= set(allp[allp >= min_prob].index)
        kept = [c for c in held if c in keep_zone]
        fresh = [c for c in s.index if c not in set(kept)]
        picks = (kept + fresh)[:top]
        r = fwd5.loc[d].reindex(picks).fillna(0.0)
        n_buy, n_sell = len(set(picks) - held), len(held - set(picks))
        cost = (n_buy * m.buy_cost + n_sell * m.sell_cost) / top
        w_stocks = len(picks) / top
        gross = (r.sum() / top) + (1 - w_stocks) * bench5[d]
        univ = fwd5.loc[d].where(m.member.loc[d]).mean()
        rows.append({"date": d, "gross": gross, "cost": cost, "net": gross - cost, "bench": bench5[d],
                     "universe": univ, "n_stocks": len(picks), "turnover": n_buy / top})
        held = set(picks)
    return pd.DataFrame(rows).set_index("date")


def book_stats(w: pd.DataFrame) -> dict:
    ex = w["net"] - w["bench"]
    eq = (1 + w["net"]).cumprod()
    years = len(w) / WEEKS
    return {
        "weeks": int(len(w)),
        "ann_return": float(eq.iloc[-1] ** (1 / years) - 1),
        "bench_ann_return": float((1 + w["bench"]).prod() ** (1 / years) - 1),
        "excess_ann": float(ex.mean() * WEEKS),
        "gross_excess_ann": float((w["gross"] - w["bench"]).mean() * WEEKS),
        "info_ratio": float(ex.mean() / ex.std() * np.sqrt(WEEKS)) if ex.std() > 0 else float("nan"),
        "weekly_hit_rate": float((ex > 0).mean()),
        "max_drawdown": float((eq / eq.cummax() - 1).min()),
        "avg_turnover": float(w["turnover"].iloc[1:].mean()),
        "cost_ann": float(w["cost"].iloc[1:].mean() * WEEKS),
        "avg_stocks": float(w["n_stocks"].mean()),
        "weeks_invested_pct": float((w["n_stocks"] > 0).mean()),
    }


GRID = [(top, keep, every) for top in (20, 40) for keep in (3, 5) for every in (1, 2)]


def design_grid(pred, m, fwd5, bench5) -> pd.DataFrame:
    """A small, fixed set of ways to trade the same predictions. All are reported; the one with
    the best information ratio after costs is carried into Phase 6 (and the number tried is
    recorded, so Phase 6 can discount for it)."""
    rows = []
    for top, keep, every in GRID:
        st = book_stats(portfolio(pred, m, fwd5, bench5, top, None, keep * top, every))
        rows.append({"stocks": top, "keep_zone": keep * top, "trade_every_weeks": every, **st})
    return pd.DataFrame(rows)


def calibration(pred: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, dict]:
    days = rebalance_days(pred.index.get_level_values("date").unique())
    pr = pred[pred.index.get_level_values("date").isin(days)].join(y.rename("y"), how="inner").dropna()
    pr["bin"] = pd.cut(pr["p"], PROB_BINS, include_lowest=True)
    t = pr.groupby("bin", observed=True).agg(n=("y", "size"), predicted=("p", "mean"), actual=("y", "mean"))
    t = t[t["n"] > 0]
    base = pr["y"].mean()
    brier = float(((pr["p"] - pr["y"]) ** 2).mean())
    brier0 = float(((base - pr["y"]) ** 2).mean())
    ece = float((t["n"] * (t["predicted"] - t["actual"]).abs()).sum() / t["n"].sum())
    hi = pr[pr["p"] >= 0.6]
    stats = {"stock_weeks": int(len(pr)), "base_rate": float(base), "brier": brier, "brier_no_skill": brier0,
             "brier_skill": float(1 - brier / brier0) if brier0 > 0 else float("nan"), "ece": ece,
             "high_conf_n": int(len(hi)), "high_conf_hit_rate": float(hi["y"].mean()) if len(hi) else float("nan"),
             "high_conf_share": float(len(hi) / len(pr)) if len(pr) else float("nan")}
    t = t.reset_index()
    t["bin"] = t["bin"].astype(str)
    return t, stats


def weekly_ic(pred: pd.DataFrame, yex: pd.Series) -> dict:
    days = rebalance_days(pred.index.get_level_values("date").unique())
    df = pred[pred.index.get_level_values("date").isin(days)].join(yex.rename("y"), how="inner").dropna()
    with np.errstate(all="ignore"):
        ic = df.groupby(level="date").apply(lambda g: g["p"].rank().corr(g["y"].rank()))
    ic = ic.dropna()
    return {"weekly_rank_ic": float(ic.mean()), "weekly_rank_ic_t": float(ic.mean() / ic.std() * np.sqrt(len(ic)))
            if len(ic) > 2 and ic.std() > 0 else float("nan")}


def by_regime(w: pd.DataFrame, reg: pd.Series) -> pd.DataFrame:
    df = w.join(reg.rename("regime"))
    df["excess"] = df["net"] - df["bench"]
    g = df.dropna(subset=["regime"]).groupby("regime")
    return pd.DataFrame({"weeks": g.size(), "hit_rate": g["excess"].apply(lambda e: (e > 0).mean()),
                         "excess_ann": g["excess"].mean() * WEEKS}).reset_index()


# ---------------------------------------------------------------- main

def run(cfg, years: list[int], top: int = 20, min_prob: float = 0.50, cutoff: str = CUTOFF,
        threads: int = 2, use_a158: bool = True, log=print) -> dict:
    t0 = time.time()
    m = load_cut(cfg, cutoff)
    assert m.dates[-1] <= pd.Timestamp(cutoff)
    fwd5, bench5, y, y_rank, yex_long, label_end = targets(m)
    reg = regimes(m)
    factors = select_llm_factors(cfg)
    log(f"  {len(factors)} LLM factors passed earlier phases; building features ...")
    X_all = build_features(m, cfg, factors, use_a158)
    has_a158 = any(c.startswith("a158_") for c in X_all.columns)
    has_llm = any(c.startswith("llm_") for c in X_all.columns)
    if has_a158 and has_llm:
        variants = {"Alpha158 only": X_all[[c for c in X_all.columns if not c.startswith("llm_")]],
                    "Alpha158 + LLM factors": X_all}
    elif has_a158:
        variants = {"Alpha158 only": X_all}
    else:
        variants = {"LLM factors only": X_all}
    res = cfg.results
    res.mkdir(parents=True, exist_ok=True)
    out = {"market": cfg.name, "cutoff": cutoff, "test_years": years, "top": top, "min_prob": min_prob,
           "llm_factors": factors, "variants": {}}
    weekly_rows, cal_rows, reg_rows, grid_rows, preds_keep = [], [], [], [], None
    for name, X in variants.items():
        log(f"  training '{name}' ({X.shape[1]} features, {len(X):,} rows) ...")
        pred, info, model = walk_forward(X, y, label_end, years, threads, log, y_train=y_rank)
        books = {"Top 20": portfolio(pred, m, fwd5, bench5, top, None),
                 f"Confident only (>= {min_prob:.0%})": portfolio(pred, m, fwd5, bench5, top, min_prob)}
        cal_t, cal_s = calibration(pred, y)
        v = {"training": info, **weekly_ic(pred, yex_long), "calibration": cal_s, "books": {}}
        g = design_grid(pred, m, fwd5, bench5)
        best = g.sort_values(["info_ratio", "avg_turnover"], ascending=[False, True]).iloc[0]
        v["design_grid_tried"] = int(len(g))
        v["chosen_design"] = {"stocks": int(best["stocks"]), "keep_zone": int(best["keep_zone"]),
                              "trade_every_weeks": int(best["trade_every_weeks"])}
        chosen_name = (f"Chosen: {int(best['stocks'])} stocks, keep top {int(best['keep_zone'])}, "
                       f"trade every {int(best['trade_every_weeks'])} wk")
        if (int(best["stocks"]), int(best["keep_zone"]), int(best["trade_every_weeks"])) != (top, 3 * top, 1):
            books[chosen_name] = portfolio(pred, m, fwd5, bench5, int(best["stocks"]), None,
                                           int(best["keep_zone"]), int(best["trade_every_weeks"]))
        grid_rows.append(g.assign(variant=name))
        for bname, w in books.items():
            v["books"][bname] = book_stats(w)
            weekly_rows.append(w.assign(variant=name, book=bname).reset_index())
            reg_rows.append(by_regime(w, reg["regime"]).assign(variant=name, book=bname))
        cal_rows.append(cal_t.assign(variant=name))
        if model is not None:
            imp = pd.Series(model.feature_importance("gain"), index=X.columns).sort_values(ascending=False)
            share = float(imp[[c for c in imp.index if c.startswith("llm_")]].sum() / imp.sum()) if imp.sum() else 0.0
            v["llm_importance_share"] = share
            v["top_features"] = [{"feature": k, "gain_share": round(float(g / imp.sum()), 4)}
                                 for k, g in imp.head(15).items()]
            if has_llm and "LLM" in name:
                imp.rename("gain").to_csv(res / "phase5_importance.csv")
        out["variants"][name] = v
        preds_keep = pred
        log(f"    {name}: weekly rank IC {v['weekly_rank_ic']:.3f}, Top 20 hit rate "
            f"{v['books']['Top 20']['weekly_hit_rate']:.0%}, excess {v['books']['Top 20']['excess_ann']:+.1%}/yr, "
            f"calibration error {cal_s['ece']:.3f}")
    pd.concat(weekly_rows).to_csv(res / "phase5_weekly.csv", index=False)
    pd.concat(cal_rows).to_csv(res / "phase5_calibration.csv", index=False)
    pd.concat(reg_rows).to_csv(res / "phase5_regimes.csv", index=False)
    pd.concat(grid_rows).to_csv(res / "phase5_grid.csv", index=False)
    days = rebalance_days(preds_keep.index.get_level_values("date").unique())
    preds_keep[preds_keep.index.get_level_values("date").isin(days)].to_parquet(res / "phase5_predictions.parquet")
    out["seconds"] = round(time.time() - t0)
    (res / "phase5_summary.json").write_text(json.dumps(out, indent=2, default=str))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--market", default=os.environ.get("LLM_ALPHA_MARKET", "india"))
    ap.add_argument("--years", default="2017-2020", help="test years, e.g. 2017-2020 (must end by the cutoff)")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--min-prob", type=float, default=0.50,
                    help="'confident only' book: hold stocks at least this likely to beat the index")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    cfg = markets.get(a.market)
    os.environ["LLM_ALPHA_MARKET"] = cfg.name
    y0, y1 = (int(x) for x in a.years.split("-"))
    if y1 > int(CUTOFF[:4]):
        raise SystemExit(f"test years must end by {CUTOFF[:4]}: later years are the Phase 6 final test")
    s = run(cfg, list(range(y0, y1 + 1)), a.top, a.min_prob, CUTOFF, a.threads)
    for name, v in s["variants"].items():
        b = v["books"]["Top 20"]
        print(f"  {name}: Top 20 {b['ann_return']:+.1%}/yr vs index {b['bench_ann_return']:+.1%}; "
              f"hit rate {b['weekly_hit_rate']:.0%}; high-confidence calls right "
              f"{v['calibration']['high_conf_hit_rate']:.0%}")
    print(f"  done in {s['seconds']}s")


if __name__ == "__main__":
    main()
