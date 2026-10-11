"""Build the Indian dataset from downloaded NSE files (run india_download.py first).

Output (data/processed/india/), in the same format as the China data so every later
step runs unchanged:
  prices.parquet      long panel (date, code): split/bonus-adjusted OHLCV + raw close + flags
  membership.parquet  boolean (date x code): in the point-in-time top-200 universe that day
  index.parquet       benchmark (Nifty 200, else Nifty 50) open/close
  data_quality.json   summary checks

Key choices
  * Universe: on the first trading day of each month, the 200 EQ-series stocks with the
    highest median daily traded value over the previous 63 trading days (needs >= 50
    trading days). Only past data is used, delisted stocks are kept, so there is no
    survivorship bias, and it needs no historical index-membership lists.
  * Only companies: ETFs and fund units also trade in the EQ series, so rows whose ISIN
    isn't a company ISIN (INE...) are dropped.
  * Splits and bonuses are found in the data itself (NSE's bhavcopy has no corporate-action
    column, and its "previous close" is NOT adjusted on ex-dates):
      - split / consolidation: the ISIN changes within a few days (India issues a new ISIN
        when the face value changes); the ratio is the nearest face-value ratio (1:2, 1:5, 1:10).
      - bonus (ISIN unchanged): the price falls to close to a standard bonus ratio (1/2, 1/3...).
      - other: any other one-day drop to 60% or less (demergers, capital reductions) is
        treated as return-neutral. Real crashes that big are rare in top-200 stocks; they are
        kept by listing them in config/india_overrides.csv.
    Every event is listed in corporate_actions.csv (check with india_diagnose.py).
    Dividends are not adjusted (the Nifty price indices aren't either).
  * Locked circuits: a stock that traded at a single price all day (high == low) after
    a move of 2% or more is treated as locked at its price band, so it can't be bought
    (locked up) or sold (locked down) at that day's open.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from india_parse import canonical_index, read_equity_file, read_index_file  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "india"
OUT = ROOT / "data" / "processed" / "india"
WINDOW, MIN_DAYS = 63, 50
LOCK_MOVE = 0.019


def load_equities() -> pd.DataFrame:
    files = sorted((RAW / "equities").glob("*/*.csv.zip"))
    if not files:
        raise SystemExit("No NSE files found. Run first:  python src/india_download.py")
    t, frames, bad = time.time(), [], []
    for i, f in enumerate(files, 1):
        try:
            frames.append(read_equity_file(f, all_series=True))
        except Exception as e:  # noqa: BLE001 - one bad file shouldn't stop the build
            bad.append(f"{f.name}: {e}")
        if i % 500 == 0:
            print(f"    read {i}/{len(files)} files ({time.time() - t:.0f}s)", flush=True)
    if bad:
        print(f"    skipped {len(bad)} unreadable files, e.g. {bad[0][:150]}", flush=True)
    df = pd.concat(frames, ignore_index=True).dropna(subset=["close"])
    # companies only: ETFs / fund units (INF... ISINs) also trade in the EQ series
    isin = df["isin"].fillna("").astype(str).str.upper()
    n_funds = int(((isin != "") & ~isin.str.startswith("INE")).sum())
    df = df[(isin == "") | isin.str.startswith("INE")]
    if n_funds:
        print(f"    dropped {n_funds:,} ETF / fund rows (non-company ISINs)", flush=True)
    # one row per stock per day; prefer the normal EQ row if a stock appears in two series
    df["_eq"] = (df["series"] == "EQ").astype(int)
    df = df.sort_values("_eq").drop_duplicates(["date", "symbol"], keep="last").drop(columns="_eq")
    return df


def universe(value: pd.DataFrame, n: int) -> pd.DataFrame:
    """Point-in-time membership: top-n by trailing median traded value, refreshed monthly."""
    med = value.where(value > 0).rolling(WINDOW, min_periods=MIN_DAYS).median()
    dates = value.index
    member = pd.DataFrame(False, index=dates, columns=value.columns)
    month = dates.to_period("M")
    for m in month.unique():
        in_month = month == m
        first = np.argmax(in_month)
        if first == 0:
            continue
        ranks = med.iloc[first - 1].dropna()   # data up to the previous trading day only
        if len(ranks) < n // 2:
            continue
        top = ranks.nlargest(n).index
        member.loc[in_month, top] = True
    return member


# shares after / shares before
SPLIT_RATIOS = [2.0, 2.5, 4.0, 5.0, 10.0, 20.0, 25.0, 50.0, 100.0, 0.5, 0.2, 0.1]   # face-value changes (+ combos)
BONUS_RATIOS = [2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 11.0, 20.0, 25.0, 50.0, 100.0]
ISIN_WINDOW = 10        # trading days: NSE files can show the new ISIN a few days before/after the ex-date
SPLIT_MATCH = 0.20      # log distance allowed when the ISIN changed (the price can move ~20% that day)
BONUS_MATCH = 0.06      # no ISIN change: 1:1 bonus (half price) needs a close match ...
BONUS_MATCH_BIG = 0.10  # ... looser for big ones (a real crash to a third or less is very rare)
CANDIDATE_MOVE = 0.60   # without an opening price: drops to 60% or lower count as corporate actions
SMALL_BONUS = [4 / 3, 1.4, 1.5]   # bonus 1:3, 2:5, 1:2: price to 75% / 71% / 67%
OPEN_RESET = 0.80       # opening 20%+ below the last close = the exchange reset the base price
OPEN_MATCH = 0.05       # an open this close to a bonus ratio is that bonus (overnight news moves it a little)
WATCH_MOVE = 0.78       # drops to 78% or lower are examined (and listed in corporate_actions.csv)
OVERRIDES = ROOT / "config" / "india_overrides.csv"


def load_overrides(path: Path | None = None) -> dict:
    """Manual decisions for specific days: symbol,date,action(keep|adjust),ratio,note."""
    path = path or OVERRIDES
    if not path.exists():
        return {}
    o = pd.read_csv(path, comment="#", dtype=str).fillna("")
    return {(r.symbol.strip(), pd.Timestamp(r.date)): (r.action.strip().lower(),
            float(r.ratio) if r.ratio.strip() else None) for r in o.itertuples()}


def _nearest(move: float, ratios: list[float]) -> tuple[float, float]:
    arr = np.array(ratios)
    i = int(np.argmin(np.abs(np.log(1 / move) - np.log(arr))))
    return float(arr[i]), float(abs(np.log(move * arr[i])))


def detect_actions(close: pd.DataFrame, volume: pd.DataFrame, value: pd.DataFrame,
                   isin: pd.DataFrame, overrides: dict | None = None,
                   open_: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Find splits, bonuses and other capital events; return (price factor, table of events).

    The key clue is the OPENING price. On an ex-date the exchange resets the stock's base price,
    so it opens far below yesterday's close (a 1:1 bonus opens near half). A real crash can't
    open that low: price bands and pre-open limits keep the open within ~20% (in the real data
    every big crash opened at -10% or less, then fell during the day).
      split   the ISIN changes within ISIN_WINDOW days -> nearest face-value ratio (1:2, 1:5, 1:10...)
      bonus   no ISIN change; opened 20%+ lower, right on a bonus ratio (1/2, 1/3, 2/3 ...)
      other   opened 20%+ lower but not on a ratio (demergers, rights, capital reductions): the
              price is rescaled at the open, so the day's own open->close move still counts
      kept    opened normally: a real price move, however big
    config/india_overrides.csv can force 'keep' or 'adjust' for any day. Without an opening
    price, a drop to 60% or less is treated as a corporate action (close-based rules)."""
    overrides = load_overrides() if overrides is None else overrides
    last = close.ffill().shift(1)
    move = close / last
    isin_f = isin.ffill()
    changed = (isin_f != isin_f.shift(1)) & isin_f.notna() & isin_f.shift(1).notna()
    near_change = changed.astype(float).rolling(2 * ISIN_WINDOW + 1, center=True, min_periods=1).max() > 0
    val_med = value.where(value > 0).rolling(20, min_periods=5).median().shift(1)
    val_ratio = value / val_med
    gap = (open_ / last).where(open_ > 0) if open_ is not None else move * np.nan
    big_move = (move <= WATCH_MOVE) | (gap <= OPEN_RESET)
    cand = move.notna() & (big_move | (near_change & (np.log(move).abs() > np.log(1.6))))
    cand = cand & (move > 0.001)
    for (sym, d) in overrides:
        if sym in close.columns and d in close.index and pd.notna(move.at[d, sym]):
            cand.at[d, sym] = True
    rows = []
    for i, j in np.argwhere(cand.to_numpy()):
        d, sym = cand.index[i], cand.columns[j]
        m = float(move.iat[i, j])
        g = float(gap.iat[i, j])
        if bool(near_change.iat[i, j]):
            k, dist = _nearest(m, SPLIT_RATIOS)
            kind, ok = "split", dist < SPLIT_MATCH
            if not ok and np.isfinite(g) and g <= OPEN_RESET:
                kind, k, ok = "other", 1.0 / g, True
        elif np.isfinite(g):
            if g <= OPEN_RESET:
                k, dist = _nearest(g, BONUS_RATIOS + SMALL_BONUS)
                if dist < OPEN_MATCH:
                    kind, ok = "bonus", True
                else:
                    kind, k, ok = "other", 1.0 / g, True
            else:
                k, _ = _nearest(m, BONUS_RATIOS + SMALL_BONUS)
                kind, ok = "kept", False
        else:  # no opening price: fall back on the close
            k, dist = _nearest(m, BONUS_RATIOS)
            kind, ok = "bonus", m <= CANDIDATE_MOVE and dist < (BONUS_MATCH if k < 3 else BONUS_MATCH_BIG)
            if not ok and m <= CANDIDATE_MOVE:
                kind, k, ok = "other", 1.0 / m, True
        note = ""
        ov = overrides.get((sym, d))
        if ov:
            action, ratio = ov
            if action == "keep":
                ok, note = False, "override: real price move"
            elif action == "adjust":
                k, ok, note = (ratio or k), True, "override: adjust"
        rows.append({"date": d, "symbol": sym, "kind": kind, "price_move": round(m, 4),
                     "open_gap": round(g, 4), "ratio": round(k, 6),
                     "match": round(abs(np.log(m * k)), 4), "value_x": round(float(val_ratio.iat[i, j]), 2),
                     "isin_changed_nearby": bool(near_change.iat[i, j]), "adjusted": bool(ok), "note": note})
    events = pd.DataFrame(rows, columns=["date", "symbol", "kind", "price_move", "open_gap", "ratio", "match", "value_x",
                                         "isin_changed_nearby", "adjusted", "note"])
    r = pd.DataFrame(1.0, index=close.index, columns=close.columns)
    for e in events[events["adjusted"]].itertuples():
        r.at[e.date, e.symbol] = 1.0 / e.ratio
    cum = r.iloc[::-1].cumprod().iloc[::-1]          # product over s >= t
    factor = cum.shift(-1).fillna(1.0)              # product over s > t: scales prices before the ex-date
    return factor, events.sort_values(["date", "symbol"]).reset_index(drop=True)


def load_index(dates: pd.DatetimeIndex) -> tuple[pd.DataFrame, str]:
    rows = []
    for f in sorted((RAW / "indices").glob("*/*.csv")):
        try:
            ix = read_index_file(f)
        except Exception:  # noqa: BLE001
            continue
        ix["canon"] = ix["name"].map(canonical_index)
        ix = ix.dropna(subset=["canon"]).drop_duplicates("canon")
        ix["date"] = pd.Timestamp(f.stem)
        rows.append(ix)
    if not rows:
        raise SystemExit("No index files found; run india_download.py again (it fetches them too).")
    allix = pd.concat(rows)
    wide = {k: allix.pivot(index="date", columns="canon", values=k).reindex(dates) for k in ["open", "close"]}
    cover = wide["close"].notna().mean()
    bench = "Nifty 200" if cover.get("Nifty 200", 0) >= 0.9 else "Nifty 50"
    out = pd.DataFrame({"open": wide["open"][bench], "close": wide["close"][bench]}, index=dates)
    out["close"] = out["close"].ffill()
    out["open"] = out["open"].where(out["open"] > 0, out["close"])
    out["level"] = out["close"]
    out["level_open"] = out["open"]
    if "Nifty 50" in wide["close"]:
        out["nifty50_close"] = wide["close"]["Nifty 50"]
    out.index.name = "date"
    return out, bench


def _stack(df: pd.DataFrame) -> pd.Series:
    """Wide (date x code) -> long Series, keeping NaN (works on pandas 2.x and 3.x)."""
    try:
        return df.stack(future_stack=True)
    except TypeError:  # pandas < 2.1
        return df.stack(dropna=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", type=int, default=200)
    ap.add_argument("--start", default="2014-01-01")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    long = load_equities()
    long = long[long["date"] >= pd.Timestamp(a.start) - pd.Timedelta(days=120)]
    dates = pd.DatetimeIndex(sorted(long["date"].unique()))
    print(f"    {len(dates)} trading days, {long['symbol'].nunique()} EQ stocks "
          f"({dates[0].date()} to {dates[-1].date()})", flush=True)

    eq = long[long["series"] == "EQ"]
    value = eq.pivot(index="date", columns="symbol", values="value").reindex(dates)
    member = universe(value, a.universe)
    keep = dates >= pd.Timestamp(a.start)
    member = member.loc[keep]
    codes = member.columns[member.any()]
    member = member[codes]
    sub_all = long[long["symbol"].isin(codes)]
    sub = sub_all[sub_all["series"] == "EQ"]
    w = {f: sub.pivot(index="date", columns="symbol", values=f).reindex(index=dates, columns=codes)
         for f in ["open", "high", "low", "close", "prevclose", "volume", "value"]}
    # corporate actions are measured on closes from any equity series (a stock can spend days in
    # the trade-to-trade BE series), but only EQ days are treated as tradable
    cont = {f: sub_all.pivot(index="date", columns="symbol", values=f).reindex(index=dates, columns=codes)
            for f in ["open", "close", "volume", "value", "isin"]}
    factor, events = detect_actions(cont["close"], cont["volume"], cont["value"], cont["isin"],
                                    open_=cont["open"])
    events = events[events["date"] >= pd.Timestamp(a.start)].reset_index(drop=True)
    events.to_csv(OUT / "corporate_actions.csv", index=False)
    done = events[events["adjusted"]]
    print(f"    corporate actions: {int((done['kind'] == 'split').sum())} splits (ISIN changed), "
          f"{int((done['kind'] == 'bonus').sum())} bonuses, {int((done['kind'] == 'other').sum())} other "
          f"(demergers etc.); {int((~events['adjusted']).sum())} kept as real moves (see corporate_actions.csv)",
          flush=True)
    raw_vwap = (w["value"] / w["volume"]).where(w["volume"] > 0)
    adj = {
        "open": w["open"] * factor, "high": w["high"] * factor, "low": w["low"] * factor,
        "close": w["close"] * factor, "vwap": raw_vwap * factor,
        "volume": w["volume"] / factor, "amount": w["value"],
        "factor": factor, "raw_close": w["close"],
    }
    adj["adjclose"] = adj["close"]
    adj["ret_1d"] = adj["close"] / adj["close"].ffill().shift(1) - 1
    gap_close = adj["ret_1d"]
    locked = (w["high"] == w["low"]) & w["high"].notna()
    adj["suspended"] = w["close"].isna() | (w["volume"].fillna(0) <= 0)
    adj["limit_up"] = locked & (gap_close >= LOCK_MOVE)
    adj["limit_down"] = locked & (gap_close <= -LOCK_MOVE)

    adj = {k: v.loc[keep] for k, v in adj.items()}
    member.index.name = "date"
    px = pd.DataFrame({k: _stack(v) for k, v in adj.items()})
    px.index.names = ["date", "code"]
    px = px[px["close"].notna()]  # rows where the stock traded
    px["in_universe"] = _stack(member).reindex(px.index).fillna(False).astype(bool)
    px.to_parquet(OUT / "prices.parquet")
    member.to_parquet(OUT / "membership.parquet")

    index, bench = load_index(member.index)
    index.to_parquet(OUT / "index.parquet")

    in_u = px[px["in_universe"]]
    per_day = in_u.groupby(level="date").size()
    first_member = member.index[member.any(axis=1)][0]
    q = {
        "market": "india", "source": "NSE bhavcopy (EQ series)",
        "start": str(member.index[0].date()), "end": str(member.index[-1].date()),
        "first_universe_day": str(first_member.date()),
        "trading_days": int(len(member)),
        "unique_stocks_ever_in_universe": int(len(codes)),
        "members_per_day_min_median_max": [int(per_day.min()), int(per_day.median()), int(per_day.max())],
        "splits_adjusted": int((done["kind"] == "split").sum()),
        "bonuses_adjusted": int((done["kind"] == "bonus").sum()),
        "other_adjusted": int((done["kind"] == "other").sum()),
        "corporate_action_adjustments": int(len(done)),
        "member_days_missing_close_pct": round(100 * float(1 - len(in_u) / max(int(member.values.sum()), 1)), 3),
        "member_days_locked_up_pct": round(100 * float(in_u["limit_up"].mean()), 3),
        "member_days_locked_down_pct": round(100 * float(in_u["limit_down"].mean()), 3),
        "member_days_abs_return_gt_25pct": int((in_u["ret_1d"].abs() > 0.25).sum()),
        "biggest_member_day_moves": [
            f"{d.date()} {c} {r:+.1%}" for (d, c), r in
            in_u["ret_1d"].dropna().pipe(lambda x: x.reindex(x.abs().sort_values(ascending=False).index)).head(10).items()],
        "benchmark": bench,
        "benchmark_first_last": [round(float(index["level"].dropna().iloc[0]), 2),
                                 round(float(index["level"].dropna().iloc[-1]), 2)],
        "seconds": round(time.time() - t0),
    }
    (OUT / "data_quality.json").write_text(json.dumps(q, indent=2))
    print(json.dumps(q, indent=2))


if __name__ == "__main__":
    main()
