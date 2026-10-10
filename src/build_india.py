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
  * Corporate actions: on the ex-date of a split, bonus or rights issue NSE adjusts the
    stock's "previous close" in the bhavcopy. The ratio previous-close / actual last
    close therefore measures the adjustment, and earlier prices are scaled by it.
    Regular dividends are not adjusted (the Nifty price indices aren't either).
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


def adjustment_factor(close: pd.DataFrame, prevclose: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    last_close = close.ffill().shift(1)
    r = prevclose / last_close
    r = r.where(np.isfinite(r))
    event = (r - 1).abs() > 0.002
    r = r.where(event & (r > 0.01) & (r < 100), 1.0).fillna(1.0)
    n_events = int((r != 1.0).sum().sum())
    cum = r.iloc[::-1].cumprod().iloc[::-1]          # product over s >= t
    factor = cum.shift(-1).fillna(1.0)              # product over s > t
    return factor, n_events


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
            for f in ["close", "prevclose"]}
    factor, n_events = adjustment_factor(cont["close"], cont["prevclose"])
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
        "corporate_action_adjustments": n_events,
        "member_days_missing_close_pct": round(100 * float(1 - len(in_u) / max(int(member.values.sum()), 1)), 3),
        "member_days_locked_up_pct": round(100 * float(in_u["limit_up"].mean()), 3),
        "member_days_locked_down_pct": round(100 * float(in_u["limit_down"].mean()), 3),
        "member_days_abs_return_gt_25pct": int((in_u["ret_1d"].abs() > 0.25).sum()),
        "benchmark": bench,
        "benchmark_first_last": [round(float(index["level"].dropna().iloc[0]), 2),
                                 round(float(index["level"].dropna().iloc[-1]), 2)],
        "seconds": round(time.time() - t0),
    }
    (OUT / "data_quality.json").write_text(json.dumps(q, indent=2))
    print(json.dumps(q, indent=2))


if __name__ == "__main__":
    main()
