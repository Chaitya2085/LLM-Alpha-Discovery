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
      - split / consolidation: the stock's ISIN changes (India issues a new ISIN when the
        face value changes) and the price jumps; the ratio is the nearest standard ratio
        (1:2, 1:5, 1:10, ...).
      - bonus (ISIN unchanged): the price falls to almost exactly a standard ratio (2/3, 1/2,
        1/3, ...) while traded value in rupees stays normal and share volume rises to match.
        A real crash comes with a burst of traded value, so it is left alone.
    Only information up to the ex-date is used. Every adjustment is listed in
    corporate_actions.csv. Dividends are not adjusted (the Nifty price indices aren't either).
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


# standard ratios: shares after / shares before (splits, bonuses, both, and consolidations)
NICE = sorted({1.25, 4 / 3, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 50.0, 100.0,
               0.5, 0.25, 0.2, 0.1})
SPLIT_MATCH = 0.15     # log distance allowed between the price move and the standard ratio (ISIN changed)
BONUS_MATCH = 0.06     # tighter when the ISIN didn't change (bonus 1:2 or 1:1) ...
BONUS_MATCH_BIG = 0.10  # ... looser for big bonuses (price to a third or less: real crashes that size are rare)
BONUS_MAX_MOVE = 0.72  # bonus candidates: price at most 72% of the last close (bonus 1:2 or bigger)
BONUS_MAX_VALUE = 3.0  # a real crash trades far more rupees than usual; a bonus doesn't


def detect_actions(close: pd.DataFrame, volume: pd.DataFrame, value: pd.DataFrame,
                   isin: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Find splits/bonuses; return (price factor to multiply prices by, table of events)."""
    last = close.ffill().shift(1)
    move = close / last
    isin_last = isin.ffill().shift(1)
    new_isin = isin.notna() & isin_last.notna() & (isin != isin_last) & close.notna()
    vol_med = volume.where(volume > 0).rolling(20, min_periods=5).median().shift(1)
    val_med = value.where(value > 0).rolling(20, min_periods=5).median().shift(1)
    vol_ratio, val_ratio = volume / vol_med, value / val_med
    nice = np.array(NICE)

    k_split = (move.notna() & new_isin & (np.log(move).abs() > np.log(1.25)))
    k_bonus = (move.notna() & ~new_isin & (move <= BONUS_MAX_MOVE) & (move > 0.005))
    rows = []
    for kind, mask in (("split", k_split), ("bonus", k_bonus)):
        for i, j in np.argwhere(mask.to_numpy()):
            d, sym = mask.index[i], mask.columns[j]
            m = float(move.at[d, sym])
            k = float(nice[np.argmin(np.abs(np.log(1 / m) - np.log(nice)))])   # shares after / before
            dist = abs(np.log(m * k))
            vr, va = float(vol_ratio.at[d, sym]), float(val_ratio.at[d, sym])
            if kind == "split":
                ok = dist < SPLIT_MATCH
            else:
                tol = BONUS_MATCH if k < 3 else BONUS_MATCH_BIG
                ok = (k > 1 and dist < tol and np.isfinite(va) and va <= BONUS_MAX_VALUE
                      and np.isfinite(vr) and vr >= 0.5 * k)
            rows.append({"date": d, "symbol": sym, "kind": kind, "price_move": round(m, 4), "ratio": k,
                         "match": round(dist, 4), "volume_x": round(vr, 2), "value_x": round(va, 2),
                         "adjusted": bool(ok)})
    events = pd.DataFrame(rows, columns=["date", "symbol", "kind", "price_move", "ratio", "match",
                                         "volume_x", "value_x", "adjusted"])
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
            for f in ["close", "volume", "value", "isin"]}
    factor, events = detect_actions(cont["close"], cont["volume"], cont["value"], cont["isin"])
    events.to_csv(OUT / "corporate_actions.csv", index=False)
    done = events[events["adjusted"]]
    print(f"    corporate actions: {int((done['kind'] == 'split').sum())} splits/consolidations (ISIN changed), "
          f"{int((done['kind'] == 'bonus').sum())} bonuses; {int((~events['adjusted']).sum())} big moves "
          "left as real price moves (see corporate_actions.csv)", flush=True)
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
