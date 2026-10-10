"""Load processed data as wide (date x code) matrices — the format every later
phase (factor evaluation, LLM-generated factors, backtests) works in."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"


@dataclass
class Market:
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    vwap: pd.DataFrame
    raw_close: pd.DataFrame
    raw_open: pd.DataFrame
    member: pd.DataFrame          # in CSI300 that day (point-in-time)
    can_buy: pd.DataFrame         # tradable to buy at that day's open
    can_sell: pd.DataFrame        # tradable to sell at that day's open
    fwd_ret: pd.DataFrame         # open(t+1) -> open(t+2): what a signal formed at close t earns
    index_close: pd.Series        # CSI300 index level at close
    index_fwd_ret: pd.Series      # CSI300 open(t+1) -> open(t+2), matches fwd_ret

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.close.index


def _limit(codes: pd.Index, dates: pd.DatetimeIndex) -> pd.DataFrame:
    lim = pd.DataFrame(0.10, index=dates, columns=codes)
    for c in codes:
        n = c[2:]
        if n.startswith("688"):
            lim[c] = 0.20
        elif n.startswith("300"):
            lim.loc[dates >= "2020-08-24", c] = 0.20
    return lim


def load_market() -> Market:
    px = pd.read_parquet(PROC / "prices.parquet")
    wide = {f: px[f].unstack("code").astype("float64")
            for f in ["open", "high", "low", "close", "volume", "vwap", "raw_close", "factor"]}
    member = pd.read_parquet(PROC / "membership.parquet")
    dates = member.index
    codes = member.columns
    for k in wide:
        wide[k] = wide[k].reindex(index=dates, columns=codes)

    raw_open = wide["open"] / wide["factor"]
    prev_raw_close = wide["raw_close"].shift(1)
    gap = raw_open / prev_raw_close - 1
    lim = _limit(codes, dates)
    suspended = wide["volume"].isna() | (wide["volume"] <= 0)
    can_buy = ~suspended & ~(gap >= lim - 0.002)     # can't buy a stock opening limit-up
    can_sell = ~suspended & ~(gap <= -lim + 0.002)   # can't sell one opening limit-down

    o = wide["open"]
    fwd_ret = o.shift(-2) / o.shift(-1) - 1

    ix = pd.read_parquet(PROC / "index.parquet").reindex(dates)
    idx = ix["level"]
    io = ix["level_open"]
    index_fwd_ret = io.shift(-2) / io.shift(-1) - 1

    return Market(
        open=o, high=wide["high"], low=wide["low"], close=wide["close"],
        volume=wide["volume"], vwap=wide["vwap"], raw_close=wide["raw_close"],
        raw_open=raw_open, member=member.astype(bool),
        can_buy=can_buy, can_sell=can_sell, fwd_ret=fwd_ret, index_close=idx,
        index_fwd_ret=index_fwd_ret,
    )
