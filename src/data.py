"""Load processed data as wide (date x code) matrices — the format every later
phase (factor evaluation, LLM-generated factors, backtests) works in.

load_market() reads the market chosen with --market / LLM_ALPHA_MARKET (default china)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import markets

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"  # China (kept for backwards compatibility)


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
    member: pd.DataFrame          # in the universe that day (point-in-time)
    can_buy: pd.DataFrame         # tradable to buy at that day's open
    can_sell: pd.DataFrame        # tradable to sell at that day's open
    fwd_ret: pd.DataFrame         # open(t+1) -> open(t+2): what a signal formed at close t earns
    index_close: pd.Series        # benchmark index level at close
    index_fwd_ret: pd.Series      # benchmark open(t+1) -> open(t+2), matches fwd_ret
    name: str = "china"
    buy_cost: float = 0.0005
    sell_cost: float = 0.0015
    bench_label: str = "CSI300"
    universe_label: str = "CSI300"

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


def load_market(market: str | None = None) -> Market:
    cfg = markets.get(market)
    proc = cfg.proc
    if not (proc / "prices.parquet").exists():
        hint = "python run_india.py" if cfg.name == "india" else "python run_phase1.py"
        raise SystemExit(f"No {cfg.name} data in {proc}. Run first:  {hint}")
    px = pd.read_parquet(proc / "prices.parquet")
    wide = {f: px[f].unstack("code").astype("float64")
            for f in ["open", "high", "low", "close", "volume", "vwap", "raw_close", "factor"]}
    member = pd.read_parquet(proc / "membership.parquet")
    dates = member.index
    codes = member.columns
    for k in wide:
        wide[k] = wide[k].reindex(index=dates, columns=codes)

    raw_open = wide["open"] / wide["factor"]
    suspended = wide["volume"].isna() | (wide["volume"] <= 0)
    if cfg.limit_rule == "china":
        prev_raw_close = wide["raw_close"].shift(1)
        gap = raw_open / prev_raw_close - 1
        lim = _limit(codes, dates)
        can_buy = ~suspended & ~(gap >= lim - 0.002)     # can't buy a stock opening limit-up
        can_sell = ~suspended & ~(gap <= -lim + 0.002)   # can't sell one opening limit-down
    else:
        # bands differ by stock (2/5/10/20%, none for F&O stocks), so detect the lock itself:
        # one price all day (high == low) after a move of 2%+ means locked at the band
        move = wide["close"] / wide["close"].ffill().shift(1) - 1
        locked = (wide["high"] == wide["low"]) & wide["high"].notna()
        can_buy = ~suspended & ~(locked & (move >= 0.019))
        can_sell = ~suspended & ~(locked & (move <= -0.019))

    o = wide["open"]
    fwd_ret = o.shift(-2) / o.shift(-1) - 1

    ix = pd.read_parquet(proc / "index.parquet").reindex(dates)
    idx = ix["level"]
    io = ix["level_open"]
    index_fwd_ret = io.shift(-2) / io.shift(-1) - 1

    return Market(
        open=o, high=wide["high"], low=wide["low"], close=wide["close"],
        volume=wide["volume"], vwap=wide["vwap"], raw_close=wide["raw_close"],
        raw_open=raw_open, member=member.astype(bool),
        can_buy=can_buy, can_sell=can_sell, fwd_ret=fwd_ret, index_close=idx,
        index_fwd_ret=index_fwd_ret, name=cfg.name, buy_cost=cfg.buy_cost,
        sell_cost=cfg.sell_cost, bench_label=cfg.bench_label, universe_label=cfg.universe_label,
    )
