"""A small synthetic market for tests, saved in the same format as the real processed data.

One planted effect: heavily traded stocks drift up (about 15%/yr spread between the most and
least traded), so `Mean(volume, 20)` predicts returns and `-Mean(volume, 20)` works in reverse.
Everything else is noise.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd

import markets


def make_market(root: Path, name: str = "synth", n: int = 120, seed: int = 1,
                start: str = "2015-06-01", end: str = "2020-06-30", drift_size: float = 0.0006):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    T = len(dates)
    codes = [f"S{i:03d}" for i in range(n)]
    liq = rng.normal(0, 1, n)                          # hidden trait: liquidity
    drift = drift_size * liq                           # liquid stocks drift up
    ret = drift + rng.normal(0, 0.015, (T, n))
    close = 100 * np.exp(np.cumsum(np.log1p(ret), axis=0))
    opn = close * (1 + rng.normal(0, 0.003, (T, n)))
    high = np.maximum(opn, close) * (1 + np.abs(rng.normal(0, 0.005, (T, n))))
    low = np.minimum(opn, close) * (1 - np.abs(rng.normal(0, 0.005, (T, n))))
    vol = np.exp(12 + 0.8 * liq + rng.normal(0, 0.3, (T, n)))
    idx = pd.MultiIndex.from_product([dates, codes], names=["date", "code"])
    flat = lambda a: a.reshape(-1)  # noqa: E731
    px = pd.DataFrame({"open": flat(opn), "high": flat(high), "low": flat(low), "close": flat(close),
                       "volume": flat(vol), "vwap": flat((high + low + close) / 3), "raw_close": flat(close),
                       "factor": 1.0}, index=idx)
    proc = root / "proc"
    proc.mkdir(parents=True, exist_ok=True)
    px.to_parquet(proc / "prices.parquet")
    pd.DataFrame(True, index=dates, columns=codes).rename_axis("date").to_parquet(proc / "membership.parquet")
    lvl = 1000 * np.exp(np.cumsum(np.log1p(ret.mean(axis=1))))
    pd.DataFrame({"level": lvl, "level_open": lvl, "open": lvl, "close": lvl},
                 index=pd.Index(dates, name="date")).to_parquet(proc / "index.parquet")
    cfg = dataclasses.replace(markets.MARKETS["india"], name=name, proc=proc, results=root / "results",
                              fig_prefix=f"{name}_")
    markets.MARKETS[name] = cfg
    return cfg
