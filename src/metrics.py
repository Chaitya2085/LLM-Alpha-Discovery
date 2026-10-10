"""Signal and portfolio metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd

ANN = 252


def ic_series(score: pd.DataFrame, fwd_ret: pd.DataFrame, member: pd.DataFrame,
              method: str = "spearman") -> pd.Series:
    """Daily cross-sectional correlation between score[t] and fwd_ret[t], CSI300 members only."""
    s = score.where(member)
    r = fwd_ret.reindex_like(s).where(member)
    if method == "spearman":
        s = s.rank(axis=1)
        r = r.rank(axis=1)
    valid = s.notna() & r.notna()
    s = s.where(valid)
    r = r.where(valid)
    sd = s.sub(s.mean(axis=1), axis=0)
    rd = r.sub(r.mean(axis=1), axis=0)
    num = (sd * rd).sum(axis=1)
    den = np.sqrt((sd ** 2).sum(axis=1) * (rd ** 2).sum(axis=1))
    ic = num / den
    ic[valid.sum(axis=1) < 30] = np.nan
    return ic


def ic_summary(ic: pd.Series) -> dict:
    ic = ic.dropna()
    return {
        "IC_mean": ic.mean(),
        "ICIR": ic.mean() / ic.std() if ic.std() > 0 else np.nan,
        "IC_t": ic.mean() / ic.std() * np.sqrt(len(ic)) if ic.std() > 0 else np.nan,
        "IC_pos_pct": (ic > 0).mean(),
    }


def perf(ret: pd.Series) -> dict:
    ret = ret.dropna()
    if ret.empty:
        return {}
    eq = (1 + ret).cumprod()
    years = len(ret) / ANN
    cagr = eq.iloc[-1] ** (1 / years) - 1
    vol = ret.std() * np.sqrt(ANN)
    dd = eq / eq.cummax() - 1
    return {
        "ann_return": cagr,
        "ann_vol": vol,
        "sharpe": ret.mean() / ret.std() * np.sqrt(ANN) if ret.std() > 0 else np.nan,
        "max_drawdown": dd.min(),
        "total_return": eq.iloc[-1] - 1,
    }


def backtest_summary(daily: pd.DataFrame) -> dict:
    net = perf(daily["net"])
    gross = perf(daily["gross"])
    bench = perf(daily["bench"])
    excess = perf(daily["net"] - daily["bench"])
    return {
        "net_ann_return": net["ann_return"],
        "net_sharpe": net["sharpe"],
        "net_max_dd": net["max_drawdown"],
        "gross_ann_return": gross["ann_return"],
        "bench_ann_return": bench["ann_return"],
        "excess_ann_return": excess["ann_return"],
        "info_ratio": excess["sharpe"],
        "excess_max_dd": excess["max_drawdown"],
        "avg_daily_turnover": daily["turnover"].mean(),
        "ann_cost_drag": daily["cost"].mean() * ANN,
    }


def yearly(daily: pd.DataFrame, col: str = "net") -> pd.Series:
    return daily[col].groupby(daily.index.year).apply(lambda r: (1 + r).prod() - 1)
