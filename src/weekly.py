"""The weekly judge: how a factor would really be used. Long-only, held a week, after costs.

Phase 2 scored factors on next-day returns. The product holds stocks for about a week
(Indian trading costs make daily trading too expensive), so Phase 4 judges every factor
the way it would be traded:

  * every 5 trading days, after the close, rank the universe by the factor
    (higher = more attractive, exactly as the LLM proposed it; it is NOT flipped)
  * buy the top fifth at the next open, equal weight, hold 5 trading days
    (earn open t+1 -> open t+6); stocks locked at a circuit can't be bought
  * pay trading costs on the part of the portfolio that changed
  * compare with holding the whole universe equally (so a factor isn't rewarded just
    for leaning towards big or small stocks), and with the benchmark index

Discovery period only (2014-07 to 2020): 2021 onward stays untouched for Phase 6.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from metrics import ic_series

DISC_START, DISC_END = "2014-07-01", "2020-12-31"
HOLD = 5            # trading days per holding period
WEEKS_PER_YEAR = 52
TOP = 0.2           # top fifth


@dataclass
class WeeklySetup:
    days: pd.DatetimeIndex      # rebalance days (signal at their close)
    member: pd.DataFrame        # universe on rebalance days
    fwd: pd.DataFrame           # open(t+1) -> open(t+6) per stock
    buyable: pd.DataFrame       # can buy at open t+1
    univ: pd.Series             # equal-weight universe return over the same week
    bench: pd.Series            # benchmark index over the same week
    cost_round_trip: float


def setup(m, start: str = DISC_START, end: str = DISC_END) -> WeeklySetup:
    all_days = m.dates
    o = m.open
    fwd = o.shift(-(HOLD + 1)) / o.shift(-1) - 1
    # benchmark over the same window, chained from its daily open->open steps (t .. t+4)
    log_i = np.log1p(m.index_fwd_ret)
    bench_all = np.expm1(log_i[::-1].rolling(HOLD, min_periods=HOLD).sum()[::-1])
    inside = all_days[(all_days >= start) & (all_days <= end)]
    # the week's return must end before `end` (no peeking into the held-out years)
    pos = {d: i for i, d in enumerate(all_days)}
    last_ok = [d for d in inside if pos[d] + HOLD + 1 < len(all_days) and all_days[pos[d] + HOLD + 1] <= pd.Timestamp(end)]
    days = pd.DatetimeIndex(last_ok[::HOLD])
    member = m.member.loc[days]
    exec_day = [all_days[pos[d] + 1] for d in days]
    buyable = pd.DataFrame(m.can_buy.loc[exec_day].to_numpy(), index=days, columns=m.member.columns)
    f = fwd.loc[days].where(member)
    return WeeklySetup(days=days, member=member, fwd=f, buyable=buyable.fillna(False).astype(bool),
                       univ=f.mean(axis=1), bench=bench_all.reindex(days),
                       cost_round_trip=m.buy_cost + m.sell_cost)


def judge(factor: pd.DataFrame, ws: WeeklySetup) -> dict:
    """All weekly numbers for one factor (as proposed: higher = buy)."""
    s = factor.reindex(index=ws.days, columns=ws.member.columns).where(ws.member)
    ic = ic_series(s, ws.fwd, ws.member, "spearman").dropna()
    if len(ic) < 52 or ic.std() == 0:
        return {"status": "too_sparse", "weeks": int(len(ic))}
    pct = s.rank(axis=1, pct=True)
    top = (pct > 1 - TOP) & ws.buyable
    n_top = top.sum(axis=1)
    gross = ws.fwd.where(top).mean(axis=1)
    # turnover: share of this week's portfolio that wasn't held last week (buys = sells in size)
    prev = top.shift(1, fill_value=False)
    overlap = (top & prev).sum(axis=1)
    turnover = (1 - overlap / n_top.where(n_top > 0)).fillna(1.0)
    turnover.iloc[0] = 1.0
    cost = turnover * ws.cost_round_trip
    net = gross - cost
    ex_univ = (net - ws.univ).dropna()
    ex_gross = (gross - ws.univ).dropna()
    ex_bench = (net - ws.bench).dropna()
    by_year = ic.groupby(ic.index.year).mean()
    sign = 1.0 if ic.mean() >= 0 else -1.0
    return {
        "status": "scored",
        "weeks": int(len(ic)),
        "weekly_ic": float(ic.mean()),
        "weekly_ic_t": float(ic.mean() / ic.std() * np.sqrt(len(ic))),
        "years_same_sign": float((np.sign(by_year) == sign).mean()),
        "gross_excess_ann": float(ex_gross.mean() * WEEKS_PER_YEAR),
        "net_excess_ann": float(ex_univ.mean() * WEEKS_PER_YEAR),
        "net_vs_bench_ann": float(ex_bench.mean() * WEEKS_PER_YEAR),
        "hit_rate": float((ex_univ > 0).mean()),
        "turnover": float(turnover.iloc[1:].mean()),
        "cost_ann": float(cost.iloc[1:].mean() * WEEKS_PER_YEAR),
    }
