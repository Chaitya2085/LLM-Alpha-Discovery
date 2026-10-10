"""Daily Top-K / Dropout backtest (same logic as Qlib's TopkDropoutStrategy).

Timing, chosen so nothing peeks ahead:
  - score[t] uses data up to the close of day t
  - trades execute at the OPEN of day t+1
  - the position then earns open(t+1) -> open(t+2)  (Market.fwd_ret[t])
Realism:
  - only CSI300 members on day t are eligible to buy
  - no buying a stock that opens limit-up or is suspended; no selling one that
    opens limit-down or is suspended (it stays in the book)
  - costs: 0.05% on buys, 0.15% on sells (Qlib's default China A-share rates,
    covering commission + stamp duty)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data import Market

BUY_COST, SELL_COST = 0.0005, 0.0015


@dataclass
class BacktestResult:
    daily: pd.DataFrame      # gross, cost, net, bench, turnover, n_holdings
    holdings: dict           # date -> list of codes held after trading


def topk_dropout(score: pd.DataFrame, mkt: Market, topk: int = 50, n_drop: int = 5,
                 start: str | None = None, end: str | None = None) -> BacktestResult:
    dates = score.index
    if start:
        dates = dates[dates >= start]
    if end:
        dates = dates[dates <= end]
    all_dates = mkt.dates
    pos_of = {d: i for i, d in enumerate(all_dates)}

    held: list[str] = []
    rows, book = [], {}
    for d in dates:
        i = pos_of[d]
        if i + 2 >= len(all_dates):
            break
        exec_day = all_dates[i + 1]
        s = score.loc[d]
        elig = mkt.member.loc[d] & s.notna()
        s_elig = s[elig].sort_values(ascending=False)

        can_buy = mkt.can_buy.loc[exec_day]
        can_sell = mkt.can_sell.loc[exec_day]

        # rank current holdings; ones that left the universe / lost a score go to the bottom
        held_scores = s.reindex(held).where(elig.reindex(held, fill_value=False), -np.inf).fillna(-np.inf)
        last = list(held_scores.sort_values(ascending=False).index)

        need = n_drop + topk - len(last)
        today = [c for c in s_elig.index if c not in set(last) and can_buy.get(c, False)][: max(need, 0)]
        comb_scores = pd.concat([held_scores, s_elig.reindex(today)]).sort_values(ascending=False)
        bottom = set(comb_scores.index[-n_drop:]) if n_drop > 0 else set()
        sell = [c for c in last if (c in bottom or held_scores[c] == -np.inf) and can_sell.get(c, False)]
        n_buy = len(sell) + topk - len(last)
        buy = today[: max(n_buy, 0)]

        new_held = [c for c in last if c not in set(sell)] + buy
        n = max(len(new_held), 1)
        turnover_buy = len(buy) / n
        turnover_sell = len(sell) / max(len(last), 1) if last else 0.0
        cost = turnover_buy * BUY_COST + turnover_sell * SELL_COST

        r = mkt.fwd_ret.loc[d].reindex(new_held).fillna(0.0)  # suspended-through = flat
        gross = float(r.mean()) if new_held else 0.0
        bench = float(mkt.index_fwd_ret.loc[d])

        rows.append({"date": d, "gross": gross, "cost": cost, "net": gross - cost,
                     "bench": bench, "turnover": (turnover_buy + turnover_sell) / 2,
                     "n_holdings": len(new_held)})
        book[d] = new_held
        held = new_held

    daily = pd.DataFrame(rows).set_index("date")
    return BacktestResult(daily=daily, holdings=book)
