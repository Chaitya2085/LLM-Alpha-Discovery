"""Paper journal: practise with pretend money, write down why, and learn from the record.

Ideas adapted from tradermonty/claude-trading-skills (trader-memory-core, pre-trade-discipline-gate,
drawdown-circuit-breaker, weekly-performance-digest; MIT). The code is our own. Personal mode only:
the journal lives in results/<market>/team/private/paper.json and is never committed. No real orders
exist anywhere in this project.

The Risk Officer's paper rules, fixed in advance:
    a written reason of at least 15 characters before any position opens
    at most 10 open positions, at most 20% of the starting balance in one stock
    the paper account down 10% or more from its high: no new positions until it recovers
    3 losing positions closed in a row: no new positions for 7 days
Each position can carry a stop (a fall from the entry price) and a planned holding time; the team flags
when either is reached. Prices are each day's close from our NSE data.
"""
from __future__ import annotations

import json
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

START_BALANCE = 1_000_000.0
MAX_OPEN = 10
MAX_SHARE = 0.20
DRAWDOWN_PAUSE = 0.10
LOSING_STREAK = 3
COOLDOWN_DAYS = 7
MIN_REASON = 15


class Refused(ValueError):
    """A rule says no; the message says which one."""


def path(folder: Path) -> Path:
    return folder / "private" / "paper.json"


def load(folder: Path) -> dict:
    f = path(folder)
    if f.exists():
        return json.loads(f.read_text())
    return {"start_balance": START_BALANCE, "trades": []}


def save(folder: Path, book: dict) -> None:
    f = path(folder)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(book, indent=1))


def _price(m, code: str, day=None) -> tuple[str, float]:
    s = m.raw_close[code].dropna() if code in m.raw_close.columns else pd.Series(dtype=float)
    if day is not None:
        s = s[s.index <= pd.Timestamp(day)]
    if s.empty:
        raise Refused(f"no price for {code}")
    return str(s.index[-1].date()), float(s.iloc[-1])


def _index_at(m, day: str) -> float:
    s = m.index_close.dropna()
    return float(s[s.index <= pd.Timestamp(day)].iloc[-1])


# ---------------------------------------------------------------- the account

def mark(book: dict, m) -> dict:
    """Every position at today's close, the account's value over time, and where the rules stand."""
    trades = book["trades"]
    start = book["start_balance"]
    rows = []
    for t in trades:
        last_day, last = _price(m, t["code"]) if t["status"] == "open" else (t["closed_on"], t["exit_price"])
        ret = last / t["entry_price"] - 1
        idx_ret = _index_at(m, last_day) / t["entry_index"] - 1
        days = (pd.Timestamp(last_day) - pd.Timestamp(t["opened_on"])).days
        flags = []
        if t["status"] == "open":
            if t.get("stop_pct") and ret <= -t["stop_pct"]:
                flags.append(f"below its stop ({ret:+.1%} vs -{t['stop_pct']:.0%})")
            if t.get("plan_weeks") and days >= 7 * t["plan_weeks"]:
                flags.append(f"planned holding time reached ({t['plan_weeks']} weeks)")
        rows.append({**t, "price": last, "price_date": last_day, "return": ret, "index_return": idx_ret,
                     "vs_index": ret - idx_ret, "value": t["amount"] * (1 + ret), "pnl": t["amount"] * ret,
                     "days": days, "flags": flags})
    # account value each day since the first position
    curve = {}
    if trades:
        first = min(pd.Timestamp(t["opened_on"]) for t in trades)
        days = m.raw_close.index[m.raw_close.index >= first]
        val = pd.Series(start, index=days)
        for t in trades:
            px = m.raw_close[t["code"]].reindex(days).ffill()
            end = pd.Timestamp(t["closed_on"]) if t["status"] == "closed" else days[-1]
            live = (days >= pd.Timestamp(t["opened_on"])) & (days <= end)
            gain = t["amount"] * (px / t["entry_price"] - 1)
            val = val + gain.where(live, 0).fillna(0)
            if t["status"] == "closed":
                val = val + pd.Series(np.where(days > end, t["amount"] * (t["exit_price"] / t["entry_price"] - 1), 0), index=days)
        curve = {"dates": [d.strftime("%Y-%m-%d") for d in val.index], "value": val.round(2).tolist()}
        value, peak = float(val.iloc[-1]), float(val.max())
    else:
        value, peak = start, start
    open_rows = [r for r in rows if r["status"] == "open"]
    closed = [r for r in rows if r["status"] == "closed"]
    invested = sum(r["amount"] for r in open_rows)
    return {"start_balance": start, "value": value, "peak": peak, "drawdown": value / peak - 1,
            "cash": value - sum(r["value"] for r in open_rows), "invested": invested,
            "open": open_rows, "closed": closed, "curve": curve, "gate": gate(book, value, peak),
            "digest": digest(closed)}


def gate(book: dict, value: float, peak: float, today: date | None = None) -> dict:
    """Whether a new position may open, and why not."""
    today = today or date.today()
    reasons = []
    open_n = sum(t["status"] == "open" for t in book["trades"])
    if open_n >= MAX_OPEN:
        reasons.append(f"already {open_n} open positions (the limit is {MAX_OPEN})")
    if peak > 0 and value / peak - 1 <= -DRAWDOWN_PAUSE:
        reasons.append(f"the paper account is {value / peak - 1:.1%} below its high (pause at -{DRAWDOWN_PAUSE:.0%})")
    closed = sorted([t for t in book["trades"] if t["status"] == "closed"], key=lambda t: t["closed_on"])
    streak = 0
    for t in reversed(closed):
        if t["exit_price"] < t["entry_price"]:
            streak += 1
        else:
            break
    if streak >= LOSING_STREAK:
        until = date.fromisoformat(closed[-1]["closed_on"]) + timedelta(days=COOLDOWN_DAYS)
        if today < until:
            reasons.append(f"{streak} losing positions in a row: cooling off until {until}")
    return {"allowed": not reasons, "reasons": reasons, "max_amount": book["start_balance"] * MAX_SHARE}


def digest(closed: list[dict]) -> dict:
    """The weekly-review numbers, from closed positions only."""
    if not closed:
        return {"closed": 0}
    r = np.array([c["return"] for c in closed])
    v = np.array([c["vs_index"] for c in closed])
    wins, losses = r[r > 0], r[r <= 0]
    by_source = {}
    for k, lab in ((True, "in the model's book when opened"), (False, "not in the book")):
        g = [c for c in closed if bool(c.get("in_book")) == k]
        if g:
            by_source[lab] = {"closed": len(g), "win_rate": float(np.mean([c["return"] > 0 for c in g])),
                              "avg_vs_index": float(np.mean([c["vs_index"] for c in g]))}
    return {"closed": len(closed), "win_rate": float((r > 0).mean()), "avg_return": float(r.mean()),
            "avg_vs_index": float(v.mean()), "beat_index_rate": float((v > 0).mean()),
            "avg_win": float(wins.mean()) if len(wins) else None, "avg_loss": float(losses.mean()) if len(losses) else None,
            "expectancy": float(r.mean()), "best": float(r.max()), "worst": float(r.min()), "by_source": by_source}


# ---------------------------------------------------------------- actions

def open_position(folder: Path, m, code: str, amount: float, reason: str, stop_pct: float | None = None,
                  plan_weeks: int | None = None, in_book: bool | None = None, rank: int | None = None,
                  today: date | None = None) -> dict:
    book = load(folder)
    reason = (reason or "").strip()
    if len(reason) < MIN_REASON:
        raise Refused(f"write down why first (at least {MIN_REASON} characters): the journal is for learning")
    if not amount or amount <= 0:
        raise Refused("the amount must be above zero")
    if amount > book["start_balance"] * MAX_SHARE:
        raise Refused(f"at most {MAX_SHARE:.0%} of the starting balance in one stock "
                      f"(₹{book['start_balance'] * MAX_SHARE:,.0f})")
    if any(t["status"] == "open" and t["code"] == code for t in book["trades"]):
        raise Refused(f"{code} is already open in the journal")
    acct = mark(book, m) if book["trades"] else {"value": book["start_balance"], "peak": book["start_balance"],
                                                  "cash": book["start_balance"]}
    g = gate(book, acct["value"], acct["peak"], today)
    if not g["allowed"]:
        raise Refused("; ".join(g["reasons"]))
    if amount > acct["cash"] + 1e-6:
        raise Refused(f"not enough paper cash (₹{acct['cash']:,.0f} left)")
    day, px = _price(m, code)
    t = {"id": uuid.uuid4().hex[:6], "code": code, "status": "open", "opened_on": day, "entry_price": px,
         "entry_index": _index_at(m, day), "amount": float(amount), "reason": reason,
         "stop_pct": stop_pct, "plan_weeks": plan_weeks, "in_book": in_book, "rank_at_open": rank,
         "created_at": datetime.now(timezone.utc).isoformat()}
    book["trades"].append(t)
    save(folder, book)
    return t


def close_position(folder: Path, m, trade_id: str, note: str = "") -> dict:
    book = load(folder)
    t = next((x for x in book["trades"] if x["id"] == trade_id), None)
    if t is None or t["status"] != "open":
        raise Refused("no open position with that id")
    day, px = _price(m, t["code"])
    t.update(status="closed", closed_on=day, exit_price=px, exit_note=note.strip(),
             closed_at=datetime.now(timezone.utc).isoformat())
    save(folder, book)
    return t
