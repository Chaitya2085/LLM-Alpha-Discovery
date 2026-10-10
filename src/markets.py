"""Per-market settings, so the same pipeline runs on China (research benchmark) or India.

Pick the market with `--market india` on any script (default: china), or the
LLM_ALPHA_MARKET environment variable. China keeps its original paths, so
existing results are untouched.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class MarketConfig:
    name: str
    proc: Path            # processed data folder
    results: Path         # results folder
    fig_prefix: str       # prefix for figure file names
    bench_label: str      # benchmark index name used in labels
    universe_label: str   # universe name used in labels
    buy_cost: float       # fraction of trade value
    sell_cost: float
    limit_rule: str       # "china": board price limits; "lock": detect locked circuits from prices
    currency: str


MARKETS = {
    "china": MarketConfig(
        name="china", proc=ROOT / "data" / "processed", results=ROOT / "results", fig_prefix="",
        bench_label="CSI300", universe_label="CSI300",
        buy_cost=0.0005, sell_cost=0.0015,      # Qlib's China defaults (commission + stamp duty)
        limit_rule="china", currency="CNY"),
    "india": MarketConfig(
        name="india", proc=ROOT / "data" / "processed" / "india", results=ROOT / "results" / "india",
        fig_prefix="india_", bench_label="Nifty 200", universe_label="top-200 NSE",
        # delivery trades: STT 0.1% each way, stamp duty 0.015% on buys, exchange + GST ~0.004%,
        # plus a flat DP charge (~Rs 16 per stock sold, ~0.02-0.03% on a typical ticket)
        buy_cost=0.0012, sell_cost=0.0013,
        limit_rule="lock", currency="INR"),
}


def get(name: str | None = None) -> MarketConfig:
    name = (name or os.environ.get("LLM_ALPHA_MARKET") or "china").lower()
    if name not in MARKETS:
        raise SystemExit(f"unknown market '{name}' (choose: {', '.join(MARKETS)})")
    return MARKETS[name]


def from_argv() -> MarketConfig:
    """Read --market from the command line (leaving other arguments alone) and remember it."""
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--market", default=None)
    known, rest = ap.parse_known_args()
    sys.argv = [sys.argv[0], *rest]
    cfg = get(known.market)
    os.environ["LLM_ALPHA_MARKET"] = cfg.name
    return cfg
