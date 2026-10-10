"""Prompts for factor generation.

Deliberate choices:
  * No market name, country, tickers or dates. The model is told only that the
    data is daily large-cap equities. This keeps the prompt identical for the
    Phase 5 leakage experiment and avoids nudging the model to recall what
    happened in a specific market and year.
  * The known Alpha158 families are listed so the model aims for new ideas
    rather than restating the baseline library.
  * Phase 1 showed that fast signals die to trading costs, so the prompt asks
    for signals that stay stable for days to weeks.
"""
from __future__ import annotations

from dsl import grammar_doc

PROMPT_VERSION = "v1.1"  # v1.1: show at most 40 existing expressions

CATEGORIES = {
    "momentum": "trend continuation over weeks",
    "reversal": "short-term overreaction that tends to undo",
    "volatility": "risk, dispersion and the shape of the return distribution",
    "volume": "trading activity, liquidity and attention",
    "price_volume": "how price moves interact with volume",
    "intraday": "open/high/low/close shape within the day and gaps between days",
}

KNOWN_LIBRARY = """\
The baseline library already contains these families (window 5/10/20/30/60 days); do NOT
restate them or trivial rescalings of them:
- candle shape: (close-open)/open, (high-low)/open, upper/lower shadows, (2*close-high-low)/open
- price level vs close: open/close, high/close, low/close, vwap/close
- Ref(close,d)/close (rate of change), Mean(close,d)/close, Std(close,d)/close
- trend: Slope, Rsquare, Resi of close; TsMax(high,d)/close, TsMin(low,d)/close
- position in range: TsRank(close,d), (close-TsMin(low,d))/(TsMax(high,d)-TsMin(low,d)), close quantiles
- IdxMax(high,d), IdxMin(low,d) and their difference
- Corr(close, Log(volume), d) and Corr(returns, Log(volume change), d)
- share of up days / down days, sum of gains vs losses over d days (RSI-like)
- Mean(volume,d)/volume, Std(volume,d)/volume, up-volume vs down-volume ratios
- Std/Mean of |returns|*volume"""

SYSTEM = f"""You are a quantitative researcher who designs alpha factors for a long-only \
equity strategy. You write each factor as an expression in a small domain-specific language.

The language:
{grammar_doc()}

How factors are used:
- Universe: a few hundred large-cap stocks, daily data only. Nothing else is available
  (no fundamentals, no news, no market or sector identifiers).
- Each day after the close, stocks are ranked by the factor; HIGHER values = more
  attractive. The top-ranked stocks are bought at the next open.
- Trading costs are significant, so signals that change slowly (useful for days to weeks)
  are much more valuable than signals that flip every day. Smoothing with Mean, EMA or Decay
  helps.
- Prefer economically motivated ideas you can explain in one sentence. Simple beats clever.

{KNOWN_LIBRARY}

Reply with ONLY a JSON object, no other text:
{{"factors": [{{"name": "short_snake_case", "category": "<one of: {', '.join(CATEGORIES)}>",
  "expression": "<valid expression>", "hypothesis": "<one sentence: why it should predict returns>"}}]}}"""


def user_prompt(n: int, batch: int, existing: list[str] | None = None) -> str:
    cats = "\n".join(f"- {k}: {v}" for k, v in CATEGORIES.items())
    msg = (f"Batch {batch}. Propose {n} new, diverse alpha factors, spread roughly evenly across "
           f"these categories:\n{cats}\n\nCheck every expression against the language rules "
           f"(operator names, argument counts, integer windows 1-60).")
    if existing:
        shown = "\n".join(f"- {e}" for e in existing[-40:])
        msg += f"\n\nThese expressions already exist; propose different ideas:\n{shown}"
    return msg
