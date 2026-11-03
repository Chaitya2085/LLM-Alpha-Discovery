"""The Compliance Officer's rules: checks every piece of text before the team saves or shares it.

Rule-based on purpose (the same text always gets the same verdict, and every refusal names its rule).
Not legal advice: a SEBI compliance professional should review these rules before anything is public.

  advice words      buy / sell / accumulate / exit / target price / stop-loss / entry price
  promises          guaranteed, sure-shot, risk-free, multibagger, "will rise", "will double"...
  return promises   "will give 20%", "expected return of ..."
  learn mode        no data newer than 30 days, and no names of stocks in a current book
  disclaimer        every report carries the research-only disclaimer
"""
from __future__ import annotations

import re
from datetime import date, timedelta

DISCLAIMER = ("Research and education only. Not investment advice, not a recommendation to buy or sell "
              "any security. Past results do not guarantee future results.")
LEARN_LAG_DAYS = 30

RULES = [
    ("advice-word", r"\b(buy|sell|accumulate|book profits?|exit (?:the )?(?:stock|position))\b"),
    ("price-call", r"\b(target(?: price)?|stop[- ]?loss|entry (?:price|level)|price target)\b"),
    ("promise", r"\b(guarantee[ds]?|sure[- ]?shot|risk[- ]?free|can'?t lose|multibagger|jackpot)\b"),
    ("prediction-promise", r"\bwill (?:surely |definitely )?(?:rise|go up|double|triple|rally|outperform|beat)\b"),
    ("return-promise", r"\b(?:will (?:give|return|deliver|make)|expected return of)\b[^.]{0,40}\d+\s?%"),
]


def check(text: str, mode: str = "personal", as_of: date | None = None, today: date | None = None,
          current_names: set[str] | None = None, require_disclaimer: bool = False) -> list[str]:
    """Problems found in `text` (empty list = it may go out)."""
    problems = []
    low = text.lower()
    body = low.replace(DISCLAIMER.lower(), "")        # the disclaimer itself says "buy or sell"
    for rule, pat in RULES:
        m = re.search(pat, body)
        if m:
            problems.append(f"{rule}: \"{m.group(0)}\"")
    if mode == "learn":
        today = today or date.today()
        if as_of and (today - as_of) < timedelta(days=LEARN_LAG_DAYS):
            problems.append(f"data-age: uses data from {as_of}, less than {LEARN_LAG_DAYS} days old")
        for name in sorted(current_names or ()):
            if re.search(rf"\b{re.escape(name.lower())}\b", low):
                problems.append(f"current-pick: names {name}, a stock in a current book")
    if require_disclaimer and DISCLAIMER.lower()[:60] not in low:
        problems.append("disclaimer: missing")
    return problems
