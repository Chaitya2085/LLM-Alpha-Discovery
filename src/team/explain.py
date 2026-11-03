"""Turning the model's inputs into plain words, so the Explainer can only say true things.

The weekly model ranks stocks from about 200 inputs, each ranked across the day's stocks. For a stock in
the book, LightGBM reports how much each input pushed its score up or down. The Explainer takes the
top few and states them as facts: what the input measures, where this stock sits among the 200, and
which way it pushed. An LLM may then rephrase those facts, but it is given nothing else.
"""
from __future__ import annotations

import re

# Alpha158 families: (what a HIGH value means, in plain words). {d} = the window in days.
A158 = {
    "ROC": "fell more than most stocks over the past {d} days",
    "MA": "trades below its {d}-day average price",
    "STD": "has moved around a lot over {d} days (high volatility)",
    "BETA": "has been in an uptrend over {d} days",
    "RSQR": "has moved in a steady straight line over {d} days",
    "RESI": "sits above its {d}-day trend line",
    "MAX": "sits well below its {d}-day high",
    "MIN": "sits close to its {d}-day low",
    "QTLU": "sits low within its {d}-day price range",
    "QTLD": "sits low within its {d}-day price range",
    "RANK": "is high within its own {d}-day price history",
    "RSV": "closed near the top of its {d}-day high-low range",
    "IMAX": "made its {d}-day high recently",
    "IMIN": "made its {d}-day low recently",
    "IMXD": "made its {d}-day high more recently than its low",
    "CORR": "price and trading volume have moved together over {d} days",
    "CORD": "price changes and volume changes have moved together over {d} days",
    "CNTP": "has had many up days over the past {d} days",
    "CNTN": "has had many down days over the past {d} days",
    "CNTD": "has had more up days than down days over {d} days",
    "SUMP": "gains have outweighed losses over {d} days",
    "SUMN": "losses have outweighed gains over {d} days",
    "SUMD": "gains minus losses have been large over {d} days",
    "VMA": "traded less than its {d}-day average volume today",
    "VSTD": "trading volume has been unusually uneven over {d} days",
    "WVMA": "big price moves have come on heavy volume over {d} days",
    "VSUMP": "volume has been rising on more days than falling over {d} days",
    "VSUMN": "volume has been falling on more days than rising over {d} days",
    "VSUMD": "volume has been rising more than falling over {d} days",
}
CANDLE = {
    "KMID": "closed well above where it opened today", "KMID2": "closed well above its open, relative to the day's range",
    "KLEN": "had a wide high-low range today", "KUP": "had a long upper wick today (sold off from its high)",
    "KUP2": "had a long upper wick relative to the day's range", "KLOW": "had a long lower wick today (bounced off its low)",
    "KLOW2": "had a long lower wick relative to the day's range", "KSFT": "closed near the top of today's range",
    "KSFT2": "closed near the top of today's range, relative to its size",
    "OPEN0": "opened above where it closed", "HIGH0": "closed well below the day's high",
    "LOW0": "closed near the day's low", "VWAP0": "closed below the day's average traded price",
}
MARKET = {
    "mkt_ret20": "the market's move over the past month", "mkt_ret60": "the market's move over the past 3 months",
    "mkt_vol20": "how jumpy the market has been this month", "mkt_vol_pct": "how jumpy the market is compared with its history",
    "mkt_above_ma200": "whether the market is above its 200-day average", "mkt_dispersion": "how differently stocks are moving",
}


def describe(feature: str, factors: dict | None = None) -> tuple[str, str]:
    """(kind, plain meaning of a HIGH value) for one model input."""
    factors = factors or {}
    if feature.startswith("llm_"):
        f = factors.get(feature[4:], {})
        name = f.get("name", feature[4:]).replace("_", " ")
        hyp = f.get("hypothesis", "")
        return "llm", f"scores high on the AI-found signal \"{name}\"" + (f" ({hyp.rstrip('.')})" if hyp else "")
    if feature in MARKET:
        return "market", MARKET[feature]
    name = feature[5:] if feature.startswith("a158_") else feature
    if name in CANDLE:
        return "a158", CANDLE[name]
    m = re.fullmatch(r"([A-Z]+?)(\d+)", name)
    if m and m.group(1) in A158:
        return "a158", A158[m.group(1)].format(d=m.group(2))
    return "other", f"input {name}"


def fact(feature: str, rank_pct: float, contribution: float, factors: dict | None = None) -> str:
    """One true sentence: what the input says about this stock and which way it pushed the model."""
    kind, meaning = describe(feature, factors)
    push = "raised" if contribution > 0 else "lowered"
    if kind == "market":
        return f"Market backdrop ({meaning}) {push} the model's view of this stock."
    if rank_pct >= 0.5:
        where = f"ranks above {min(99, round(rank_pct * 100))}% of the 200 stocks"
    else:
        where = f"ranks below {min(99, round((1 - rank_pct) * 100))}% of the 200 stocks"
    return f"On \"{meaning}\", this stock {where}. That {push} its score."


def template_note(facts: list[str]) -> dict:
    """The words used when no LLM is available: the facts themselves, plus a generic watch-out."""
    return {"summary": " ".join(facts[:3]),
            "watch": "This view would be wrong if the patterns above reverse, or if company news changes the picture."}
