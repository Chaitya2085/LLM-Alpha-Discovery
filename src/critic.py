"""The Critic agent: turns the weekly judge's numbers into a verdict per factor and plain
feedback for the Researcher that proposed them.

It is deliberately rule-based (no LLM): the same scores always give the same feedback, it
costs nothing, and every verdict can be traced to a number. The feedback never mentions a
market, dates or tickers, so prompts stay market-blind for the Phase 6 leakage test.

Verdicts, checked in this order:
  broken     the expression didn't parse or couldn't be computed
  duplicate  exactly the same expression as an earlier factor
  copy       ranks stocks almost like a known baseline feature or an earlier good factor (|corr| >= 0.7)
  backwards  a real signal, but in the opposite direction: the stocks it likes do worse (t <= -2)
  weak       no reliable signal (|t| < 2)
  unstable   right in some years and wrong in others (same direction in < 60% of years)
  costly     a real signal, but after trading costs it doesn't beat holding everything
  keep       works after costs
  star       works after costs, strongly: t >= 3, +3%/yr or more, beats the average in 55%+ of weeks
"""
from __future__ import annotations

from collections import Counter

T_SIGNAL = 2.0
COPY = 0.7
STABLE = 0.6
USEFUL = ("keep", "star")
ORDER = ["star", "keep", "costly", "unstable", "backwards", "copy", "weak", "duplicate", "broken"]


def verdict(r: dict) -> tuple[str, str]:
    """(verdict, one-line reason) for one factor record with judge results merged in."""
    if r.get("status") == "invalid":
        return "broken", f"invalid expression: {r.get('error', '')[:140]}"
    if r.get("status") == "duplicate":
        return "duplicate", "same expression as an earlier factor"
    if r.get("judge_status") != "scored":
        return "broken", r.get("judge_error") or "could not be computed on enough days"
    t, net, gross = r["weekly_ic_t"], r["net_excess_ann"], r["gross_excess_ann"]
    copy_corr = max(r.get("max_corr_known") or 0, r.get("max_corr_prev") or 0)
    if copy_corr >= COPY:
        like = r.get("closest_known") if (r.get("max_corr_known") or 0) >= (r.get("max_corr_prev") or 0) \
            else r.get("closest_prev")
        return "copy", f"ranks stocks almost like {like} (corr {copy_corr:.2f})"
    if t <= -T_SIGNAL:
        return "backwards", f"works in reverse: the stocks it likes do worse (t {t:.1f}); the opposite idea may be right"
    if abs(t) < T_SIGNAL:
        return "weak", f"no reliable signal (t {t:.1f})"
    if r["years_same_sign"] < STABLE:
        return "unstable", f"right in only {r['years_same_sign']:.0%} of years"
    if net <= 0:
        why = (f"before costs {gross:+.1%}/yr, after costs {net:+.1%}/yr; "
               f"{r['turnover']:.0%} of the portfolio is replaced each week")
        return "costly", why
    star = t >= 3 and net >= 0.03 and r["hit_rate"] >= 0.55
    return ("star" if star else "keep"), (f"{net:+.1%}/yr after costs, beats the average in "
                                          f"{r['hit_rate']:.0%} of weeks (t {t:.1f})")


def review(records: list[dict]) -> list[dict]:
    """Add 'verdict' and 'reason' to each record (returns new dicts)."""
    out = []
    for r in records:
        v, why = verdict(r)
        out.append({**r, "verdict": v, "reason": why})
    return out


def round_stats(reviewed: list[dict]) -> dict:
    n = len(reviewed)
    c = Counter(r["verdict"] for r in reviewed)
    scored = [r for r in reviewed if r.get("judge_status") == "scored" and r["verdict"] not in ("duplicate",)]
    useful = [r for r in reviewed if r["verdict"] in USEFUL]
    abs_ic = sorted(abs(r["weekly_ic"]) for r in scored)
    return {
        "proposed": n,
        "valid_pct": 100 * sum(r.get("status") == "valid" for r in reviewed) / n if n else 0.0,
        "scored": len(scored),
        "useful": len(useful),
        "useful_pct": 100 * len(useful) / n if n else 0.0,
        "stars": c.get("star", 0),
        "backwards_pct": 100 * c.get("backwards", 0) / n if n else 0.0,
        "copy_pct": 100 * c.get("copy", 0) / n if n else 0.0,
        "costly_pct": 100 * c.get("costly", 0) / n if n else 0.0,
        "broken_pct": 100 * c.get("broken", 0) / n if n else 0.0,
        "median_abs_weekly_ic": abs_ic[len(abs_ic) // 2] if abs_ic else float("nan"),
        "best_net_excess": max((r["net_excess_ann"] for r in useful), default=float("nan")),
        "verdicts": dict(c),
    }


def lessons(reviewed: list[dict]) -> list[str]:
    out = []
    by_cat: dict[str, list[dict]] = {}
    for r in reviewed:
        if r.get("judge_status") == "scored":
            by_cat.setdefault(r.get("category") or "other", []).append(r)
    for cat, rs in sorted(by_cat.items()):
        back = sum(r["verdict"] == "backwards" for r in rs)
        good = sum(r["verdict"] in USEFUL for r in rs)
        if len(rs) >= 2 and back / len(rs) >= 0.5:
            out.append(f"Your {cat} ideas mostly worked in reverse here; consider the opposite effect.")
        elif good >= 2:
            out.append(f"Your {cat} ideas worked best; push further in that direction.")
    signal = [r for r in reviewed if r.get("judge_status") == "scored" and abs(r["weekly_ic_t"]) >= T_SIGNAL]
    if signal and sorted(r["turnover"] for r in signal)[len(signal) // 2] > 0.5:
        out.append("Your signals change too fast: over half the portfolio is replaced each week. "
                   "Smooth them (Mean, EMA, Decay) or use longer windows.")
    n = len(reviewed)
    if n and sum(r["verdict"] == "broken" for r in reviewed) / n > 0.2:
        out.append("Many expressions broke the language rules; check operator names, argument counts "
                   "and integer windows.")
    if sum(r["verdict"] == "copy" for r in reviewed) >= 3:
        out.append("Several ideas restated known factors; look for different information.")
    return out


def _line(r: dict) -> str:
    return f"- {r.get('name') or r.get('id')}: {r.get('expression') or r.get('raw_expression')}  ->  {r['reason']}"


def feedback_text(reviewed: list[dict], max_items: int = 5) -> str:
    """Plain feedback for the next prompt (market-blind)."""
    st = round_stats(reviewed)
    c = st["verdicts"]
    parts = [
        "Results of your previous round. Each factor was tested the way it will be used: every week, "
        "buy the top fifth of stocks ranked by the factor, hold them 5 trading days, pay trading "
        "costs, and compare with simply holding all stocks equally.",
        f"Scoreboard: {st['proposed']} proposed; {st['useful']} worked after costs"
        f" ({c.get('star', 0)} strongly), {c.get('backwards', 0)} worked in reverse, {c.get('costly', 0)} too costly, "
        f"{c.get('copy', 0)} copies, {c.get('weak', 0) + c.get('unstable', 0)} no reliable signal, "
        f"{c.get('broken', 0) + c.get('duplicate', 0)} broken or duplicate.",
    ]
    sections = [
        (("star", "keep"), "Build on these (they worked after costs):", lambda r: -r["net_excess_ann"]),
        (("backwards",), "Worked in REVERSE (the stocks they liked did worse):", lambda r: r["weekly_ic_t"]),
        (("costly",), "Real signal but too costly (the portfolio changes too much each week):",
         lambda r: -r["gross_excess_ann"]),
        (("copy",), "Too similar to something known:", lambda r: 0),
        (("unstable",), "Unstable (direction changes from year to year):", lambda r: 0),
        (("broken",), "Broken (fix these mistakes):", lambda r: 0),
    ]
    for verdicts, title, key in sections:
        rs = sorted((r for r in reviewed if r["verdict"] in verdicts), key=key)[:max_items]
        if rs:
            parts.append(title + "\n" + "\n".join(_line(r) for r in rs))
    weak = [r.get("name") or r.get("id") for r in reviewed if r["verdict"] == "weak"]
    if weak:
        parts.append("No reliable signal: " + ", ".join(weak[:12]) + ("" if len(weak) <= 12 else ", ..."))
    ls = lessons(reviewed)
    if ls:
        parts.append("Lessons:\n" + "\n".join(f"- {x}" for x in ls))
    return "\n\n".join(parts)
