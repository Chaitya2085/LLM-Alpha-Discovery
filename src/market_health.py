"""Market health: a 0-100 score for how broad and healthy the NSE market is, from our own data, and an
honest test of whether a low score would have warned of bad weeks.

Idea adapted from tradermonty/claude-trading-skills (market-breadth-analyzer, uptrend-analyzer,
exposure-coach, market-top-detector; MIT). Those read US breadth files; this computes the same kind
of measures for the NSE top 200 from the prices we already have. The code is our own.

Six measures, each known at the close of the day (no future data):
    above_200   share of the 200 stocks above their 200-day average     (long-term participation)
    above_50    share above their 50-day average                        (medium-term participation)
    adv_dec     stocks rising minus falling, 10-day average              (day-to-day breadth)
    highs_lows  new 52-week highs minus new lows, 10-day average         (leadership)
    sectors     share of sectors whose typical stock is above its 50-day average
    index_trend the Nifty 200 against its 200-day average                 (the index itself)
Each is turned into "how high is this compared with every earlier day" (0-100, using only the past),
and the score is their average.

Zones: under 30 weak, 30-70 middle, over 70 strong. These cut-offs are fixed here, before the test.

The test (would it have warned?): every week, the score at Friday's close against what the Nifty 200
did next (the next week, and the worst fall over the next 4 weeks), in 2014-2020 and again in
2021-2026 separately. On the exam weeks it also checks the model's own results by zone and a simple
pre-declared rule (hold half the book in cash while the score is under 30). The team treats the score
as context only, unless the test shows it would really have helped in both periods.

    python market_health.py              # writes results/<market>/market_health.csv and .json
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

LOW, HIGH = 30, 70
MIN_HISTORY = 252
DISCOVERY_END = "2020-12-31"
COMPONENTS = {
    "above_200": "Stocks above their 200-day average",
    "above_50": "Stocks above their 50-day average",
    "adv_dec": "Stocks rising minus falling (10-day)",
    "highs_lows": "New 52-week highs minus lows (10-day)",
    "sectors": "Sectors in an uptrend",
    "index_trend": "Nifty 200 against its 200-day average",
}
SWITCH_COST = 0.0025          # moving half the book in or out of cash: about one round trip on half


def zone(score: float | None) -> str:
    if score is None or np.isnan(score):
        return "unknown"
    return "weak" if score < LOW else "strong" if score > HIGH else "middle"


# ---------------------------------------------------------------- the score

def components(m, sectors: dict | None = None) -> pd.DataFrame:
    """The raw daily measures, from prices only (point-in-time membership)."""
    c = m.close.where(m.member)
    n = m.member.sum(axis=1).replace(0, np.nan)
    ma200 = m.close.rolling(200, min_periods=150).mean()
    ma50 = m.close.rolling(50, min_periods=40).mean()
    ret = m.close.pct_change(fill_method=None).where(m.member)
    hi252 = m.close.rolling(252, min_periods=200).max()
    lo252 = m.close.rolling(252, min_periods=200).min()
    out = pd.DataFrame(index=m.dates)
    out["above_200"] = (c > ma200).where(ma200.notna() & m.member).sum(axis=1) / n
    out["above_50"] = (c > ma50).where(ma50.notna() & m.member).sum(axis=1) / n
    out["adv_dec"] = (((ret > 0).sum(axis=1) - (ret < 0).sum(axis=1)) / n).rolling(10).mean()
    out["highs_lows"] = ((((c >= hi252) & hi252.notna()).sum(axis=1)
                          - ((c <= lo252) & lo252.notna()).sum(axis=1)) / n).rolling(10).mean()
    if sectors:
        up = (c > ma50).where(ma50.notna() & m.member)
        lab = pd.Series({col: sectors.get(str(col).upper()) for col in up.columns}).dropna()
        if lab.nunique() >= 4:
            per = pd.DataFrame({s: up[cols].mean(axis=1) for s, cols in lab.groupby(lab).groups.items()})
            out["sectors"] = (per > 0.5).where(per.notna()).mean(axis=1)
    idx = m.index_close.reindex(m.dates).ffill()
    out["index_trend"] = idx / idx.rolling(200, min_periods=150).mean() - 1
    # the first ~200 days have no averages yet
    out.loc[out.index[:MIN_HISTORY // 2], ["above_200", "highs_lows", "index_trend"]] = np.nan
    return out


def score(raw: pd.DataFrame) -> pd.DataFrame:
    """Each measure against its own earlier history (0-100), and the average of them."""
    out = pd.DataFrame(index=raw.index)
    for col in raw.columns:
        out[col] = raw[col].expanding(min_periods=MIN_HISTORY).rank(pct=True) * 100
    out["score"] = out[list(raw.columns)].mean(axis=1, skipna=True).where(out[list(raw.columns)].notna().sum(axis=1) >= 4)
    out["zone"] = out["score"].map(zone)
    return out


def compute(m, sectors: dict | None = None) -> pd.DataFrame:
    raw = components(m, sectors)
    s = score(raw)
    return pd.concat([s, raw.add_prefix("raw_")], axis=1)


# ---------------------------------------------------------------- would it have warned?

def weekly_outcomes(m, health: pd.DataFrame) -> pd.DataFrame:
    """One row per week (the last trading day): the score then, and what the index did next."""
    import phase5_model as p5
    _, bench5 = p5.labels(m)
    idx = m.index_close.reindex(m.dates).ffill()
    fwd_min = idx[::-1].rolling(20, min_periods=20).min()[::-1].shift(-1)       # lowest close in the next 20 days
    dd4 = fwd_min / idx - 1
    last = pd.Series(m.dates, index=m.dates).groupby(m.dates.to_period("W-FRI")).max()
    d = pd.DataFrame({"score": health["score"], "next_week": bench5, "fall_4w": dd4}).loc[last.values]
    d = d.dropna(subset=["score", "next_week"])
    d["zone"] = d["score"].map(zone)
    return d


def _diff(y: pd.Series, low: pd.Series) -> dict:
    """Mean of y in weak weeks minus the other weeks, with Newey-West t."""
    from edge_audit import ols_hac
    if low.sum() < 8 or (~low).sum() < 8:
        return {"diff": None, "t": None}
    r = ols_hac(y.to_numpy(float), low.astype(float).to_numpy())
    return {"diff": float(r["coef"][1]), "t": float(r["t"][1])}


def zone_table(d: pd.DataFrame) -> dict:
    rows = {}
    for z in ("weak", "middle", "strong"):
        g = d[d["zone"] == z]
        rows[z] = {"weeks": int(len(g)),
                   "next_week": float(g["next_week"].mean()) if len(g) else None,
                   "bad_weeks": float((g["next_week"] < -0.02).mean()) if len(g) else None,
                   "fall_4w": float(g["fall_4w"].mean()) if len(g) else None}
    low = d["zone"] == "weak"
    return {"zones": rows, "weak_vs_rest_next_week": _diff(d["next_week"], low),
            "weak_vs_rest_fall_4w": _diff(d["fall_4w"].dropna(), low[d["fall_4w"].notna()])}


def exam_check(d: pd.DataFrame, weekly: pd.DataFrame) -> dict:
    """The model's exam weeks by zone, and the declared rule: half in cash while the score is under 30."""
    out = {}
    for v, g in weekly.groupby("variant"):
        g = g.copy()
        g["date"] = pd.to_datetime(g["date"])
        s = d["score"].reindex(g["date"], method="ffill", tolerance=pd.Timedelta(days=6))
        g["score"] = s.to_numpy()
        g = g.dropna(subset=["score"])
        if len(g) < 26:
            continue
        g["zone"] = g["score"].map(zone)
        low = g["zone"] == "weak"
        hold = np.where(low, 0.5, 1.0)
        switches = np.abs(np.diff(np.r_[1.0, hold]))
        rule = hold * g["net"].to_numpy() - switches * SWITCH_COST
        def stats(r):  # noqa: E306
            r = np.asarray(r, float)
            curve = np.cumprod(1 + r)
            return {"ann_return": float(curve[-1] ** (52 / len(r)) - 1),
                    "worst_fall": float((curve / np.maximum.accumulate(curve) - 1).min()),
                    "sharpe": float(r.mean() / r.std() * np.sqrt(52)) if r.std() > 0 else None}
        out[v] = {
            "weeks": int(len(g)), "weak_weeks": int(low.sum()),
            "excess_vs_index": {z: float((g.loc[g["zone"] == z, "net"] - g.loc[g["zone"] == z, "bench"]).mean())
                                if (g["zone"] == z).any() else None for z in ("weak", "middle", "strong")},
            "excess_vs_equal": {z: float((g.loc[g["zone"] == z, "net"] - g.loc[g["zone"] == z, "universe"]).mean())
                                if (g["zone"] == z).any() else None for z in ("weak", "middle", "strong")},
            "always": stats(g["net"]), "half_cash_when_weak": stats(rule), "switches": int(switches.sum() * 2),
        }
    return out


def verdict(disc: dict, hold: dict) -> tuple[str, str]:
    """Fixed rules. A warning has to show up in BOTH periods, for the next week or the next 4 weeks."""
    a, b = disc["weak_vs_rest_next_week"], hold["weak_vs_rest_next_week"]
    fa, fb = disc["weak_vs_rest_fall_4w"], hold["weak_vs_rest_fall_4w"]
    if a["t"] is None or b["t"] is None:
        return "NOT ENOUGH WEAK WEEKS", "Too few weak weeks in one of the periods to test it"
    week_worse = a["diff"] < 0 and b["diff"] < 0
    falls_deeper = fa["diff"] is not None and fb["diff"] is not None and fa["diff"] < 0 and fb["diff"] < 0
    if (week_worse and max(a["t"], b["t"]) <= -2) or (falls_deeper and max(fa["t"], fb["t"]) <= -2):
        what = "worse next weeks" if week_worse and max(a["t"], b["t"]) <= -2 else "deeper falls over the next 4 weeks"
        return "USEFUL WARNING", f"Weak scores came before {what} in both periods, beyond luck"
    parts = []
    if week_worse:
        parts.append("worse next weeks")
    if falls_deeper:
        parts.append("deeper falls over the next 4 weeks")
    if parts and ((week_worse and max(a["t"], b["t"]) < -1) or (falls_deeper and max(fa["t"], fb["t"]) < -1)):
        why = f"Weak scores came before {' and '.join(parts)} in both periods, but not beyond luck"
        if not week_worse:
            why += "; the very next week was not worse in both periods"
        return "WEAK HINT", why
    return "NO WARNING VALUE", "A weak score did not reliably come before worse weeks; treat it as a description, not a warning"


def run(cfg, m=None, sectors: dict | None = None, log=print) -> dict:
    if m is None:
        from data import load_market
        m = load_market(cfg.name)
    if sectors is None:
        try:
            import risk_lists
            sectors = risk_lists.sectors()
        except Exception:  # noqa: BLE001 - the score works without sectors
            sectors = {}
    h = compute(m, sectors)
    cfg.results.mkdir(parents=True, exist_ok=True)
    h.round(5).to_csv(cfg.results / "market_health.csv", index_label="date")
    d = weekly_outcomes(m, h)
    disc, hold = d[d.index <= DISCOVERY_END], d[d.index > DISCOVERY_END]
    test = {"discovery": {"from": str(disc.index.min().date()) if len(disc) else None, "weeks": len(disc), **zone_table(disc)},
            "holdout": {"from": str(hold.index.min().date()) if len(hold) else None, "weeks": len(hold), **zone_table(hold)}}
    v, why = verdict(test["discovery"], test["holdout"])
    w6 = cfg.results / "phase6_weekly.csv"
    exam = exam_check(d, pd.read_csv(w6)) if w6.exists() else {}
    latest = h.dropna(subset=["score"]).iloc[-1]
    used = [c for c in COMPONENTS if c in h.columns and pd.notna(latest.get(c))]
    out = {"created_at": datetime.now(timezone.utc).isoformat(), "as_of": str(latest.name.date()),
           "score": float(latest["score"]), "zone": latest["zone"], "low": LOW, "high": HIGH,
           "components": {c: {"name": COMPONENTS[c], "score": float(latest[c]), "raw": float(latest[f"raw_{c}"])}
                          for c in used},
           "verdict": v, "why": why, "test": test, "exam": exam,
           "note": "Cut-offs (30/70) and the half-cash rule were fixed before the test. Sectors use today's "
                   "Nifty 200 sector list for all years." if "sectors" in used else
                   "Cut-offs (30/70) and the half-cash rule were fixed before the test. No sector list yet "
                   "(python src/risk_lists.py), so 5 measures are used."}
    (cfg.results / "market_health.json").write_text(json.dumps(out, indent=1, default=float))
    log(f"  market health {out['score']:.0f}/100 ({out['zone']}) on {out['as_of']}; "
        f"would it have warned? {v.lower()}")
    return out


def history(cfg, every: int = 5, years: int = 3) -> dict:
    f = cfg.results / "market_health.csv"
    if not f.exists():
        return {}
    h = pd.read_csv(f, parse_dates=["date"]).dropna(subset=["score"])
    h = h[h["date"] >= h["date"].max() - pd.Timedelta(days=365 * years)].iloc[::-1].iloc[::every].iloc[::-1]
    return {"dates": h["date"].dt.strftime("%Y-%m-%d").tolist(), "score": h["score"].round(1).tolist()}


def table(j: dict) -> str:
    rows = ["| Period | Weeks | Weak weeks (under 30): Nifty 200 next week | Other weeks | Difference (t) | Worst 4-week fall, weak vs other |",
            "|---|---|---|---|---|---|"]
    for k, lab in (("discovery", "2014–2020"), ("holdout", "2021–now")):
        t = j["test"][k]
        z = t["zones"]
        other = [z[x] for x in ("middle", "strong") if z[x]["weeks"]]
        ow = sum(x["weeks"] for x in other)
        om = sum(x["next_week"] * x["weeks"] for x in other) / ow if ow else None
        of = sum(x["fall_4w"] * x["weeks"] for x in other) / ow if ow else None
        dd = t["weak_vs_rest_next_week"]
        f = lambda x: "–" if x is None else f"{x:+.2%}"  # noqa: E731
        tt = "–" if dd["t"] is None else f"{dd['t']:.1f}"
        rows.append(f"| {lab} | {t['weeks']} | {f(z['weak']['next_week'])} ({z['weak']['weeks']} weeks) | {f(om)} | "
                    f"{f(dd['diff'])} ({tt}) | {f(z['weak']['fall_4w'])} vs {f(of)} |")
    rows += ["", f"Today: **{j['score']:.0f}/100 ({j['zone']})** on {j['as_of']}. Verdict: **{j['verdict']}**: {j['why']}."]
    for v, e in j.get("exam", {}).items():
        a, h = e["always"], e["half_cash_when_weak"]
        rows.append(f"- {v} (exam weeks): always invested {a['ann_return']:+.1%} a year, worst fall {a['worst_fall']:.0%}; "
                    f"half in cash when weak {h['ann_return']:+.1%}, worst fall {h['worst_fall']:.0%} "
                    f"({e['weak_weeks']} weak weeks of {e['weeks']}).")
    return "\n".join(rows)


if __name__ == "__main__":
    import markets
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--market", default=os.environ.get("LLM_ALPHA_MARKET", "india"))
    res = run(markets.get(ap.parse_args().market))
    print("\n" + table(res))
