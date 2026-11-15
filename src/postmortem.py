"""The Critic's post-mortems: after a sealed week is revealed and scored, why did it go the way it did?

Idea adapted from tradermonty/claude-trading-skills (signal-postmortem; MIT). The code is our own.
Every week the ledger reveals (31+ days later) gets one post-mortem, written once and never edited:

    result       the book against the Nifty 200 and against all 200 stocks equally
    ranking      did higher-ranked stocks do better across all 200? (rank correlation, top 40 vs bottom 40)
    sectors      which sector bets helped or hurt (weight vs the universe, and picking within the sector)
    reasons      the model's most-used reasons that week, and whether the stocks they favoured did better
    context      market mood and market health at the time
    picks        the three best and three worst picks (the book is public once revealed)

Once 8 weeks are in, it also looks for patterns: how often the book beat the index in each market
mood and health zone. These are descriptions to learn from, not rules: nothing here changes the frozen
model.

    python postmortem.py        # writes results/<market>/postmortems.json
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

MIN_PATTERN_WEEKS = 8


def _tag(variant: str) -> str:
    return "llm" if "LLM" in variant else "a158"


def sector_effects(book: pd.Series, ranked: pd.Series, ret: pd.Series, sectors: dict) -> list[dict]:
    """Brinson-style split per sector: allocation (over/underweight x how the sector did vs all 200)
    and selection (did our stocks beat the rest of their sector)."""
    lab = pd.Series({c: sectors.get(str(c).upper(), "Unknown") for c in ranked})
    r = ret.reindex(ranked)
    ok = r.notna()
    lab, r = lab[ok], r[ok]
    in_book = lab.index.isin(book)
    if in_book.sum() == 0:
        return []
    r_all = r.mean()
    rows = []
    for s, idx in lab.groupby(lab).groups.items():
        wu = len(idx) / len(lab)
        b_idx = [c for c in idx if c in set(book)]
        wb = len(b_idx) / in_book.sum()
        rs_u = r[idx].mean()
        rs_b = r[b_idx].mean() if b_idx else rs_u
        rows.append({"sector": s, "book_weight": wb, "universe_weight": wu, "sector_return": float(rs_u),
                     "allocation": float((wb - wu) * (rs_u - r_all)), "selection": float(wb * (rs_b - rs_u)),
                     "stocks_in_book": len(b_idx)})
    for x in rows:
        x["total"] = x["allocation"] + x["selection"]
    return sorted(rows, key=lambda x: x["total"], reverse=True)


def one(as_of: str, variant: str, ranked_df: pd.DataFrame, fwd5: pd.DataFrame, bench5: pd.Series,
        sectors: dict, why: pd.DataFrame | None, mood: str | None, health: dict | None) -> dict | None:
    d = pd.Timestamp(as_of)
    if d not in fwd5.index:
        return None
    ranked_df = ranked_df.sort_values("p", ascending=False).reset_index(drop=True)
    ret = fwd5.loc[d]
    ranked = ranked_df["code"].astype(str)
    r = ret.reindex(ranked).to_numpy(float)
    ok = ~np.isnan(r)
    if ok.sum() < 20:
        return None
    book = ranked_df.loc[ranked_df["in_book"], "code"].astype(str)
    rb = ret.reindex(book).dropna()
    eq, idx = float(np.nanmean(r)), float(bench5.get(d, np.nan))
    p = ranked_df["p"].to_numpy(float)
    ic = float(pd.Series(p[ok]).rank().corr(pd.Series(r[ok]).rank()))
    n = len(book)
    top, bottom = np.nanmean(r[:n]), np.nanmean(r[-n:])
    sec = sector_effects(book, ranked, ret, sectors) if sectors else []
    picks = rb.sort_values()
    out = {
        "as_of": as_of, "variant": variant,
        "result": {"book": float(rb.mean()), "index": idx, "equal": eq,
                   "beat_index": bool(rb.mean() > idx), "beat_equal": bool(rb.mean() > eq)},
        "ranking": {"rank_ic": ic, "top_minus_bottom": float(top - bottom), "stocks": int(ok.sum()), "book_size": n},
        "context": {"mood": mood, "health": health},
        "best": [{"code": c, "return": float(v)} for c, v in picks.iloc[::-1].head(3).items()],
        "worst": [{"code": c, "return": float(v)} for c, v in picks.head(3).items()],
        "sectors": {"helped": [s for s in sec if s["total"] > 0][:2], "hurt": [s for s in sec if s["total"] < 0][-2:][::-1]},
        "reasons": [],
    }
    if why is not None and len(why):
        w = why[why["code"].astype(str).isin(set(book)) & ~why["feature"].str.startswith("mkt_")]
        use = w.groupby("feature")["contribution"].apply(lambda x: x.abs().sum()).sort_values(ascending=False).head(3)
        from team import explain
        for feat in use.index:
            g = w[w["feature"] == feat].set_index("code")["contribution"]
            pushed = g[g > 0].index
            rp = ret.reindex(pushed).mean() if len(pushed) else np.nan
            out["reasons"].append({"feature": feat, "meaning": explain.describe(feat)[1], "stocks": int(len(pushed)),
                                   "book_stocks": int(len(rb)),
                                   "favoured_return": None if np.isnan(rp) else float(rp),
                                   "book_return": float(rb.mean())})
    out["notes"] = notes(out)
    return out


def _p(x: float) -> str:
    return f"{x:+.2%}"


def notes(pm: dict) -> list[str]:
    """The Critic's plain sentences, built only from the numbers above."""
    r, k = pm["result"], pm["ranking"]
    s = [f"The book returned {_p(r['book'])} against {_p(r['index'])} for the Nifty 200 and {_p(r['equal'])} "
         f"for all the ranked stocks equally: it {'beat' if r['beat_index'] else 'trailed'} the index and "
         f"{'beat' if r['beat_equal'] else 'trailed'} equal weight."]
    nb = k.get("book_size", 40)
    if k["rank_ic"] > 0.05:
        s.append(f"The ranking worked across all {k['stocks']} stocks: higher-ranked stocks did better "
                 f"(rank correlation {k['rank_ic']:+.2f}; top {nb} minus bottom {nb} {_p(k['top_minus_bottom'])}).")
    elif k["rank_ic"] < -0.05:
        s.append(f"The ranking worked backwards this week: lower-ranked stocks did better "
                 f"(rank correlation {k['rank_ic']:+.2f}; top {nb} minus bottom {nb} {_p(k['top_minus_bottom'])}).")
    else:
        s.append(f"The ranking barely mattered this week (rank correlation {k['rank_ic']:+.2f}).")
    for x in pm["sectors"]["helped"][:1]:
        why = "being overweight" if x["allocation"] > x["selection"] else "picking well within it"
        s.append(f"{x['sector']} helped most ({_p(x['total'])}), mainly from {why}.")
    for x in pm["sectors"]["hurt"][:1]:
        why = "the sector weight" if abs(x["allocation"]) > abs(x["selection"]) else "the picks within it"
        s.append(f"{x['sector']} hurt most ({_p(x['total'])}), mainly from {why}.")
    for x in pm["reasons"][:1]:
        if x["favoured_return"] is not None:
            gap = x["favoured_return"] - x["book_return"]
            if x["stocks"] >= x.get("book_stocks", 0):
                how = f"it pushed up every stock in the book, which returned {_p(x['book_return'])}"
            else:
                how = ("about the same as" if abs(gap) < 0.0005 else "better than" if gap > 0 else "worse than") + \
                      " the rest of the book"
                how = f"the {x['stocks']} stocks it favoured returned {_p(x['favoured_return'])}, {how}"
            s.append(f"The most-used reason was \"{x['meaning']}\"; {how}.")
    c = pm["context"]
    if c.get("mood") or c.get("health"):
        h = c.get("health") or {}
        s.append("Market then: " + ", ".join(x for x in [c.get("mood"),
                 f"health {h['score']:.0f}/100 ({h['zone']})" if h else None] if x) + ".")
    return s


def patterns(pms: list[dict]) -> dict:
    out = {}
    for v in {p["variant"] for p in pms}:
        g = [p for p in pms if p["variant"] == v]
        if len(g) < MIN_PATTERN_WEEKS:
            out[v] = {"weeks": len(g), "waiting": f"patterns start at {MIN_PATTERN_WEEKS} weeks"}
            continue
        df = pd.DataFrame([{"mood": p["context"]["mood"] or "unknown",
                            "zone": (p["context"]["health"] or {}).get("zone", "unknown"),
                            "beat_index": p["result"]["beat_index"], "beat_equal": p["result"]["beat_equal"],
                            "ic": p["ranking"]["rank_ic"]} for p in g])
        out[v] = {"weeks": len(g), "avg_rank_ic": float(df["ic"].mean()),
                  "by_mood": {k: {"weeks": int(len(x)), "beat_index": float(x["beat_index"].mean()),
                                  "beat_equal": float(x["beat_equal"].mean())} for k, x in df.groupby("mood")},
                  "by_health": {k: {"weeks": int(len(x)), "beat_index": float(x["beat_index"].mean()),
                                    "beat_equal": float(x["beat_equal"].mean())} for k, x in df.groupby("zone")}}
    return out


def run(cfg, p=None, m=None, sectors: dict | None = None, log=print) -> dict:
    import ledger
    import phase5_model as p5
    p = p or ledger.paths()
    f = cfg.results / "postmortems.json"
    store = json.loads(f.read_text()) if f.exists() else {"weeks": []}
    done = {(x["as_of"], x["variant"]) for x in store["weeks"]}
    todo = [c for c in ledger.read_commits(p) if (c["as_of"], c["variant"]) not in done
            and (p["revealed"] / c["file"]).exists()]
    if todo:
        if m is None:
            from data import load_market
            m = load_market(cfg.name)
        if sectors is None:
            try:
                import risk_lists
                sectors = risk_lists.sectors()
            except Exception:  # noqa: BLE001
                sectors = {}
        fwd5, bench5 = p5.labels(m)
        reg = p5.regimes(m)["regime"]
        hf = cfg.results / "market_health.csv"
        hh = pd.read_csv(hf, parse_dates=["date"]).set_index("date") if hf.exists() else None
        for c in todo:
            d = pd.Timestamp(c["as_of"])
            ranked = pd.read_csv(p["revealed"] / c["file"])
            why_f = p["private"] / f"{c['as_of']}_{_tag(c['variant'])}_why.csv"
            why = pd.read_csv(why_f) if why_f.exists() else None
            mood = reg.get(d) if d in reg.index else None
            health = None
            if hh is not None and d in hh.index and pd.notna(hh.at[d, "score"]):
                health = {"score": float(hh.at[d, "score"]), "zone": str(hh.at[d, "zone"])}
            pm = one(c["as_of"], c["variant"], ranked, fwd5, bench5, sectors, why,
                     None if mood is None or (isinstance(mood, float) and np.isnan(mood)) else str(mood), health)
            if pm is None:
                continue                                # the week's returns aren't complete yet
            pm["written_at"] = datetime.now(timezone.utc).isoformat()
            store["weeks"].append(pm)
            log(f"  post-mortem {c['as_of']} {c['variant']}: {pm['notes'][0]}")
    store["weeks"].sort(key=lambda x: (x["as_of"], x["variant"]))
    store["patterns"] = patterns(store["weeks"])
    cfg.results.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(store, indent=1, default=float))
    return store


def table(store: dict) -> str | None:
    if not store.get("weeks"):
        return None
    rows = ["| Week after | Version | Book | Nifty 200 | Equal weight | Ranking worked? | The Critic |", "|---|---|---|---|---|---|---|"]
    for pm in store["weeks"][-8:][::-1]:
        r, k = pm["result"], pm["ranking"]
        rows.append(f"| {pm['as_of']} | {pm['variant']} | {_p(r['book'])} | {_p(r['index'])} | {_p(r['equal'])} | "
                    f"{k['rank_ic']:+.2f} | {pm['notes'][2] if len(pm['notes']) > 2 else ''} |")
    return "\n".join(rows)


if __name__ == "__main__":
    import markets
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--market", default=os.environ.get("LLM_ALPHA_MARKET", "india"))
    s = run(markets.get(ap.parse_args().market))
    print(table(s) or "  no revealed weeks yet (the first opens 31 days after it was sealed)")
