"""Is the edge real? Two checks on the model's weekly returns, run on weeks it had never seen.

1. Where does the return come from? (idea from tradermonty/claude-trading-skills, residual-edge-analyzer)
   The exam showed the model beat the Nifty 200 but not "hold all 200 stocks equally". So we split the
   model's weekly return into the part a simple copy explains (beta x the baseline) and what is left
   (alpha = the picking). The baselines are declared here, before looking at results:
       primary   all 200 stocks, equal weight        (the simple copy of our universe)
       alternate the Nifty 200                        (the index we report against)
       two-factor Nifty 200 + (equal weight - Nifty 200), i.e. market + size tilt
   Ordinary least squares with Newey-West (HAC) standard errors, because weekly returns overlap in
   risk and are not independent. Basis: the model AFTER costs against baselines WITHOUT costs, which
   is strict on the model (holding the baselines costs very little, but not nothing).

2. The range of outcomes (idea from jakeschincariol/day-trader-skill, the Time Machine)
   Replays the real exam weeks 10,000 times in shuffled 4-week blocks (keeping bad stretches
   together) and reports how one year could have gone: the chance of beating the index and equal
   weight, and the spread of results. A replay of the past, not a forecast.

    python edge_audit.py                 # exam weeks (results/<market>/phase6_weekly.csv)
    python edge_audit.py --live          # the revealed ledger weeks (needs 26+)
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from statistics import NormalDist

import numpy as np
import pandas as pd

WEEKS_A_YEAR = 52
SEED = 7
REPLAYS = 10_000
BLOCK = 4
MIN_WEEKS = 26
BASELINES = {
    "equal": ["universe"],
    "index": ["bench"],
    "two_factor": ["bench", "size"],
}
BASELINE_NAMES = {
    "equal": "All 200 stocks, equal weight",
    "index": "Nifty 200",
    "two_factor": "Nifty 200 + size tilt",
}


# ---------------------------------------------------------------- regression

def newey_west_lags(n: int) -> int:
    return max(1, int(math.floor(4 * (n / 100) ** (2 / 9))))


def ols_hac(y: np.ndarray, X: np.ndarray, lags: int | None = None) -> dict:
    """OLS of y on [1, X] with Newey-West standard errors (Bartlett weights)."""
    y = np.asarray(y, float)
    X = np.column_stack([np.ones(len(y)), np.asarray(X, float).reshape(len(y), -1)])
    n, k = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    lags = newey_west_lags(n) if lags is None else lags
    xe = X * e[:, None]
    S = xe.T @ xe
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        G = xe[L:].T @ xe[:-L]
        S += w * (G + G.T)
    XtX_inv = np.linalg.inv(X.T @ X)
    cov = XtX_inv @ S @ XtX_inv * n / (n - k)        # small-sample correction
    se = np.sqrt(np.diag(cov))
    t = beta / se
    p = [2 * (1 - NormalDist().cdf(abs(x))) for x in t]
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (e ** 2).sum() / ss_tot if ss_tot > 0 else float("nan")
    return {"coef": beta.tolist(), "se": se.tolist(), "t": t.tolist(), "p": p, "r2": float(r2),
            "resid": e, "lags": lags, "n": n}


def _design(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return df[cols].to_numpy(float)


def verdict(alpha_ann: float, t: float) -> tuple[str, str]:
    if alpha_ann > 0 and t >= 2:
        return "PICKING EDGE", "Picking adds a return that the simple copy doesn't explain, and it is unlikely to be luck"
    if alpha_ann > 0 and t >= 1:
        return "MAYBE", "Picking may add a little, but it could still be luck (t between 1 and 2)"
    if alpha_ann > 0:
        return "NOT PROVEN", "The leftover return is positive but too small and noisy to tell apart from luck"
    return "NO PICKING EDGE", "Once the simple copy is taken out, nothing positive is left"


def decompose(df: pd.DataFrame) -> dict:
    """df: weekly net (model after costs), bench, universe. Returns the three regressions."""
    d = df.copy()
    d["size"] = d["universe"] - d["bench"]
    out = {}
    for key, cols in BASELINES.items():
        r = ols_hac(d["net"].to_numpy(), _design(d, cols))
        a, ta = r["coef"][0], r["t"][0]
        v, why = verdict(a * WEEKS_A_YEAR, ta)
        out[key] = {
            "name": BASELINE_NAMES[key], "alpha_ann": a * WEEKS_A_YEAR, "alpha_t": ta, "alpha_p": r["p"][0],
            "betas": dict(zip(cols, r["coef"][1:])), "beta_t": dict(zip(cols, r["t"][1:])),
            "r2": r["r2"], "lags": r["lags"], "weeks": r["n"], "verdict": v, "why": why,
        }
    # how the total return over the index splits: what the copy explains vs the leftover
    eq = out["equal"]
    b = eq["betas"]["universe"]
    mean = lambda s: float(d[s].mean()) * WEEKS_A_YEAR  # noqa: E731
    out["split"] = {
        "model_ann": mean("net"), "index_ann": mean("bench"), "equal_ann": mean("universe"),
        "explained_by_equal_ann": b * mean("universe"), "alpha_ann": eq["alpha_ann"],
        "beta_equal": b,
    }
    return out


def rolling_alpha(df: pd.DataFrame, window: int = 52) -> dict:
    """Alpha against equal weight over each trailing year (annualised)."""
    dates, vals = [], []
    for i in range(window, len(df) + 1):
        g = df.iloc[i - window:i]
        X = np.column_stack([np.ones(window), g["universe"].to_numpy()])
        beta, *_ = np.linalg.lstsq(X, g["net"].to_numpy(), rcond=None)
        dates.append(str(df["date"].iloc[i - 1])[:10])
        vals.append(round(float(beta[0]) * WEEKS_A_YEAR, 5))
    return {"dates": dates, "alpha_ann": vals}


def by_period(df: pd.DataFrame) -> dict:
    """Alpha against equal weight by calendar year, and in rising vs falling markets.
    Rising/falling is the Nifty 200's own trailing 13-week move, decided before the week."""
    d = df.copy()
    d["year"] = pd.to_datetime(d["date"]).dt.year
    trail = (1 + d["bench"]).rolling(13).apply(np.prod, raw=True).shift(1) - 1
    d["market"] = np.where(trail.isna(), None, np.where(trail >= 0, "rising market", "falling market"))
    out = {"years": [], "markets": []}
    for key, col in (("years", "year"), ("markets", "market")):
        for lab, g in d.dropna(subset=[col]).groupby(col):
            if len(g) < 13:
                continue
            X = np.column_stack([np.ones(len(g)), g["universe"].to_numpy()])
            beta, *_ = np.linalg.lstsq(X, g["net"].to_numpy(), rcond=None)
            out[key].append({"label": str(lab), "weeks": int(len(g)), "alpha_ann": float(beta[0]) * WEEKS_A_YEAR,
                             "model_ann": float(g["net"].mean()) * WEEKS_A_YEAR,
                             "equal_ann": float(g["universe"].mean()) * WEEKS_A_YEAR,
                             "index_ann": float(g["bench"].mean()) * WEEKS_A_YEAR})
    return out


# ---------------------------------------------------------------- range of outcomes

def block_bootstrap(df: pd.DataFrame, replays: int = REPLAYS, horizon: int = WEEKS_A_YEAR,
                    block: int = BLOCK, seed: int = SEED) -> dict:
    """Replay the real weeks in random 4-week blocks; one year per replay. Same seed, same answer."""
    rng = np.random.default_rng(seed)
    R = df[["net", "bench", "universe"]].to_numpy(float)
    n = len(R)
    starts = rng.integers(0, n - block + 1, size=(replays, math.ceil(horizon / block)))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(replays, -1)[:, :horizon]
    paths = R[idx]                                            # replays x horizon x 3
    year = np.prod(1 + paths, axis=1) - 1                    # one-year return per replay
    curve = np.cumprod(1 + paths[:, :, 0], axis=1)
    drawdown = (curve / np.maximum.accumulate(curve, axis=1) - 1).min(axis=1)
    pct = lambda x: {str(q): float(np.percentile(x, q)) for q in (5, 25, 50, 75, 95)}  # noqa: E731
    vs_idx, vs_eq = year[:, 0] - year[:, 1], year[:, 0] - year[:, 2]
    hist_edges = np.linspace(np.percentile(vs_eq, 0.5), np.percentile(vs_eq, 99.5), 31)
    return {
        "replays": replays, "horizon_weeks": horizon, "block_weeks": block, "seed": seed, "source_weeks": n,
        "p_beat_index": float((vs_idx > 0).mean()), "p_beat_equal": float((vs_eq > 0).mean()),
        "p_loss": float((year[:, 0] < 0).mean()),
        "model": pct(year[:, 0]), "index": pct(year[:, 1]), "equal": pct(year[:, 2]),
        "vs_index": pct(vs_idx), "vs_equal": pct(vs_eq),
        "worst_drawdown": pct(drawdown),
        "hist_vs_equal": {"edges": hist_edges.round(5).tolist(),
                          "counts": np.histogram(np.clip(vs_eq, hist_edges[0], hist_edges[-1]), hist_edges)[0].tolist()},
    }


# ---------------------------------------------------------------- run

def audit(df: pd.DataFrame) -> dict:
    df = df.sort_values("date").reset_index(drop=True)
    return {"weeks": len(df), "from": str(df["date"].iloc[0])[:10], "to": str(df["date"].iloc[-1])[:10],
            "decompose": decompose(df), "rolling": rolling_alpha(df), "periods": by_period(df),
            "replay": block_bootstrap(df)}


def run(cfg, live: bool = False, p=None, log=print) -> dict:
    out = {"created_at": datetime.now(timezone.utc).isoformat(), "basis":
           "model after costs vs baselines without costs; Newey-West errors; baselines declared before results",
           "exam": {}, "live": {}}
    f = cfg.results / "edge_audit.json"
    if f.exists():
        out = {**json.loads(f.read_text()), "created_at": out["created_at"]}
    if not live:
        w = cfg.results / "phase6_weekly.csv"
        if not w.exists():
            raise SystemExit("no exam results yet: run python run_phase6.py first")
        weekly = pd.read_csv(w)
        out["exam"] = {v: audit(g[["date", "net", "bench", "universe"]]) for v, g in weekly.groupby("variant")}
        for v, a in out["exam"].items():
            e = a["decompose"]["equal"]
            log(f"  {v}: {e['alpha_ann']:+.1%} a year beyond equal weight (t {e['alpha_t']:.1f}) -> {e['verdict']}; "
                f"beat equal weight in {a['replay']['p_beat_equal']:.0%} of one-year replays")
    else:
        import ledger
        p = p or ledger.paths()
        scores = pd.read_csv(p["scores"]) if p["scores"].exists() else pd.DataFrame()
        need = {"book_return", "index_return", "equal_return"}
        if scores.empty or not need <= set(scores.columns):
            log("  no scored live weeks with an equal-weight return yet")
        else:
            for v, g in scores.groupby("variant"):
                if len(g) < MIN_WEEKS:
                    out["live"][v] = {"weeks": len(g), "waiting": f"needs {MIN_WEEKS} revealed weeks"}
                    continue
                d = g.rename(columns={"as_of": "date", "book_return": "net", "index_return": "bench",
                                      "equal_return": "universe"})[["date", "net", "bench", "universe"]]
                out["live"][v] = audit(d)
    f.write_text(json.dumps(out, indent=1, default=float))
    return out


def summary_table(audit_json: dict) -> str:
    rows = ["| Version | Model (average a year, after costs) | Nifty 200 | Equal weight | Picking edge beyond equal weight | Verdict | Beat equal weight in 1-year replays |",
            "|---|---|---|---|---|---|---|"]
    for v, a in audit_json.get("exam", {}).items():
        s, e, r = a["decompose"]["split"], a["decompose"]["equal"], a["replay"]
        rows.append(f"| {v} | {s['model_ann']:+.1%} | {s['index_ann']:+.1%} | {s['equal_ann']:+.1%} | "
                    f"{e['alpha_ann']:+.1%} (t {e['alpha_t']:.1f}) | {e['verdict']} | {r['p_beat_equal']:.0%} |")
    return "\n".join(rows)


if __name__ == "__main__":
    import markets
    import os
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--market", default=os.environ.get("LLM_ALPHA_MARKET", "india"))
    ap.add_argument("--live", action="store_true")
    a = ap.parse_args()
    res = run(markets.get(a.market), live=a.live)
    print("\n" + summary_table(res))
