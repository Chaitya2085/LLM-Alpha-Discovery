"""The prediction ledger: proof that predictions were made BEFORE the week they predict.

    python ledger.py record     # this week's predictions -> private file + public fingerprint
    python ledger.py reveal     # publish predictions older than 31 days (fingerprint checked)
    python ledger.py score      # how revealed weeks actually went vs the Nifty 200
    python ledger.py verify     # re-check every fingerprint and the chain

How it works (commit, then reveal)
  * record: trains the frozen Phase 6 model on all data up to the latest close, ranks every stock,
    picks the book with the frozen design, and writes it to ledger/private/<date>.csv. That folder is
    never committed (it names stocks, and SEBI's rules keep live stock calls out of public view
    without registration). What IS committed is ledger/commitments.csv: the file's SHA-256
    fingerprint, chained to the previous entry, so no entry can be changed or back-dated without
    breaking every fingerprint after it. Git's history timestamps each one.
  * reveal: once an entry is more than 31 days old (SEBI's 30-day rule for educational use of price
    data), its file is copied to ledger/revealed/ after checking it matches its fingerprint.
  * score: each revealed week's picks against the Nifty 200 over the same 5 trading days.

This is a research record, not advice. Get a compliance review before promoting results publicly.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import markets
import phase5_model as p5

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "ledger"
FROZEN = ROOT / "config" / "phase6_frozen.json"
REVEAL_AFTER_DAYS = 31
FIELDS = ["as_of", "variant", "file", "sha256", "prev_hash", "entry_hash", "created_at", "fingerprint"]


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def entry_hash(prev: str, file_sha: str, as_of: str, variant: str) -> str:
    return sha(f"{prev}|{file_sha}|{as_of}|{variant}".encode())


def paths(root: Path = LEDGER):
    return {"root": root, "private": root / "private", "revealed": root / "revealed",
            "commits": root / "commitments.csv", "scores": root / "scores.csv"}


def read_commits(p) -> list[dict]:
    if not p["commits"].exists():
        return []
    with p["commits"].open() as f:
        return list(csv.DictReader(f))


def append_commit(p, row: dict) -> None:
    new = not p["commits"].exists()
    with p["commits"].open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def commit_file(p, as_of: str, variant: str, body: bytes, fingerprint: str, now: datetime | None = None) -> dict:
    """Store a prediction privately and append its chained fingerprint publicly."""
    p["private"].mkdir(parents=True, exist_ok=True)
    tag = "llm" if "LLM" in variant else "a158"
    name = f"{as_of}_{tag}.csv"
    (p["private"] / name).write_bytes(body)
    commits = read_commits(p)
    prev = commits[-1]["entry_hash"] if commits else "genesis"
    fs = sha(body)
    row = {"as_of": as_of, "variant": variant, "file": name, "sha256": fs, "prev_hash": prev,
           "entry_hash": entry_hash(prev, fs, as_of, variant),
           "created_at": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
           "fingerprint": fingerprint}
    append_commit(p, row)
    return row


def verify(p) -> list[str]:
    """Problems found (empty list = everything checks out)."""
    problems, prev = [], "genesis"
    for c in read_commits(p):
        if c["prev_hash"] != prev:
            problems.append(f"{c['file']}: chain broken (prev_hash does not match the entry before)")
        if entry_hash(c["prev_hash"], c["sha256"], c["as_of"], c["variant"]) != c["entry_hash"]:
            problems.append(f"{c['file']}: entry fingerprint does not match its contents")
        for folder in ("revealed", "private"):
            f = p[folder] / c["file"]
            if f.exists() and sha(f.read_bytes()) != c["sha256"]:
                problems.append(f"{folder}/{c['file']}: file was changed after it was committed")
        prev = c["entry_hash"]
    return problems


def reveal(p, now: datetime | None = None) -> list[str]:
    now = now or datetime.now(timezone.utc)
    p["revealed"].mkdir(parents=True, exist_ok=True)
    done = []
    for c in read_commits(p):
        src, dst = p["private"] / c["file"], p["revealed"] / c["file"]
        if dst.exists() or not src.exists():
            continue
        if now - datetime.fromisoformat(c["created_at"]) < timedelta(days=REVEAL_AFTER_DAYS):
            continue
        if sha(src.read_bytes()) != c["sha256"]:
            raise SystemExit(f"{c['file']} does not match its committed fingerprint; not revealing it")
        shutil.copyfile(src, dst)
        done.append(c["file"])
    return done


# ---------------------------------------------------------------- predictions

def predict_latest(m, X: pd.DataFrame, y: pd.Series, y_rank: pd.Series, label_end: pd.Series,
                   threads: int = 2, return_model: bool = False):
    """Train on everything known at the latest close; rank the latest day's stocks.
    With return_model, also returns the model and the latest day's inputs (for explanations)."""
    import lightgbm as lgb
    d = X.index.get_level_values("date")
    le = label_end.reindex(d).to_numpy()
    last = m.dates[-1]
    known = pd.Series(le < pd.Timestamp("2262-01-01"), index=X.index) & y.reindex(X.index).notna().to_numpy()
    known_days = d[known.to_numpy()]
    v0 = known_days.max() - pd.Timedelta(days=365)
    yt, yv = y_rank.reindex(X.index), y.reindex(X.index)
    tr = known.to_numpy() & (le < v0) & (d >= pd.Timestamp(p5.TRAIN_START))
    va = known.to_numpy() & (d >= v0)
    model = lgb.train({**p5.PARAMS, "num_threads": threads}, lgb.Dataset(X.values[tr], yt.values[tr].astype(float)),
                      num_boost_round=800, valid_sets=[lgb.Dataset(X.values[va], yt.values[va].astype(float))],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
    raw_va = model.predict(X.values[va], num_iteration=model.best_iteration)
    cal = p5.platt_fit(p5.week_pct(raw_va, d[va]), yv.values[va].astype(float))
    today = d == last
    raw = model.predict(X.values[today], num_iteration=model.best_iteration)
    pct = p5.week_pct(raw, d[today])
    out = pd.DataFrame({"code": X.index.get_level_values("code")[today], "p": cal(pct), "rank_pct": pct + 0.5})
    out = out.sort_values("p", ascending=False).reset_index(drop=True)
    if return_model:
        x_today = pd.DataFrame(X.values[today], columns=X.columns, index=X.index.get_level_values("code")[today])
        return out, model, x_today
    return out


def reasons(model, x_today: pd.DataFrame, codes, top: int = 5) -> pd.DataFrame:
    """The inputs that pushed each stock's score most (LightGBM's own per-input contributions)."""
    xs = x_today.reindex(list(codes))
    contrib = model.predict(xs.values, pred_contrib=True, num_iteration=model.best_iteration)[:, :-1]
    rows = []
    for i, code in enumerate(xs.index):
        order = np.argsort(-np.abs(contrib[i]))[:top]
        for j in order:
            col = xs.columns[j]
            v = float(xs.iat[i, j])
            rows.append({"code": code, "feature": col, "rank_pct": v + 0.5 if not col.startswith("mkt_") else v,
                         "contribution": float(contrib[i, j])})
    return pd.DataFrame(rows)


def choose_book(ranked: pd.DataFrame, held: set, stocks: int, keep_zone: int) -> pd.DataFrame:
    keep = set(ranked["code"].head(keep_zone))
    kept = [c for c in ranked["code"] if c in held and c in keep]
    fresh = [c for c in ranked["code"] if c not in set(kept)]
    picks = (kept + fresh)[:stocks]
    r = ranked.copy()
    r["in_book"] = r["code"].isin(picks)
    r["kept_from_last_week"] = r["code"].isin(kept)
    return r


def record(cfg, threads: int = 2, p=None, log=print) -> list[dict]:
    p = p or paths()
    frozen = json.loads(FROZEN.read_text())
    m = p5.load_cut(cfg, "2262-01-01")
    as_of = str(m.dates[-1].date())
    if any(c["as_of"] == as_of for c in read_commits(p)):
        log(f"  already recorded for {as_of}; download newer data first (python run_india.py)")
        return []
    fwd5, bench5, y, y_rank, yex_long, label_end = p5.targets(m)
    X_all = p5.build_features(m, cfg, frozen["llm_factors"], use_a158=True)
    a158 = X_all[[c for c in X_all.columns if not c.startswith("llm_")]]
    rows = []
    for variant in frozen["variants"]:
        X = X_all if "LLM" in variant else a158
        d = frozen["designs"][variant]
        ranked, model, x_today = predict_latest(m, X, y, y_rank, label_end, threads, return_model=True)
        tag = "llm" if "LLM" in variant else "a158"
        prev = sorted(p["private"].glob(f"*_{tag}.csv")) if p["private"].exists() else []
        held = set(pd.read_csv(prev[-1]).query("in_book")["code"]) if prev else set()
        book = choose_book(ranked, held, d["stocks"], d["keep_zone"])
        book.insert(0, "as_of", as_of)
        body = book.to_csv(index=False, float_format="%.5f").encode()
        rows.append(commit_file(p, as_of, variant, body, frozen["fingerprint"]))
        reasons(model, x_today, book.loc[book["in_book"], "code"]).to_csv(
            p["private"] / f"{as_of}_{tag}_why.csv", index=False, float_format="%.5f")
        log(f"  {variant}: {int(book['in_book'].sum())} stocks recorded for the week after {as_of} "
            f"(fingerprint {rows[-1]['sha256'][:12]})")
    return rows


def rebuild_reasons(cfg, as_of: str, threads: int = 2, p=None, log=print) -> list[str]:
    """Write the missing <as_of>_<tag>_why.csv for books recorded before reasons were saved.
    Retrains the same frozen model on the same data (fixed seed), so the reasons match the book.
    Never touches the ledger itself. Only possible while the latest data day is still `as_of`."""
    p = p or paths()
    frozen = json.loads(FROZEN.read_text())
    m = p5.load_cut(cfg, "2262-01-01")
    if str(m.dates[-1].date()) != as_of:
        log(f"  can't rebuild reasons for {as_of}: the data now runs to {m.dates[-1].date()}")
        return []
    fwd5, bench5, y, y_rank, yex_long, label_end = p5.targets(m)
    X_all = p5.build_features(m, cfg, frozen["llm_factors"], use_a158=True)
    a158 = X_all[[c for c in X_all.columns if not c.startswith("llm_")]]
    done = []
    for variant in frozen["variants"]:
        tag = "llm" if "LLM" in variant else "a158"
        book_f, why_f = p["private"] / f"{as_of}_{tag}.csv", p["private"] / f"{as_of}_{tag}_why.csv"
        if not book_f.exists() or why_f.exists():
            continue
        X = X_all if "LLM" in variant else a158
        _, model, x_today = predict_latest(m, X, y, y_rank, label_end, threads, return_model=True)
        book = pd.read_csv(book_f)
        reasons(model, x_today, book.loc[book["in_book"], "code"]).to_csv(why_f, index=False, float_format="%.5f")
        done.append(why_f.name)
        log(f"  {variant}: reasons saved for the book of {as_of}")
    return done


def score(cfg, p=None) -> pd.DataFrame:
    p = p or paths()
    from data import load_market
    m = load_market(cfg.name)
    fwd5, bench5 = p5.labels(m)
    out = []
    for c in read_commits(p):
        f = p["revealed"] / c["file"]
        d = pd.Timestamp(c["as_of"])
        if not f.exists() or d not in fwd5.index or pd.isna(bench5.get(d)):
            continue
        picks = pd.read_csv(f).query("in_book")["code"]
        r = fwd5.loc[d].reindex(picks)
        if r.notna().sum() == 0:
            continue
        out.append({"as_of": c["as_of"], "variant": c["variant"], "stocks": int(r.notna().sum()),
                    "book_return": float(r.mean()), "index_return": float(bench5[d]),
                    "excess": float(r.mean() - bench5[d]), "beat_index": bool(r.mean() > bench5[d])})
    s = pd.DataFrame(out)
    if len(s):
        s.to_csv(p["scores"], index=False)
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["record", "reveal", "score", "verify", "why"])
    ap.add_argument("--market", default=os.environ.get("LLM_ALPHA_MARKET", "india"))
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    cfg = markets.get(a.market)
    os.environ["LLM_ALPHA_MARKET"] = cfg.name
    p = paths()
    if a.action == "record":
        if not FROZEN.exists():
            raise SystemExit("Freeze Phase 6 first:  python phase6_exam.py --freeze")
        record(cfg, a.threads, p)
    elif a.action == "why":
        commits = read_commits(p)
        if not commits:
            raise SystemExit("nothing recorded yet")
        rebuild_reasons(cfg, commits[-1]["as_of"], a.threads, p)
    elif a.action == "reveal":
        done = reveal(p)
        print(f"  revealed {len(done)} entr{'y' if len(done) == 1 else 'ies'}" + (": " + ", ".join(done) if done else ""))
    elif a.action == "score":
        s = score(cfg, p)
        if len(s):
            print(s.groupby("variant")[["excess", "beat_index"]].mean())
        else:
            print("  nothing to score yet (entries are revealed 31 days after they are recorded)")
    else:
        probs = verify(p)
        print("  ledger checks out" if not probs else "\n".join("  PROBLEM: " + x for x in probs))
        if probs:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
