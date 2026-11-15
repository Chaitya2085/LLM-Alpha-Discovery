"""Tests for update 24: the Critic's post-mortems, the lab's robustness check, and the paper journal.
Synthetic market only; no internet, no API keys.

Run:  python tests/test_update24.py   (or pytest)
"""
import json
import sys
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))
import ledger  # noqa: E402
import paper  # noqa: E402
import postmortem  # noqa: E402
import research_lab  # noqa: E402
from data import load_market  # noqa: E402
from synth_market import make_market  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="u24_test_"))
CFG = make_market(TMP / "mk", name="u24", n=60, seed=8, start="2016-01-01", end="2021-06-30")
M = load_market("u24")


def _seal_and_reveal(as_of: str, perfect: bool):
    """A recorded week whose ranking is either perfect (ranked by the real next-week return) or random."""
    import phase5_model as p5
    p = ledger.paths(TMP / f"ledger_{as_of}_{perfect}")
    fwd5, _ = p5.labels(M)
    r = fwd5.loc[pd.Timestamp(as_of)].dropna()
    order = r.sort_values(ascending=False).index if perfect else r.sample(frac=1, random_state=1).index
    df = pd.DataFrame({"as_of": as_of, "code": order, "p": np.linspace(0.6, 0.4, len(order)),
                       "rank_pct": np.linspace(1, 0, len(order)), "in_book": [i < 10 for i in range(len(order))]})
    ledger.commit_file(p, as_of, "Alpha158 only", df.to_csv(index=False).encode(), "test",
                       now=datetime.now(timezone.utc) - timedelta(days=40))
    pd.DataFrame({"code": order[:10].repeat(2), "feature": ["a158_ROC5", "a158_STD20"] * 10,
                  "rank_pct": 0.8, "contribution": [0.1, -0.02] * 10}).to_csv(p["private"] / f"{as_of}_a158_why.csv", index=False)
    assert ledger.reveal(p)
    return p


def test_postmortem_reads_a_good_and_a_bad_week():
    day = str(M.dates[-60].date())
    sect = {c: ["Banks", "IT", "Auto", "Pharma"][i % 4] for i, c in enumerate(M.close.columns)}
    good = postmortem.run(CFG, p=_seal_and_reveal(day, True), m=M, sectors=sect, log=lambda *a: None)
    pm = next(w for w in good["weeks"] if w["as_of"] == day)
    assert pm["ranking"]["rank_ic"] > 0.9 and pm["result"]["beat_equal"] and "worked across all" in pm["notes"][1]
    assert pm["best"][0]["return"] >= pm["worst"][0]["return"] and pm["reasons"][0]["feature"] == "a158_ROC5"
    assert pm["sectors"]["helped"] or pm["sectors"]["hurt"]
    again = postmortem.run(CFG, p=ledger.paths(TMP / f"ledger_{day}_True"), m=M, sectors=sect, log=lambda *a: None)
    assert len(again["weeks"]) == len(good["weeks"])                 # written once, never rewritten
    bday = str(M.dates[-80].date())
    bad = postmortem.run(CFG, p=_seal_and_reveal(bday, False), m=M, sectors={}, log=lambda *a: None)
    assert abs(next(w for w in bad["weeks"] if w["as_of"] == bday)["ranking"]["rank_ic"]) < 0.5
    assert postmortem.table(bad) and "patterns start" in json.dumps(bad["patterns"])


def test_window_variants():
    v = research_lab.variants("-Pct(close, 20) * Mean(volume, 60) / Mean(volume, 20)")
    assert [x[1] for x in v][0] == "-Pct(close, 10) * Mean(volume, 30) / Mean(volume, 10)"
    assert all("60)" in e or "40)" in e or "30)" in e for _, e in v)    # clamped at 60
    assert research_lab.variants("close / open") == []


def test_robustness_rule():
    class FakeLab:
        def __init__(self, ts):
            self.ts = ts
        def measure(self, expr):  # noqa: E301
            return {"judge": {"status": "scored", "weekly_ic_t": self.ts.pop(0), "net_excess_ann": 0.01}}
    rec = {"expression": "Mean(close, 20)", "weekly_ic_t": 4.0}
    assert research_lab.robustness(FakeLab([3.5, 2.5, 1.0]), rec)["status"] == "robust"     # 2 of 3 hold
    assert research_lab.robustness(FakeLab([3.5, -2.0, 1.0]), rec)["status"] == "fragile"   # 1 of 3


def test_paper_rules():
    folder = TMP / "team"
    code = str(M.close.columns[0])
    try:
        paper.open_position(folder, M, code, 50000, "too short")
        raise AssertionError("a short reason should be refused")
    except paper.Refused as e:
        assert "why" in str(e)
    try:
        paper.open_position(folder, M, code, 300000, "a long enough reason to open this one")
        raise AssertionError("over 20% should be refused")
    except paper.Refused as e:
        assert "20%" in str(e)
    t = paper.open_position(folder, M, code, 100000, "a long enough reason to open this one", stop_pct=0.05,
                            plan_weeks=1, in_book=True, rank=3)
    acct = paper.mark(paper.load(folder), M)
    assert len(acct["open"]) == 1 and acct["open"][0]["return"] == 0 and acct["gate"]["allowed"]
    paper.close_position(folder, M, t["id"], "done")
    acct = paper.mark(paper.load(folder), M)
    assert acct["digest"]["closed"] == 1 and "in the model's book when opened" in acct["digest"]["by_source"]


def test_paper_pauses_after_losses_and_drawdown():
    book = {"start_balance": 1_000_000.0, "trades": [
        {"id": str(i), "code": "X", "status": "closed", "opened_on": "2026-11-01", "closed_on": "2026-11-10",
         "entry_price": 100, "exit_price": 95, "amount": 1000} for i in range(3)]}
    g = paper.gate(book, 990_000, 1_000_000, today=date(2026, 11, 12))
    assert not g["allowed"] and "cooling off" in g["reasons"][0]
    assert paper.gate(book, 990_000, 1_000_000, today=date(2026, 11, 20))["allowed"]
    assert not paper.gate({"start_balance": 1e6, "trades": []}, 880_000, 1_000_000)["allowed"]   # 12% below the high


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\nall {len(tests)} tests passed")
