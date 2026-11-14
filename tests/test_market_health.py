"""Tests for the market health score and its warning test. Synthetic data only.

Run:  python tests/test_market_health.py   (or pytest)
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))
import market_health as mh  # noqa: E402
from data import load_market  # noqa: E402
from synth_market import make_market  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="health_test_"))
CFG = make_market(TMP, name="health_t", n=60, seed=5, start="2013-01-01", end="2022-12-30")
M = load_market("health_t")


def test_score_is_bounded_and_uses_only_the_past():
    full = mh.compute(M)
    s = full["score"].dropna()
    assert len(s) > 1000 and s.between(0, 100).all()
    cut = M.dates[1500]
    import copy
    m2 = copy.copy(M)
    for f in ("open", "high", "low", "close", "volume", "member"):
        setattr(m2, f, getattr(M, f).loc[:cut])
    m2.index_close = M.index_close.loc[:cut]
    early = mh.compute(m2)["score"]
    both = early.dropna().index
    assert np.allclose(early.loc[both], full["score"].loc[both])   # adding later days never changes a past score


def test_sectors_become_a_measure():
    sect = {c: ["Banks", "IT", "Auto", "Pharma", "Energy"][i % 5] for i, c in enumerate(M.close.columns)}
    raw = mh.components(M, sect)
    assert "sectors" in raw and raw["sectors"].dropna().between(0, 1).all()
    assert "sectors" not in mh.components(M, {})


def _weeks(effect, seed):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2014-01-03", "2026-10-30", freq="W-FRI")
    score = rng.uniform(0, 100, len(dates))
    nxt = rng.normal(0.002, 0.02, len(dates)) + np.where(score < mh.LOW, effect, 0)
    d = pd.DataFrame({"score": score, "next_week": nxt, "fall_4w": nxt * 1.5 - 0.02}, index=dates)
    d["zone"] = d["score"].map(mh.zone)
    return d[d.index <= mh.DISCOVERY_END], d[d.index > mh.DISCOVERY_END]


def test_verdict_finds_a_real_warning_and_rejects_noise():
    a, b = _weeks(-0.015, 1)
    assert mh.verdict(mh.zone_table(a), mh.zone_table(b))[0] == "USEFUL WARNING"
    a, b = _weeks(0.0, 2)
    assert mh.verdict(mh.zone_table(a), mh.zone_table(b))[0] in ("NO WARNING VALUE", "WEAK HINT")


def test_half_cash_rule_arithmetic():
    dates = pd.date_range("2021-01-01", periods=40, freq="W-FRI")
    d = pd.DataFrame({"score": [10.0] * 20 + [60.0] * 20}, index=dates)
    weekly = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "net": 0.01, "bench": 0.005, "universe": 0.008,
                           "variant": "x"})
    e = mh.exam_check(d, weekly)["x"]
    assert e["weak_weeks"] == 20 and e["switches"] == 2               # into cash at the start, back once
    expected = np.r_[[0.005] * 20, [0.01] * 20] - np.r_[0.5, [0] * 19, 0.5, [0] * 19] * mh.SWITCH_COST
    assert abs(e["half_cash_when_weak"]["ann_return"] - (np.prod(1 + expected) ** (52 / 40) - 1)) < 1e-9


def test_run_writes_files_and_table():
    out = mh.run(CFG, m=M, sectors={}, log=lambda *a: None)
    assert (CFG.results / "market_health.csv").exists() and 0 <= out["score"] <= 100
    assert out["test"]["discovery"]["weeks"] > 200 and out["test"]["holdout"]["weeks"] > 50
    assert "Verdict" in mh.table(out) and len(mh.history(CFG)["dates"]) > 50


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\nall {len(tests)} tests passed")
