"""Tests for Phase 6 (final exam, honest statistics, leakage test, prediction ledger), on synthetic
markets. No internet or API keys needed.

Run:  python tests/test_phase6.py   (or pytest)
"""
import json
import math
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))
import generate_factors  # noqa: E402
import honest_stats as hs  # noqa: E402
import leakage_test  # noqa: E402
import ledger  # noqa: E402
import phase5_model as p5  # noqa: E402
import phase6_exam as p6  # noqa: E402
from synth_market import make_market  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="phase6_test_"))
QUIET = dict(log=lambda *a: None)


def _library(root, factors, provider="fake", model="m"):
    generate_factors.LIB = root / "library.jsonl"
    recs = generate_factors.ingest(json.dumps({"factors": [
        {"name": n, "category": "volume", "expression": e, "hypothesis": "x"} for n, e in factors]}),
        {"provider": provider, "model": model, "prompt_version": "t", "run_file": "t", "created_at": "t"},
        generate_factors.load_library())
    generate_factors.append_library(recs)
    return recs


def test_honest_stats():
    rng = np.random.default_rng(0)
    x = list(rng.normal(0.1, 1.0, 5000))
    sr = hs.sharpe(x)
    assert abs(hs.psr(x, 0.0) - hs.N01.cdf(sr * math.sqrt(len(x) - 1))) < 0.01
    assert hs.expected_max_sr(100, 0.01) > hs.expected_max_sr(10, 0.01) > hs.expected_max_sr(2, 0.01) > 0
    small = list(rng.normal(0.05, 1.0, 200))
    assert hs.dsr(small, 16, 0.02) < hs.psr(small, 0.0)          # counting the tries lowers confidence
    frozen = {"pass_marks": p6.PASS_MARKS, "trials": {"count": 16, "weekly_sharpe_variance": 0.002}}
    assert p6.verdict(list(rng.normal(0.01, 0.02, 300)), frozen)["verdict"] == "REAL EDGE"
    assert p6.verdict(list(rng.normal(-0.002, 0.02, 300)), frozen)["verdict"] == "NO PROVEN EDGE"


def test_exam_is_frozen_one_shot_and_finds_a_real_edge():
    root = TMP / "exam"
    cfg = make_market(root, name="exam6", n=100, seed=5, start="2014-01-01", end="2023-06-30")
    recs = _library(root, [("liquidity", "Mean(volume, 20)"), ("noise", "-Delta(close, 3)")])
    cfg.results.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"id": r["id"], "name": r["name"], "verdict": "keep"} for r in recs]).to_csv(
        cfg.results / "phase4_factors.csv", index=False)
    p5.run(cfg, [2018, 2019, 2020], top=20, threads=1, use_a158=False, **QUIET)
    p6.FROZEN = root / "config" / "phase6_frozen.json"
    f = p6.freeze(cfg)
    assert f["variants"] == ["LLM factors only"] and f["trials"]["count"] == 8
    assert {x["id"] for x in f["llm_factors"]} == {r["id"] for r in recs}
    s = p6.run(cfg, threads=1, **QUIET)
    v = s["variants"]["LLM factors only"]
    assert s["years"][0] == 2021 and v["book"]["weeks"] > 100
    assert v["verdict"]["verdict"] == "REAL EDGE", v["verdict"]
    w = pd.read_csv(cfg.results / "phase6_weekly.csv", parse_dates=["date"])
    assert w["date"].min() >= pd.Timestamp("2021-01-01")
    try:
        p6.run(cfg, threads=1, **QUIET)
        raise AssertionError("a second run without a reason must be refused")
    except SystemExit as e:
        assert "already run" in str(e)
    p6.run(cfg, reason="test rerun", threads=1, **QUIET)
    log = json.loads((cfg.results / "phase6_runlog.json").read_text())
    assert [r["reason"] for r in log] == ["first run", "test rerun"]


def test_leakage_test_flags_a_remembered_effect():
    root = TMP / "leak"
    cut = "2022-01-01"
    cfg = make_market(root, name="leak6", n=100, seed=6, start="2014-01-01", end="2023-06-30",
                      drift_until=cut)               # the effect really stops at the 'cutoff'
    generate_factors.LIB = root / "library.jsonl"
    _library(root, [("a", "Mean(volume, 20)"), ("b", "Mean(volume, 10)"), ("c", "EMA(volume, 15)")],
             provider="fake", model="mem")
    _library(root, [("d", "-Delta(close, 3)"), ("e", "Std(close, 10)"), ("f", "-Delta(close, 7)")],
             provider="fake", model="honest")
    s = leakage_test.run(cfg, cutoffs={"fake: mem": pd.Timestamp(cut)}, **QUIET)
    mem = s["models"]["fake: mem"]
    assert mem["status"] == "MEMORY SIGNAL", mem
    assert mem["ic_before"] > 0.05 and abs(mem["ic_after"]) < 0.05
    assert s["models_without_cutoff"] == ["fake: honest"]
    s2 = leakage_test.run(cfg, cutoffs={"fake: honest": pd.Timestamp(cut)}, **QUIET)
    assert s2["models"]["fake: honest"]["status"] == "no memory signal", s2


def test_ledger_commit_reveal_verify():
    p = ledger.paths(TMP / "ledger")
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(3):
        ledger.commit_file(p, f"2026-01-0{i + 1}", "Alpha158 only", f"code,p\nS00{i},0.5\n".encode(), "fp",
                           now=t0 + timedelta(days=i))
    assert ledger.verify(p) == []
    assert ledger.reveal(p, now=t0 + timedelta(days=10)) == []                 # too soon
    done = ledger.reveal(p, now=t0 + timedelta(days=31, hours=12))
    assert done == ["2026-01-01_a158.csv"]
    (p["private"] / "2026-01-02_a158.csv").write_text("code,p\nS999,0.9\n")    # someone edits a past call
    assert any("changed after it was committed" in x for x in ledger.verify(p))
    rows = (p["commits"]).read_text().replace("2026-01-03,", "2026-01-04,", 1)  # back-dating an entry
    p["commits"].write_text(rows)
    assert any("fingerprint" in x for x in ledger.verify(p))


def test_ledger_records_the_frozen_model_once_per_day():
    root = TMP / "exam"                         # reuse the exam market + frozen settings
    cfg = make_market(root, name="exam6", n=100, seed=5, start="2014-01-01", end="2023-06-30")
    generate_factors.LIB = root / "library.jsonl"
    ledger.FROZEN = root / "config" / "phase6_frozen.json"
    p = ledger.paths(root / "ledger")
    rows = ledger.record(cfg, threads=1, p=p, **QUIET)
    assert len(rows) == 1 and rows[0]["as_of"] == "2023-06-30"
    book = pd.read_csv(p["private"] / rows[0]["file"])
    assert book["in_book"].sum() == 40 and book["p"].between(0, 1).all()
    assert ledger.record(cfg, threads=1, p=p, **QUIET) == []               # same day: nothing new
    assert ledger.verify(p) == []


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\nall {len(tests)} tests passed")
