"""Tests for Phase 5 (prediction model, calibration, weekly books), on a synthetic market with
one planted effect (heavily traded stocks drift up). No internet or API keys needed.

Run:  python tests/test_phase5.py   (or pytest)
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))
import generate_factors  # noqa: E402
import phase5_model as p5  # noqa: E402
from synth_market import make_market  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="phase5_test_"))
YEARS = [2018, 2019, 2020]


def _setup(name, drift):
    root = TMP / name
    cfg = make_market(root, name=name, n=100, seed=4, start="2014-01-01", end="2021-06-30", drift_size=drift)
    # the 'library' and a Phase 4 result that passed: the planted effect, plus a noise factor
    generate_factors.LIB = root / "library.jsonl"
    recs = generate_factors.ingest(json.dumps({"factors": [
        {"name": "liquidity", "category": "volume", "expression": "Mean(volume, 20)", "hypothesis": "x"},
        {"name": "noise", "category": "reversal", "expression": "-Delta(close, 3)", "hypothesis": "x"}]}),
        {"provider": "fake", "model": "m", "prompt_version": "t", "run_file": "t", "created_at": "t"}, [])
    generate_factors.append_library(recs)
    cfg.results.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"id": r["id"], "name": r["name"], "verdict": "keep"} for r in recs]).to_csv(
        cfg.results / "phase4_factors.csv", index=False)
    return cfg


def test_platt_recovers_a_known_curve():
    rng = np.random.default_rng(0)
    x = rng.uniform(-0.5, 0.5, 200_000)
    p_true = 1 / (1 + np.exp(-(-0.2 + 0.8 * x)))
    y = (rng.random(len(x)) < p_true).astype(float)
    f = p5.platt_fit(x, y)
    assert np.abs(f(x) - p_true).max() < 0.01


def test_model_finds_the_effect_is_calibrated_and_never_sees_later_years():
    cfg = _setup("signal", 0.0006)
    s = p5.run(cfg, YEARS, top=20, min_prob=0.5, threads=1, use_a158=False, log=lambda *a: None)
    v = s["variants"]["LLM factors only"]
    assert v["weekly_rank_ic"] > 0.05 and v["weekly_rank_ic_t"] > 3, v
    assert v["books"]["Top 20"]["excess_ann"] > 0.03, v["books"]
    assert v["calibration"]["ece"] < 0.05, v["calibration"]
    cal = pd.read_csv(cfg.results / "phase5_calibration.csv")
    assert cal["actual"].iloc[-1] > cal["actual"].iloc[0], cal      # more confident -> right more often
    imp = pd.read_csv(cfg.results / "phase5_importance.csv", index_col=0)["gain"]
    assert imp.idxmax() == "llm_F0001", imp.head()                   # it relies on the real effect
    pred = pd.read_parquet(cfg.results / "phase5_predictions.parquet")
    assert pred.index.get_level_values("date").max() <= pd.Timestamp(p5.CUTOFF)

    # leakage check: scramble every price after the cutoff -> identical predictions
    px = pd.read_parquet(cfg.proc / "prices.parquet")
    after = px.index.get_level_values("date") > pd.Timestamp(p5.CUTOFF)
    rng = np.random.default_rng(1)
    for c in ["open", "high", "low", "close", "volume", "vwap", "raw_close"]:
        px.loc[after, c] = px.loc[after, c].to_numpy() * rng.uniform(0.2, 5, after.sum())
    px.to_parquet(cfg.proc / "prices.parquet")
    p5.run(cfg, YEARS, top=20, min_prob=0.5, threads=1, use_a158=False, log=lambda *a: None)
    pred2 = pd.read_parquet(cfg.results / "phase5_predictions.parquet")
    pd.testing.assert_frame_equal(pred, pred2)


def test_no_effect_means_no_skill():
    cfg = _setup("noise", 0.0)
    s = p5.run(cfg, YEARS, top=20, min_prob=0.5, threads=1, use_a158=False, log=lambda *a: None)
    v = s["variants"]["LLM factors only"]
    assert abs(v["weekly_rank_ic_t"]) < 3, v
    assert v["calibration"]["brier_skill"] < 0.01, v["calibration"]


def test_refuses_held_out_years():
    sys.argv = ["x", "--years", "2019-2021"]
    try:
        p5.main()
        raise AssertionError("should refuse to test on 2021")
    except SystemExit as e:
        assert "Phase 6" in str(e)


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\nall {len(tests)} tests passed")
