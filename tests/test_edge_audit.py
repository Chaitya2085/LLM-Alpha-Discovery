"""Tests for the edge audit (is the edge real? and the range of outcomes). Synthetic returns only.

Run:  python tests/test_edge_audit.py   (or pytest)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
import edge_audit as ea  # noqa: E402


def weeks(n=300, alpha=0.0, beta=0.8, noise=0.006, seed=1):
    rng = np.random.default_rng(seed)
    bench = rng.normal(0.002, 0.02, n)
    size = rng.normal(0.0005, 0.01, n)
    universe = bench + size
    net = alpha / 52 + beta * universe + rng.normal(0, noise, n)
    return pd.DataFrame({"date": pd.date_range("2021-01-01", periods=n, freq="W-FRI").strftime("%Y-%m-%d"),
                         "net": net, "bench": bench, "universe": universe})


def test_hac_matches_plain_ols_without_autocorrelation():
    d = weeks()
    r = ea.ols_hac(d["net"], d[["universe"]], lags=0)
    X = np.column_stack([np.ones(len(d)), d["universe"]])
    b = np.linalg.lstsq(X, d["net"], rcond=None)[0]
    e = d["net"] - X @ b
    white = np.sqrt(np.diag(np.linalg.inv(X.T @ X) @ (X.T * e.to_numpy() ** 2) @ X @ np.linalg.inv(X.T @ X)
                            * len(d) / (len(d) - 2)))
    assert np.allclose(r["coef"], b) and np.allclose(r["se"], white)
    assert abs(r["coef"][1] - 0.8) < 0.05


def test_finds_a_planted_edge_and_not_a_fake_one():
    real = ea.decompose(weeks(alpha=0.08))["equal"]
    assert real["verdict"] == "PICKING EDGE" and 0.05 < real["alpha_ann"] < 0.11 and real["alpha_t"] > 2
    none = ea.decompose(weeks(alpha=0.0, seed=3))["equal"]
    assert none["verdict"] in ("NOT PROVEN", "NO PICKING EDGE", "MAYBE") and abs(none["alpha_t"]) < 2


def test_beating_the_index_by_size_alone_is_not_picking():
    # the model is just the universe (all stocks equally): it beats the index, but has no picking edge
    d = weeks(alpha=0.0, beta=1.0, noise=0.001, seed=5)
    out = ea.decompose(d)
    assert abs(out["equal"]["alpha_ann"]) < 0.01 and out["equal"]["verdict"] != "PICKING EDGE"
    assert abs(out["two_factor"]["betas"]["size"] - 1.0) < 0.05


def test_replays_are_repeatable_and_sensible():
    d = weeks(alpha=0.05)
    a, b = ea.block_bootstrap(d, replays=2000), ea.block_bootstrap(d, replays=2000)
    assert a == b                                                     # same seed, same answer
    assert 0 <= a["p_beat_equal"] <= 1 and a["model"]["5"] < a["model"]["50"] < a["model"]["95"]
    assert sum(a["hist_vs_equal"]["counts"]) == 2000
    worse = ea.block_bootstrap(weeks(alpha=-0.05), replays=2000)
    assert a["p_beat_equal"] > worse["p_beat_equal"]


def test_full_audit_shapes():
    a = ea.audit(weeks(n=160))
    assert a["weeks"] == 160 and len(a["rolling"]["dates"]) == 160 - 52 + 1
    assert {p["label"] for p in a["periods"]["markets"]} <= {"rising market", "falling market"}
    assert "|" in ea.summary_table({"exam": {"x": a}})


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\nall {len(tests)} tests passed")
