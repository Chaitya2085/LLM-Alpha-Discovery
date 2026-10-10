"""Tests for the factor language. Run:  python tests/test_dsl.py   (or pytest)

Uses a small synthetic market so it needs no downloaded data.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dsl import OPS, Evaluator, FactorError, parse  # noqa: E402


def fake_market(T=150, N=40, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=T)
    cols = [f"S{i:03d}" for i in range(N)]
    close = pd.DataFrame(50 * np.exp(rng.normal(0, 0.02, (T, N)).cumsum(0)), idx, cols)
    open_ = close.shift(1).fillna(close) * (1 + rng.normal(0, 0.005, (T, N)))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.02, (T, N)))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.02, (T, N)))
    vol = pd.DataFrame(rng.lognormal(10, 0.5, (T, N)), idx, cols)
    member = pd.DataFrame(True, idx, cols)
    member.iloc[:, :3] = False  # a few non-members
    return SimpleNamespace(open=open_, high=high, low=low, close=close, volume=vol,
                           vwap=(high + low + close) / 3, member=member)


def must_fail(expr, why=""):
    try:
        parse(expr)
    except FactorError:
        return
    raise AssertionError(f"should have been rejected: {expr} {why}")


def test_rejects_unsafe_and_invalid():
    for e in ["__import__('os').system('ls')", "close.__class__", "open('x')",
              "(lambda: 1)()", "[close]", "close if volume else open", "close ** 2",
              "eval('1')", "Mean(close, d=5)", "Mean(close, 5.5)", "Mean(close, 0)",
              "Mean(close, 999)", "Mean(close)", "Foo(close)", "price", "1 + 2",
              "Mean(3, 5)", "close / 0", "True", "", "close; close"]:
        must_fail(e)


def test_complexity_limits():
    deep = "close"
    for _ in range(9):
        deep = f"Abs({deep})"
    must_fail(deep, "too deep")
    wide = " + ".join(["close"] * 20)
    must_fail(wide, "too many nodes")
    must_fail("Mean(Mean(Mean(close, 60), 60), 60)", "lookback too long")


def test_parse_roundtrip():
    n = parse("CsRank(Corr(close, volume, 20)) - 0.5 * Delta(close, 5)")
    assert n.fields() == {"close", "volume"}
    assert set(n.ops()) == {"CsRank", "Corr", "Delta"}
    assert n.max_lookback() == 20
    assert str(parse(str(n))) == str(n)


def test_canonical_text_is_minimal_and_equivalent():
    m = fake_market()
    ev = Evaluator(m)
    cases = {
        "-Decay(close / vwap - 1, 10)": "-Decay(close / vwap - 1, 10)",
        "((close - open)) / (high - low)": "(close - open) / (high - low)",
        "close - (open - low)": "close - (open - low)",
        "close - open - low": "close - open - low",
        "close / (open * low)": "close / (open * low)",
        "-(close - open)": "-(close - open)",
        "close * -returns": "close * (-returns)",
        "2.0 * Mean(close, 5)": "2 * Mean(close, 5)",
    }
    for src, want in cases.items():
        got = str(parse(src))
        assert got == want, (src, got)
        a, b = ev(src).values, ev(got).values
        assert np.allclose(a, b, equal_nan=True), f"meaning changed: {src} -> {got}"


def test_operators_match_pandas():
    m = fake_market()
    ev = Evaluator(m)
    c, v, mem = m.close, m.volume, m.member
    pairs = {
        "Mean(close, 10)": c.rolling(10).mean(),
        "Std(close, 10)": c.rolling(10).std(),
        "Delta(close, 3)": c - c.shift(3),
        "Pct(close, 5)": c / c.shift(5) - 1,
        "Ref(volume, 2)": v.shift(2),
        "TsMax(high, 7)": m.high.rolling(7).max(),
        "Corr(close, volume, 15)": c.rolling(15).corr(v),
        "Skew(returns, 20)": (c / c.shift(1) - 1).rolling(20).skew(),
        "Kurt(returns, 20)": (c / c.shift(1) - 1).rolling(20).kurt(),
        "Log(volume)": np.log1p(v),
        "returns": c / c.shift(1) - 1,
    }
    for e, want in pairs.items():
        got = ev(e)
        want = want.where(mem)
        ok = np.allclose(got.values, want.values, equal_nan=True, rtol=1e-6, atol=1e-9)
        assert ok, f"mismatch for {e}"
    # cross-sectional rank uses members only
    r = ev("CsRank(close)")
    assert r.iloc[:, :3].isna().all().all()
    assert np.allclose(r.iloc[-1].dropna().max(), 1.0)
    # IdxMax: 0 when today is the max
    up = m.close.copy() * 0 + np.arange(len(m.close))[:, None]
    m2 = fake_market(); m2.close = up
    assert (Evaluator(m2)("IdxMax(close, 10)").iloc[20:, 5] == 0).all()
    # Slope of a straight line
    assert np.allclose(Evaluator(m2)("Slope(close, 10)").iloc[20:, 5], 1.0)


def test_no_lookahead():
    """Changing data after day k must not change any factor value on days <= k."""
    m = fake_market()
    k = 100
    exprs = [f"{name}(close, 10)" if OPS[name].n_series == 1 and OPS[name].window else None
             for name in OPS]
    exprs = [e for e in exprs if e] + [
        "Corr(close, volume, 10)", "Cov(returns, volume, 10)", "CsRank(Delta(close, 5))",
        "CsZscore(volume)", "CsDemean(returns)", "Max(close, open)", "Sign(returns)",
        "Abs(returns)", "Sqrt(volume)", "Log(volume)",
    ]
    base = Evaluator(m)
    m2 = fake_market()
    for f in ["open", "high", "low", "close", "volume", "vwap"]:
        df = getattr(m2, f).copy()
        df.iloc[k + 1:] *= 3.7
        setattr(m2, f, df)
    shocked = Evaluator(m2)
    for e in exprs:
        a = base(e).iloc[: k + 1].values
        b = shocked(e).iloc[: k + 1].values
        assert np.allclose(a, b, equal_nan=True), f"look-ahead detected in {e}"


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\nall {len(tests)} tests passed")
