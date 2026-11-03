"""Statistics that keep a backtest honest about luck (standard library only).

  psr(returns, sr_star)   Probabilistic Sharpe Ratio: the probability that the true Sharpe ratio is
                          above sr_star, given the sample's length, skew and fat tails
                          (Bailey and Lopez de Prado, 2012).
  expected_max_sr(...)    the best Sharpe ratio you'd expect by luck alone after trying N strategies
                          whose Sharpe ratios vary by `var` (the "false strategy" benchmark).
  dsr(returns, trials)    Deflated Sharpe Ratio = PSR against that luck benchmark: the probability
                          the edge is real AFTER counting how many things were tried
                          (Bailey and Lopez de Prado, 2014).

All Sharpe ratios here are per period (per week for weekly returns), not annualised.
"""
from __future__ import annotations

import math
from statistics import NormalDist

EULER = 0.5772156649015329
N01 = NormalDist()


def moments(x: list[float]) -> tuple[float, float, float, float]:
    """mean, std (sample), skew, kurtosis (not excess; normal = 3)."""
    n = len(x)
    mu = sum(x) / n
    d = [v - mu for v in x]
    m2 = sum(v * v for v in d) / n
    if m2 == 0:
        return mu, 0.0, 0.0, 3.0
    m3 = sum(v ** 3 for v in d) / n
    m4 = sum(v ** 4 for v in d) / n
    sd = math.sqrt(m2 * n / (n - 1))
    return mu, sd, m3 / m2 ** 1.5, m4 / m2 ** 2


def sharpe(x: list[float]) -> float:
    mu, sd, _, _ = moments(x)
    return mu / sd if sd > 0 else 0.0


def psr(x: list[float], sr_star: float = 0.0) -> float:
    n = len(x)
    if n < 3:
        return float("nan")
    mu, sd, skew, kurt = moments(x)
    if sd == 0:
        return float("nan")
    sr = mu / sd
    denom = 1 - skew * sr + (kurt - 1) / 4 * sr ** 2
    if denom <= 0:
        return float("nan")
    return N01.cdf((sr - sr_star) * math.sqrt(n - 1) / math.sqrt(denom))


def expected_max_sr(n_trials: int, var_sr: float) -> float:
    """Expected best per-period Sharpe among n_trials strategies with no real edge."""
    if n_trials <= 1 or var_sr <= 0:
        return 0.0
    a = N01.inv_cdf(1 - 1 / n_trials)
    b = N01.inv_cdf(1 - 1 / (n_trials * math.e))
    return math.sqrt(var_sr) * ((1 - EULER) * a + EULER * b)


def dsr(x: list[float], n_trials: int, var_sr: float) -> float:
    return psr(x, expected_max_sr(n_trials, var_sr))


def t_stat(x: list[float]) -> float:
    mu, sd, _, _ = moments(x)
    return mu / sd * math.sqrt(len(x)) if sd > 0 else float("nan")


def paired_t(a: list[float], b: list[float]) -> float:
    return t_stat([u - v for u, v in zip(a, b)])
