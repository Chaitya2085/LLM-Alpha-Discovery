"""A small, safe language for alpha factors.

The LLM writes factors as expressions such as

    CsRank(Corr(close, volume, 20)) - CsRank(Delta(close, 5))

and this module parses and evaluates them on wide (date x stock) frames.

Why a restricted language instead of letting the LLM write Python:
  * Safety: expressions are parsed with Python's `ast` and only whitelisted
    nodes are allowed (numbers, field names, + - * /, unary minus, and calls to
    the operators below). No attribute access, imports, lambdas or builtins,
    so nothing the LLM writes can touch the file system or network.
  * No look-ahead by construction: every time-series operator only looks
    backwards (window ending at day t). tests/test_dsl.py checks this by
    changing future data and confirming past factor values don't move.
  * Complexity control: node count, nesting depth and windows are bounded,
    which limits overfitting and keeps factors readable.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from features import _roll_apply, _rolling_corr, _slope_rsq_resi

FIELDS = ("open", "high", "low", "close", "volume", "vwap", "returns")
MAX_WINDOW = 60
MAX_NODES = 30
MAX_DEPTH = 8
EPS = 1e-12


class FactorError(ValueError):
    """Raised for any expression that is invalid, unsafe or too complex."""


# ---------------------------------------------------------------- operators
@dataclass(frozen=True)
class Op:
    fn: Callable
    n_series: int          # number of series (expression) arguments
    window: bool           # last argument is an integer window
    kind: str              # "elementwise" | "timeseries" | "cross_section"
    doc: str
    min_window: int = 1


def _ts(fn):
    return lambda x, d: fn(x, d)


def _decay_linear(x: pd.DataFrame, d: int) -> pd.DataFrame:
    w = np.arange(1, d + 1, dtype="float64")
    w /= w.sum()
    out = sum(w[d - 1 - k] * x.shift(k) for k in range(d))
    return out.where(x.notna())


def _ts_rank(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(d, min_periods=d).rank(pct=True)


def _window_moments(v: np.ndarray):
    """Central moments of each window (last axis); NaN if the window has any gap."""
    n = v.shape[-1]
    mu = v.mean(axis=-1, keepdims=True)
    c = v - mu
    m2 = (c ** 2).sum(-1) / n
    m3 = (c ** 3).sum(-1) / n
    m4 = (c ** 4).sum(-1) / n
    return n, m2, m3, m4


def _skew_fn(v):
    n, m2, m3, _ = _window_moments(v)
    g1 = m3 / np.where(m2 > EPS, m2, np.nan) ** 1.5
    return np.sqrt(n * (n - 1)) / (n - 2) * g1          # same bias-corrected skew as pandas


def _kurt_fn(v):
    n, m2, _, m4 = _window_moments(v)
    g2 = m4 / np.where(m2 > EPS, m2, np.nan) ** 2 - 3
    return ((n + 1) * g2 + 6) * (n - 1) / ((n - 2) * (n - 3))   # same excess kurtosis as pandas


def _skew(x, d):
    # exact per-window maths: pandas' rolling skew/kurt use whole-series sums for
    # stability, which lets later data nudge earlier values in some versions
    return _roll_apply(x, d, _skew_fn)


def _kurt(x, d):
    return _roll_apply(x, d, _kurt_fn)


def _slope(x, d):
    return _slope_rsq_resi(x, d)[0]


def _rsquare(x, d):
    return _slope_rsq_resi(x, d)[1]


def _resi(x, d):
    return _slope_rsq_resi(x, d)[2]


def _idxmax(x, d):
    # NaN anywhere in the window -> NaN (nanargmax would silently skip gaps)
    am = _roll_apply(x, d, lambda v: np.where(np.isnan(v).any(-1), np.nan,
                                              np.argmax(np.nan_to_num(v, nan=-np.inf), axis=-1)))
    return (d - 1 - am) / d


def _idxmin(x, d):
    am = _roll_apply(x, d, lambda v: np.where(np.isnan(v).any(-1), np.nan,
                                              np.argmin(np.nan_to_num(v, nan=np.inf), axis=-1)))
    return (d - 1 - am) / d


def _safe_div(a, b):
    b = b if np.isscalar(b) else b.where(b.abs() > EPS)
    if np.isscalar(b) and abs(b) <= EPS:
        raise FactorError("division by zero constant")
    return a / b


OPS: dict[str, Op] = {
    # element-wise
    "Abs": Op(lambda x: x.abs(), 1, False, "elementwise", "absolute value"),
    "Log": Op(lambda x: np.sign(x) * np.log1p(x.abs()), 1, False, "elementwise",
              "sign(x)*log(1+|x|), safe for negatives"),
    "Sign": Op(lambda x: np.sign(x), 1, False, "elementwise", "-1, 0 or 1"),
    "Sqrt": Op(lambda x: np.sign(x) * np.sqrt(x.abs()), 1, False, "elementwise", "signed square root"),
    "Max": Op(lambda a, b: np.maximum(a, b), 2, False, "elementwise", "larger of two values"),
    "Min": Op(lambda a, b: np.minimum(a, b), 2, False, "elementwise", "smaller of two values"),
    # time-series (per stock, window of the last d days including today)
    "Ref": Op(lambda x, d: x.shift(d), 1, True, "timeseries", "value d days ago"),
    "Delta": Op(lambda x, d: x - x.shift(d), 1, True, "timeseries", "x - Ref(x, d)"),
    "Pct": Op(lambda x, d: _safe_div(x, x.shift(d)) - 1, 1, True, "timeseries", "x / Ref(x, d) - 1"),
    "Mean": Op(lambda x, d: x.rolling(d, min_periods=d).mean(), 1, True, "timeseries", "rolling mean", 2),
    "Sum": Op(lambda x, d: x.rolling(d, min_periods=d).sum(), 1, True, "timeseries", "rolling sum", 2),
    "Std": Op(lambda x, d: x.rolling(d, min_periods=d).std(), 1, True, "timeseries", "rolling std dev", 2),
    "TsMax": Op(lambda x, d: x.rolling(d, min_periods=d).max(), 1, True, "timeseries", "rolling max", 2),
    "TsMin": Op(lambda x, d: x.rolling(d, min_periods=d).min(), 1, True, "timeseries", "rolling min", 2),
    "TsRank": Op(_ts_rank, 1, True, "timeseries", "percentile of today's value in last d days", 3),
    "Skew": Op(_skew, 1, True, "timeseries", "rolling skewness", 5),
    "Kurt": Op(_kurt, 1, True, "timeseries", "rolling kurtosis", 5),
    "EMA": Op(lambda x, d: x.ewm(span=d, min_periods=d, adjust=False).mean(), 1, True, "timeseries",
              "exponential moving average with span d", 2),
    "Decay": Op(_decay_linear, 1, True, "timeseries", "linearly weighted mean, recent days heavier", 2),
    "Slope": Op(_slope, 1, True, "timeseries", "OLS slope of x over the last d days", 3),
    "Rsquare": Op(_rsquare, 1, True, "timeseries", "R^2 of linear trend over last d days", 3),
    "Resi": Op(_resi, 1, True, "timeseries", "today's residual from linear trend", 3),
    "IdxMax": Op(_idxmax, 1, True, "timeseries", "days since the d-day max, scaled to [0,1)", 2),
    "IdxMin": Op(_idxmin, 1, True, "timeseries", "days since the d-day min, scaled to [0,1)", 2),
    "Corr": Op(lambda a, b, d: _rolling_corr(a, b, d), 2, True, "timeseries",
               "rolling correlation of x and y", 3),
    "Cov": Op(lambda a, b, d: (a * b).rolling(d, min_periods=d).mean()
              - a.rolling(d, min_periods=d).mean() * b.rolling(d, min_periods=d).mean(),
              2, True, "timeseries", "rolling covariance", 3),
    # cross-sectional (across the stocks in the index that day)
    "CsRank": Op(None, 1, False, "cross_section", "percentile rank across stocks today"),
    "CsZscore": Op(None, 1, False, "cross_section", "z-score across stocks today"),
    "CsDemean": Op(None, 1, False, "cross_section", "minus the cross-sectional mean today"),
}


# ------------------------------------------------------------------ parsing
@dataclass
class Node:
    kind: str                      # "field" | "const" | "op" | "bin" | "neg"
    value: object = None           # field name, number, op name or operator symbol
    args: tuple = ()
    window: int | None = None

    def size(self) -> int:
        return 1 + sum(a.size() for a in self.args)

    def depth(self) -> int:
        return 1 + max((a.depth() for a in self.args), default=0)

    def fields(self) -> set[str]:
        s = {self.value} if self.kind == "field" else set()
        for a in self.args:
            s |= a.fields()
        return s

    def ops(self) -> list[str]:
        s = [self.value] if self.kind == "op" else []
        for a in self.args:
            s += a.ops()
        return s

    def max_lookback(self) -> int:
        """Total days of history needed (windows nest additively)."""
        inner = max((a.max_lookback() for a in self.args), default=0)
        return inner + (self.window or 0)

    def is_constant(self) -> bool:
        return not self.fields()

    def __str__(self) -> str:
        """Canonical text with only the brackets that are needed (also the de-dup key)."""
        if self.kind == "field":
            return str(self.value)
        if self.kind == "const":
            v = float(self.value)
            return str(int(v)) if v.is_integer() and abs(v) < 1e15 else repr(v)
        if self.kind == "neg":
            a = self.args[0]
            inner = str(a)
            return f"-{inner}" if a.kind in ("field", "op", "const") else f"-({inner})"
        if self.kind == "bin":
            prec = _PREC[self.value]
            left, right = self.args

            def wrap(child, is_right):
                t = str(child)
                if child.kind == "bin":
                    cp = _PREC[child.value]
                    if cp < prec or (is_right and cp == prec and self.value in "-/"):
                        return f"({t})"
                elif child.kind == "neg" and is_right:
                    return f"({t})"
                return t

            return f"{wrap(left, False)} {self.value} {wrap(right, True)}"
        inner = [str(a) for a in self.args] + ([str(self.window)] if self.window is not None else [])
        return f"{self.value}({', '.join(inner)})"


_PREC = {"+": 1, "-": 1, "*": 2, "/": 2}
_BINOPS = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}


def _convert(n: ast.AST) -> Node:
    if isinstance(n, ast.Name):
        if n.id not in FIELDS:
            raise FactorError(f"unknown field '{n.id}' (allowed: {', '.join(FIELDS)})")
        return Node("field", n.id)
    if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
        return Node("const", float(n.value))
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
        inner = _convert(n.operand)
        if isinstance(n.op, ast.UAdd):
            return inner
        if inner.kind == "const":
            return Node("const", -inner.value)
        return Node("neg", args=(inner,))
    if isinstance(n, ast.BinOp) and type(n.op) in _BINOPS:
        left, right = _convert(n.left), _convert(n.right)
        if isinstance(n.op, ast.Div) and right.kind == "const" and abs(right.value) <= EPS:
            raise FactorError("division by zero")
        return Node("bin", _BINOPS[type(n.op)], (left, right))
    if isinstance(n, ast.Call):
        if not isinstance(n.func, ast.Name) or n.func.id not in OPS:
            name = getattr(n.func, "id", ast.dump(n.func))
            raise FactorError(f"unknown operator '{name}'")
        if n.keywords:
            raise FactorError("keyword arguments are not allowed")
        op = OPS[n.func.id]
        want = op.n_series + (1 if op.window else 0)
        if len(n.args) != want:
            raise FactorError(f"{n.func.id} takes {want} arguments, got {len(n.args)}")
        window = None
        series_args = n.args
        if op.window:
            w = n.args[-1]
            if not (isinstance(w, ast.Constant) and isinstance(w.value, int) and not isinstance(w.value, bool)):
                raise FactorError(f"{n.func.id}: window must be an integer literal")
            window = int(w.value)
            if not op.min_window <= window <= MAX_WINDOW:
                raise FactorError(f"{n.func.id}: window must be {op.min_window}..{MAX_WINDOW}, got {window}")
            series_args = n.args[:-1]
        args = tuple(_convert(a) for a in series_args)
        if all(a.is_constant() for a in args):
            raise FactorError(f"{n.func.id} applied to a constant")
        return Node("op", n.func.id, args, window)
    raise FactorError(f"syntax not allowed: {type(n).__name__}")


def parse(expr: str) -> Node:
    if not isinstance(expr, str) or not expr.strip():
        raise FactorError("empty expression")
    if len(expr) > 400:
        raise FactorError("expression too long")
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as e:
        raise FactorError(f"syntax error: {e.msg}") from None
    node = _convert(tree.body)
    if node.is_constant():
        raise FactorError("expression uses no data fields")
    if node.size() > MAX_NODES:
        raise FactorError(f"too complex: {node.size()} nodes (max {MAX_NODES})")
    if node.depth() > MAX_DEPTH:
        raise FactorError(f"too deeply nested: depth {node.depth()} (max {MAX_DEPTH})")
    if node.max_lookback() > 2 * MAX_WINDOW:
        raise FactorError(f"needs {node.max_lookback()} days of history (max {2 * MAX_WINDOW})")
    return node


# --------------------------------------------------------------- evaluation
class Evaluator:
    """Evaluates parsed factors on a Market. Caches shared sub-expressions."""

    def __init__(self, market):
        m = market
        self.member = m.member
        self.data = {
            "open": m.open, "high": m.high, "low": m.low, "close": m.close,
            "volume": m.volume.where(m.volume > 0), "vwap": m.vwap,
            "returns": m.close / m.close.shift(1) - 1,
        }
        self._cache: dict[str, pd.DataFrame] = {}

    def clear_cache(self) -> None:
        self._cache.clear()

    def __call__(self, expr: str | Node) -> pd.DataFrame:
        node = parse(expr) if isinstance(expr, str) else expr
        with np.errstate(all="ignore"):
            out = self._eval(node)
        if np.isscalar(out):
            raise FactorError("expression evaluates to a constant")
        out = out.astype("float64").replace([np.inf, -np.inf], np.nan)
        return out.where(self.member)

    def _eval(self, n: Node):
        key = str(n)
        if key in self._cache:
            return self._cache[key]
        if n.kind == "field":
            out = self.data[n.value]
        elif n.kind == "const":
            return n.value
        elif n.kind == "neg":
            out = -self._eval(n.args[0])
        elif n.kind == "bin":
            a, b = self._eval(n.args[0]), self._eval(n.args[1])
            out = {"+": lambda: a + b, "-": lambda: a - b, "*": lambda: a * b,
                   "/": lambda: _safe_div(a, b)}[n.value]()
        else:
            op = OPS[n.value]
            args = [self._eval(a) for a in n.args]
            args = [a if not np.isscalar(a) else pd.DataFrame(a, index=self.member.index,
                                                              columns=self.member.columns)
                    for a in args]
            if op.kind == "cross_section":
                x = args[0].where(self.member)   # rank only among index members that day
                if n.value == "CsRank":
                    out = x.rank(axis=1, pct=True)
                elif n.value == "CsZscore":
                    out = x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1) + EPS, axis=0)
                else:
                    out = x.sub(x.mean(axis=1), axis=0)
            elif op.window:
                out = op.fn(*args, n.window)
            else:
                out = op.fn(*args)
            if isinstance(out, pd.DataFrame):
                out = out.replace([np.inf, -np.inf], np.nan)
        if len(self._cache) > 400:
            self._cache.clear()
        self._cache[key] = out
        return out


def grammar_doc() -> str:
    """Human/LLM-readable description of the language, used in the prompt."""
    lines = ["Fields (daily, split-adjusted, per stock): " + ", ".join(FIELDS)
             + "  (returns = close / previous close - 1)",
             "Arithmetic: + - * /, unary minus, numeric constants.",
             f"Windows are integer literals between 1 and {MAX_WINDOW}.", "Operators:"]
    for kind, title in [("elementwise", "Element-wise"), ("timeseries", "Time-series (per stock, looks back only)"),
                        ("cross_section", "Cross-sectional (across stocks on the same day)")]:
        lines.append(f"  {title}:")
        for name, op in OPS.items():
            if op.kind != kind:
                continue
            sig = ", ".join(["x", "y"][: op.n_series] + (["d"] if op.window else []))
            lines.append(f"    {name}({sig}): {op.doc}")
    lines.append(f"Limits: at most {MAX_NODES} nodes, nesting depth {MAX_DEPTH}.")
    return "\n".join(lines)
