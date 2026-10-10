"""Alpha158 (Qlib's standard handcrafted factor set), reimplemented on wide
(date x code) frames. These are the 'human-designed' factors the LLM has to beat,
and Phase 3 uses them as the library that new factors must not duplicate.

Definitions follow qlib/contrib/data/loader.py (Alpha158DL). Every feature uses
data up to and including day t only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from data import Market

WINDOWS = [5, 10, 20, 30, 60]
EPS = 1e-12


CHUNK = 200  # rows of time per block, keeps strided temporaries small


def _roll_apply(df: pd.DataFrame, w: int, fn) -> pd.DataFrame:
    """Apply fn(window_array[T', N, w]) -> [T', N] on a strided view, in time chunks."""
    a = df.to_numpy(dtype="float64")
    out = np.full_like(a, np.nan)
    if len(a) >= w:
        v = sliding_window_view(a, w, axis=0)          # (T-w+1, N, w), no copy
        with np.errstate(all="ignore"):
            for s in range(0, len(v), CHUNK):
                out[w - 1 + s: w - 1 + s + CHUNK] = fn(v[s:s + CHUNK])
    return pd.DataFrame(out, index=df.index, columns=df.columns)


def _slope_rsq_resi(y: pd.DataFrame, w: int):
    """Rolling OLS of y on time 0..w-1 -> slope, R^2, last residual (qlib Slope/Rsquare/Resi)."""
    x = np.arange(w, dtype="float64")
    xc = x - x.mean()
    xv = (xc ** 2).sum()
    a = y.to_numpy(dtype="float64")
    slope = np.full_like(a, np.nan)
    rsq = np.full_like(a, np.nan)
    resi = np.full_like(a, np.nan)
    if len(a) >= w:
        v = sliding_window_view(a, w, axis=0)
        with np.errstate(all="ignore"):
            for s in range(0, len(v), CHUNK):
                blk = v[s:s + CHUNK]
                ym = blk.mean(axis=-1)
                b = (blk @ xc) / xv
                ss_tot = ((blk - ym[..., None]) ** 2).sum(axis=-1)
                ss_res = ss_tot - b ** 2 * xv
                sl = slice(w - 1 + s, w - 1 + s + len(blk))
                slope[sl] = b
                rsq[sl] = 1 - ss_res / (ss_tot + EPS)
                resi[sl] = blk[..., -1] - (ym + b * xc[-1])
    mk = lambda arr: pd.DataFrame(arr, index=y.index, columns=y.columns)
    return mk(slope), mk(rsq), mk(resi)


def _rolling_corr(a: pd.DataFrame, b: pd.DataFrame, w: int) -> pd.DataFrame:
    ma, mb = a.rolling(w).mean(), b.rolling(w).mean()
    cov = (a * b).rolling(w).mean() - ma * mb
    va = (a * a).rolling(w).mean() - ma ** 2
    vb = (b * b).rolling(w).mean() - mb ** 2
    return cov / np.sqrt((va * vb).clip(lower=EPS))


def alpha158(m: Market) -> dict[str, pd.DataFrame]:
    """All 158 features as a dict (memory-heavy; prefer build_matrix)."""
    return dict(iter_alpha158(m))


def iter_alpha158(m: Market):
    """Yield (name, wide DataFrame) one feature group at a time."""
    o, h, l, c, v, vw = m.open, m.high, m.low, m.close, m.volume, m.vwap
    v = v.where(v > 0)
    # --- k-bar shape (9) ---
    f: dict[str, pd.DataFrame] = {}
    rng = (h - l) + EPS
    f["KMID"] = (c - o) / o
    f["KLEN"] = (h - l) / o
    f["KMID2"] = (c - o) / rng
    f["KUP"] = (h - np.maximum(o, c)) / o
    f["KUP2"] = (h - np.maximum(o, c)) / rng
    f["KLOW"] = (np.minimum(o, c) - l) / o
    f["KLOW2"] = (np.minimum(o, c) - l) / rng
    f["KSFT"] = (2 * c - h - l) / o
    f["KSFT2"] = (2 * c - h - l) / rng

    yield from f.items()
    f = {}
    # --- today's prices relative to close (4) ---
    f["OPEN0"], f["HIGH0"], f["LOW0"], f["VWAP0"] = o / c, h / c, l / c, vw / c

    ret = c / c.shift(1)
    up = (c - c.shift(1)).clip(lower=0)
    dn = (c.shift(1) - c).clip(lower=0)
    absd = (c - c.shift(1)).abs()
    lvc = np.log(v + 1)
    lvchg = np.log(v / v.shift(1) + 1)
    vup = (v - v.shift(1)).clip(lower=0)
    vdn = (v.shift(1) - v).clip(lower=0)
    vabs = (v - v.shift(1)).abs()
    wv = (c / c.shift(1) - 1).abs() * v

    yield from f.items()
    for w in WINDOWS:
        f = {}
        f[f"ROC{w}"] = c.shift(w) / c
        f[f"MA{w}"] = c.rolling(w).mean() / c
        f[f"STD{w}"] = c.rolling(w).std() / c
        slope, rsq, resi = _slope_rsq_resi(c, w)
        f[f"BETA{w}"], f[f"RSQR{w}"], f[f"RESI{w}"] = slope / c, rsq, resi / c
        hmax, lmin = h.rolling(w).max(), l.rolling(w).min()
        f[f"MAX{w}"], f[f"MIN{w}"] = hmax / c, lmin / c
        f[f"QTLU{w}"] = c.rolling(w).quantile(0.8) / c
        f[f"QTLD{w}"] = c.rolling(w).quantile(0.2) / c
        f[f"RANK{w}"] = c.rolling(w).rank(pct=True)
        f[f"RSV{w}"] = (c - lmin) / (hmax - lmin + EPS)
        imax = _roll_apply(h, w, lambda x: np.argmax(x, axis=-1).astype(float))
        imin = _roll_apply(l, w, lambda x: np.argmin(x, axis=-1).astype(float))
        f[f"IMAX{w}"], f[f"IMIN{w}"] = imax / w, imin / w
        f[f"IMXD{w}"] = (imax - imin) / w
        f[f"CORR{w}"] = _rolling_corr(c, lvc, w)
        f[f"CORD{w}"] = _rolling_corr(ret, lvchg, w)
        f[f"CNTP{w}"] = (c > c.shift(1)).astype(float).rolling(w).mean()
        f[f"CNTN{w}"] = (c < c.shift(1)).astype(float).rolling(w).mean()
        f[f"CNTD{w}"] = f[f"CNTP{w}"] - f[f"CNTN{w}"]
        su, sd, sa = up.rolling(w).sum(), dn.rolling(w).sum(), absd.rolling(w).sum() + EPS
        f[f"SUMP{w}"], f[f"SUMN{w}"], f[f"SUMD{w}"] = su / sa, sd / sa, (su - sd) / sa
        f[f"VMA{w}"] = v.rolling(w).mean() / (v + EPS)
        f[f"VSTD{w}"] = v.rolling(w).std() / (v + EPS)
        f[f"WVMA{w}"] = wv.rolling(w).std() / (wv.rolling(w).mean() + EPS)
        vu, vd, va = vup.rolling(w).sum(), vdn.rolling(w).sum(), vabs.rolling(w).sum() + EPS
        f[f"VSUMP{w}"], f[f"VSUMN{w}"], f[f"VSUMD{w}"] = vu / va, vd / va, (vu - vd) / va
        yield from f.items()
    return f


def build_matrix(m: Market) -> pd.DataFrame:
    """Long float32 matrix: rows = (date, code) where the stock is a CSI300 member,
    columns = 158 features. ~1M rows x 158 cols, ~0.6 GB."""
    mask = m.member.to_numpy()
    rows_d, rows_c = np.nonzero(mask)
    cols = {}
    for name, df in iter_alpha158(m):
        arr = df.to_numpy(dtype="float64")[rows_d, rows_c]
        arr[~np.isfinite(arr)] = np.nan
        cols[name] = arr.astype("float32")
    idx = pd.MultiIndex.from_arrays(
        [m.dates[rows_d], m.member.columns[rows_c]], names=["date", "code"])
    return pd.DataFrame(cols, index=idx)
