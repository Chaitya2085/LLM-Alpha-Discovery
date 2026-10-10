"""Synthetic NSE archive for tests: realistic bhavcopy and index files in both formats.

Prices are random walks (no predictable signal), with stock splits/bonuses, new listings,
delistings, holidays, locked-circuit days, non-equity rows and non-EQ series mixed in.
Files are written with the same paths NSE uses, so a tiny HTTP server can serve them to
the real downloader.
"""
from __future__ import annotations

import csv
import http.server
import io
import threading
import zipfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np

MON = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
UDIFF_FROM = date(2024, 7, 8)
OLD_COLS = ["SYMBOL", "SERIES", "OPEN", "HIGH", "LOW", "CLOSE", "LAST", "PREVCLOSE", "TOTTRDQTY",
            "TOTTRDVAL", "TIMESTAMP", "TOTALTRADES", "ISIN", ""]
NEW_COLS = ["TradDt", "BizDt", "Sgmt", "Src", "FinInstrmTp", "FinInstrmId", "ISIN", "TckrSymb", "SctySrs",
            "XpryDt", "FininstrmActlXpryDt", "StrkPric", "OptnTp", "FinInstrmNm", "OpnPric", "HghPric",
            "LwPric", "ClsPric", "LastPric", "PrvsClsgPric", "UndrlygPric", "SttlmPric", "OpnIntrst",
            "ChngInOpnIntrst", "TtlTradgVol", "TtlTrfVal", "TtlNbOfTxsExctd", "SsnId", "NewBrdLotQty",
            "Rmks", "Rsvd1", "Rsvd2", "Rsvd3", "Rsvd4"]
IX_COLS = ["Index Name", "Index Date", "Open Index Value", "High Index Value", "Low Index Value",
           "Closing Index Value", "Points Change", "Change(%)", "Volume", "Turnover (Rs. Cr.)",
           "P/E", "P/B", "Div Yield"]


def _tick(x: float) -> float:
    return round(round(x / 0.05) * 0.05, 2)


def make_archive(root: Path, start: date, end: date, n_stocks: int = 60, seed: int = 0) -> dict:
    """Write a fake NSE archive under root. Returns ground truth used by the tests."""
    rng = np.random.default_rng(seed)
    days = [start + timedelta(d) for d in range((end - start).days + 1)]
    days = [d for d in days if d.weekday() < 5]
    holidays = {d for d in days if rng.random() < 0.04} | {d for d in days if (d.month, d.day) == (1, 26)}
    days = [d for d in days if d not in holidays]
    T = len(days)
    syms = [f"STK{i:04d}" for i in range(n_stocks)]
    size = np.exp(rng.normal(0, 1.2, n_stocks))             # liquidity differs a lot by stock
    vol = rng.uniform(0.012, 0.03, n_stocks)
    listed = np.zeros((T, n_stocks), bool)
    for j in range(n_stocks):
        a = 0 if rng.random() < 0.8 else int(rng.integers(0, T // 2))
        b = T if rng.random() < 0.9 else int(rng.integers(a + 80, T)) if a + 80 < T else T
        listed[a:b, j] = True
    true_ret = rng.normal(0.0003, vol, (T, n_stocks))
    true_ret[0] = 0
    true_px = 200 * np.exp(rng.normal(0, 0.8, n_stocks)) * np.cumprod(1 + true_ret, axis=0)
    # corporate actions: some stocks split 1:2 or 1:5 (or 1:1 bonus) on a random day
    split_ratio = np.ones((T, n_stocks))
    actions = []
    for j in rng.choice(n_stocks, size=max(2, n_stocks // 6), replace=False):
        t = int(rng.integers(T // 5, T - 5))
        k = float(rng.choice([2.0, 5.0]))
        split_ratio[t:, j] = k
        actions.append((syms[j], days[t], k))
    raw_px = true_px / split_ratio
    locked_up = rng.random((T, n_stocks)) < 0.002
    gt = {"days": days, "holidays": sorted(holidays), "symbols": syms, "actions": actions,
          "true_close": true_px, "listed": listed, "size": size}

    last_close: dict[int, float] = {}
    for t, d in enumerate(days):
        rows_old, rows_new = [], []
        for j, s in enumerate(syms):
            if not listed[t, j]:
                continue
            c = _tick(raw_px[t, j])
            if t > 0 and listed[t - 1, j]:
                # previous close is the real last close; NSE adjusts it on a split's ex-date
                prev = _tick(last_close[j] * split_ratio[t - 1, j] / split_ratio[t, j])
            else:
                prev = c
            if locked_up[t, j] and t > 0:
                c = _tick(prev * 1.05)
                o = h = lo = c
            else:
                o = _tick(c * (1 + rng.normal(0, vol[j] / 3)))
                h = max(o, c) * (1 + abs(rng.normal(0, vol[j] / 2)))
                lo = min(o, c) * (1 - abs(rng.normal(0, vol[j] / 2)))
                h, lo = _tick(h), _tick(lo)
            last_close[j] = c
            qty = int(max(1, size[j] * 2e5 * np.exp(rng.normal(0, 0.4)) / max(c, 1) * 100))
            val = round(qty * (o + h + lo + c) / 4, 2)
            series = "EQ" if rng.random() > 0.01 else "BE"
            isin = f"INE{j:05d}A01{d.year % 10}"
            rows_old.append([s, series, o, h, lo, c, c, prev, qty, val, f"{d.day:02d}-{MON[d.month - 1]}-{d.year}",
                             int(qty / 50) + 1, isin, ""])
            rows_new.append([d.isoformat(), d.isoformat(), "CM", "NSE", "STK", str(1000 + j), isin, s, series,
                             "", "", "", "", s + " LTD", o, h, lo, c, c, prev, "", c, "", "", qty, val,
                             int(qty / 50) + 1, "F1", 1, "", "", "", "", ""])
        # non-equity rows that must be ignored
        rows_old.append(["GSEC2030", "GS", 100, 100, 100, 100, 100, 100, 10, 1000,
                         f"{d.day:02d}-{MON[d.month - 1]}-{d.year}", 1, "IN0020", ""])
        rows_new.append([d.isoformat(), d.isoformat(), "CM", "NSE", "BND", "9", "IN0020", "GSEC2030", "GS",
                         "", "", "", "", "GSEC", 100, 100, 100, 100, 100, 100, "", 100, "", "", 10, 1000, 1,
                         "F1", 1, "", "", "", "", ""])
        buf = io.StringIO()
        w = csv.writer(buf)
        if d >= UDIFF_FROM:
            w.writerow(NEW_COLS)
            w.writerows(rows_new)
            rel = f"content/cm/BhavCopy_NSE_CM_0_0_0_{d:%Y%m%d}_F_0000.csv.zip"
            inner = f"BhavCopy_NSE_CM_0_0_0_{d:%Y%m%d}_F_0000.csv"
        else:
            w.writerow(OLD_COLS)
            w.writerows(rows_old)
            rel = f"content/historical/EQUITIES/{d.year}/{MON[d.month - 1]}/cm{d.day:02d}{MON[d.month - 1]}{d.year}bhav.csv.zip"
            inner = f"cm{d.day:02d}{MON[d.month - 1]}{d.year}bhav.csv"
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr(inner, buf.getvalue())

        # index file: Nifty 50 / Nifty 200 as equal-weight averages of true returns
        lvl50 = 8000 * float(np.prod(1 + true_ret[: t + 1, :50].mean(axis=1)))
        lvl200 = 4000 * float(np.prod(1 + true_ret[: t + 1].mean(axis=1)))
        ib = io.StringIO()
        iw = csv.writer(ib)
        iw.writerow(IX_COLS)
        for name, lv in [("Nifty 50", lvl50), ("Nifty 200", lvl200), ("Nifty Bank", lvl50 * 4)]:
            iw.writerow([name, f"{d:%d-%m-%Y}", f"{lv * 0.999:.2f}", f"{lv * 1.004:.2f}", f"{lv * 0.995:.2f}",
                         f"{lv:.2f}", "1.0", "0.01", "100", "10", "20", "3", "1.2"])
        ip = root / f"content/indices/ind_close_all_{d:%d%m%Y}.csv"
        ip.parent.mkdir(parents=True, exist_ok=True)
        ip.write_text(ib.getvalue())
    return gt


def serve(root: Path, block: set[str] | None = None, flaky: dict[str, int] | None = None,
          down: set[str] | None = None) -> tuple[str, http.server.HTTPServer]:
    """Serve the archive like NSE does: 404 for missing days, optional 403 'blocked' paths,
    'flaky' paths that answer 503 the first N times, and 'down' paths that always answer 503.
    srv.down can be changed while running (e.g. cleared to simulate NSE recovering)."""
    block = block or set()
    flaky_left = dict(flaky or {})
    calls: list[str] = []
    lock = threading.Lock()

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            calls.append(self.path)
            if any(b in self.path for b in block):
                self.send_response(403)
                self.end_headers()
                return
            busy = any(b in self.path for b in srv.down)
            with lock:
                for k, n in flaky_left.items():
                    if k in self.path and n > 0:
                        flaky_left[k] = n - 1
                        busy = True
            if busy:
                self.send_response(503)
                self.send_header("Retry-After", "0")
                self.end_headers()
                return
            p = root / self.path.lstrip("/")
            if not p.is_file():
                self.send_response(404)
                self.end_headers()
                return
            data = p.read_bytes()
            self.send_response(200)
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    srv.calls = calls  # type: ignore[attr-defined]
    srv.down = set(down or ())  # type: ignore[attr-defined]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{srv.server_port}", srv
