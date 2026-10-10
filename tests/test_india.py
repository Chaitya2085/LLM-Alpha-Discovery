"""Tests for the Indian data pipeline, on a synthetic NSE archive (no internet needed).

Run:  python tests/test_india.py   (or pytest)
"""
import dataclasses
import json
import sys
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))
import build_india  # noqa: E402
import india_download  # noqa: E402
import markets  # noqa: E402
from fake_nse import make_archive, serve  # noqa: E402
from india_parse import read_equity_file, read_index_file  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="nse_test_"))
ARCHIVE = TMP / "archive"
START, END = date(2024, 3, 1), date(2024, 10, 31)   # crosses NSE's July 2024 format change
GT = make_archive(ARCHIVE, START, END, n_stocks=40, seed=3)
URL, SRV = serve(ARCHIVE)
RAW, OUT = TMP / "raw", TMP / "processed"


def _download(*extra):
    india_download.BASES = [URL]
    india_download.HOME = URL
    india_download.RAW = RAW
    sys.argv = ["india_download.py", "--start", str(START), "--end", str(END), "--pause", "0", *extra]
    india_download.main()


def test_parse_both_formats():
    old = ARCHIVE / "content/historical/EQUITIES/2024/JUN/cm03JUN2024bhav.csv.zip"
    new = ARCHIVE / "content/cm/BhavCopy_NSE_CM_0_0_0_20240808_F_0000.csv.zip"
    for f in (old, new):
        if not f.exists():  # pick the next available day if that date was a holiday
            f = sorted(f.parent.glob("*.zip"))[0]
        df = read_equity_file(f)
        assert set(["symbol", "open", "high", "low", "close", "prevclose", "volume", "value"]) <= set(df.columns)
        assert (df["series"] == "EQ").all() and "GSEC2030" not in set(df["symbol"])
        allser = read_equity_file(f, all_series=True)
        assert set(allser["series"]) <= {"EQ", "BE", "BZ"} and len(allser) >= len(df)
        assert df[["open", "close", "volume"]].notna().all().all() and len(df) > 20
    ix = read_index_file(sorted((ARCHIVE / "content/indices").glob("*.csv"))[0])
    assert {"Nifty 50", "Nifty 200"} <= set(ix["name"]) and ix["close"].notna().all()


def test_download_resume_and_holidays():
    _download("--workers", "3")
    got = sorted((RAW / "equities").glob("*/*.csv.zip"))
    assert len(got) == len(GT["days"]), (len(got), len(GT["days"]))
    assert len(list((RAW / "indices").glob("*/*.csv"))) == len(GT["days"])
    no_file = set(json.loads((RAW / "no_file_days.json").read_text()))
    assert {d.isoformat() for d in GT["holidays"] if START <= d <= END} <= no_file
    n_before = len(SRV.calls)
    _download()                       # second run: everything already there
    assert len(SRV.calls) - n_before <= 2, "re-run should not download again"


def test_blocked_download_stops_cleanly():
    url, srv = serve(ARCHIVE, block={"/content/"})
    india_download.BASES, india_download.HOME = [url], url
    india_download.RAW = TMP / "raw_blocked"
    sys.argv = ["x", "--start", "2024-03-04", "--end", "2024-03-08", "--pause", "0"]
    try:
        india_download.main()
        raise AssertionError("should stop when NSE refuses")
    except SystemExit as e:
        assert e.code == 2


def test_busy_server_is_retried_and_never_saved_as_holiday():
    india_download.BACKOFF = [0, 0, 0]          # don't actually wait in tests
    india_download.MAX_PAUSE = 0
    lo, hi = date(2024, 4, 1), date(2024, 4, 30)
    good = [d for d in GT["days"] if lo <= d <= hi]
    # a) NSE answers 503 a couple of times, then works: every day still arrives
    flaky = {f"cm{d.day:02d}APR2024": 2 for d in good[:5]} | {f"ind_close_all_{good[6]:%d%m%Y}": 3}
    url, srv = serve(ARCHIVE, flaky=flaky)
    india_download.BASES, india_download.HOME = [url], url
    india_download.RAW = raw = TMP / "raw_flaky"
    sys.argv = ["x", "--start", str(lo), "--end", str(hi), "--pause", "0", "--workers", "2"]
    india_download.main()
    assert len(list((raw / "equities").glob("*/*.zip"))) == len(good)
    assert len(list((raw / "indices").glob("*/*.csv"))) == len(good)

    # b) NSE stays down for some days: run finishes, exits 3, those days are not holidays
    down_days = good[3:6]
    url, srv = serve(ARCHIVE, down={f"cm{d.day:02d}APR2024" for d in down_days})
    india_download.BASES, india_download.HOME = [url], url
    india_download.RAW = raw = TMP / "raw_down"
    try:
        india_download.main()
        raise AssertionError("should exit with code 3 when days are left to retry")
    except SystemExit as e:
        assert e.code == 3
    no_file = set(json.loads((raw / "no_file_days.json").read_text()))
    failed = set(json.loads((raw / "failed_days.json").read_text()))
    assert {d.isoformat() for d in down_days} == failed and not (failed & no_file)
    assert len(list((raw / "equities").glob("*/*.zip"))) == len(good) - len(down_days)

    # c) NSE recovers: a re-run fetches only the missing days
    srv.down.clear()
    n0 = len(srv.calls)
    india_download.main()
    assert len(list((raw / "equities").glob("*/*.zip"))) == len(good)
    assert len(srv.calls) - n0 <= 2 * len(down_days) + 2
    assert not (raw / "failed_days.json").exists()

    # d) NSE completely down: stops early instead of grinding through every day
    url, srv = serve(ARCHIVE, down={"/content/"})
    india_download.BASES, india_download.HOME = [url], url
    india_download.RAW = TMP / "raw_dead"
    sys.argv = ["x", "--start", "2024-03-01", "--end", "2024-10-31", "--pause", "0", "--workers", "1"]
    try:
        india_download.main()
        raise AssertionError("should stop")
    except SystemExit as e:
        assert e.code == 3
    dead_calls = len(srv.calls)
    assert dead_calls < 4 * (india_download.STOP_AFTER + 3), dead_calls
    assert not json.loads((TMP / "raw_dead" / "no_file_days.json").read_text())


def test_build_adjusts_splits_and_builds_point_in_time_universe():
    build_india.RAW, build_india.OUT = RAW, OUT
    sys.argv = ["build_india.py", "--universe", "25", "--start", str(START)]
    build_india.main()
    px = pd.read_parquet(OUT / "prices.parquet")
    member = pd.read_parquet(OUT / "membership.parquet")
    q = json.loads((OUT / "data_quality.json").read_text())
    assert q["benchmark"] == "Nifty 200"

    # splits AND bonuses: adjusted returns must follow the true returns, with no -50%/-90% jump
    # on the ex-date; the genuine -50% crash must stay; a weekend special session changes nothing
    days = pd.DatetimeIndex(GT["days"])
    true_close = pd.DataFrame(GT["true_close"], index=days, columns=GT["symbols"])
    adj = px["close"].unstack("code")
    locked = (px["high"] == px["low"]).unstack("code")
    checked = 0
    for sym in adj.columns:
        a = adj[sym].dropna()
        tr = true_close[sym].reindex(a.index)
        ra, rt = a.pct_change(), tr.pct_change()
        lk = locked[sym].reindex(a.index).fillna(False)
        ok = ~(lk | lk.shift(1, fill_value=False)) & ra.notna() & rt.notna()  # skip locked days + day after
        err = (ra[ok] - rt[ok]).abs()
        assert err.max() < 0.01, f"{sym}: adjusted returns differ from true returns by {err.max():.3f}"
        checked += 1
    assert checked >= 20, checked
    for sym, d, k, kind in GT["actions"]:
        if sym in adj.columns and pd.Timestamp(d) in adj[sym].dropna().index:
            assert abs(adj[sym].pct_change().get(pd.Timestamp(d), 0)) < 0.2, f"{kind} {sym} on {d} not adjusted"
    crash_sym, crash_day = GT["crash"]
    assert crash_sym in adj.columns, "the crash stock should be in the universe"
    assert adj[crash_sym].pct_change()[pd.Timestamp(crash_day)] < -0.45, "a real crash must not be 'adjusted' away"
    ev = pd.read_csv(OUT / "corporate_actions.csv")
    done = ev[ev["adjusted"]]
    in_uni = {(a[0], a[3]) for a in GT["actions"]
              if a[0] in adj.columns and pd.Timestamp(a[1]) > adj[a[0]].first_valid_index()}
    assert q["splits_adjusted"] == sum(1 for _, k in in_uni if k == "split"), (q["splits_adjusted"], in_uni)
    assert q["bonuses_adjusted"] == sum(1 for _, k in in_uni if k == "bonus"), (q["bonuses_adjusted"], in_uni)
    assert q["bonuses_adjusted"] >= 1 and q["splits_adjusted"] >= 1
    assert crash_sym not in set(done["symbol"])
    assert GT["etf"] not in set(px.index.get_level_values("code")), "ETFs must be dropped"

    # universe: at most 25 a day, only changes at month starts, decided from past data
    assert member.sum(axis=1).max() <= 25
    changes = member.astype(int).diff().abs().sum(axis=1)
    changed_days = changes[changes > 0].index
    assert all(d.month != (d - pd.offsets.BDay(1)).month or d.day <= 7 for d in changed_days)
    first = member.index[member.any(axis=1)][0]
    assert first >= pd.Timestamp(START) + pd.Timedelta(days=60), "universe needs ~3 months of history"


def test_market_loader_detects_locked_circuits():
    cfg = dataclasses.replace(markets.MARKETS["india"], proc=OUT)
    markets.MARKETS["india"] = cfg
    import data
    m = data.load_market("india")
    assert m.name == "india" and m.buy_cost == 0.0012
    locked = (m.high == m.low) & (m.close / m.close.ffill().shift(1) - 1 >= 0.019)
    assert locked.values.sum() > 0, "synthetic data should contain locked-up days"
    assert not (m.can_buy & locked).values.any(), "can't buy a stock locked at its upper band"
    assert m.index_close.notna().mean() > 0.95


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\nall {len(tests)} tests passed")
