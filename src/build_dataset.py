"""Phase 1 data build: CSI300 point-in-time universe, 2014-01-01 onward.

Outputs (data/processed/):
  prices.parquet      long panel (date, code): adjusted OHLCV + raw close + flags
  membership.parquet  boolean (date x code): was the stock in CSI300 that day
  index.parquet       CSI300 index daily close (benchmark)
  data_quality.json   summary checks
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from qlib_reader import QlibReader  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "qlib_bin"
OUT = ROOT / "data" / "processed"
START, UNIVERSE, INDEX = "2014-01-01", "csi300", "SH000300"


def limit_pct(code: str, date: pd.Series) -> np.ndarray:
    """Daily price limit by board: 20% for STAR (688) always and ChiNext (300)
    from 2020-08-24, otherwise 10%. (ST stocks at 5% are not flagged here.)"""
    num = code[2:]
    if num.startswith("688"):
        return np.full(len(date), 0.20)
    if num.startswith("300"):
        return np.where(date >= pd.Timestamp("2020-08-24"), 0.20, 0.10)
    return np.full(len(date), 0.10)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    r = QlibReader(RAW)
    cal = r.calendar[r.calendar >= START]

    inst = r.instruments(UNIVERSE)
    inst = inst[inst["end"] >= START]
    codes = sorted(inst["code"].unique())

    # Point-in-time membership matrix
    member = pd.DataFrame(False, index=cal, columns=codes)
    for row in inst.itertuples():
        member.loc[(cal >= row.start) & (cal <= row.end), row.code] = True

    px = r.panel(codes, start=START)
    px["raw_close"] = px["close"] / px["factor"]
    px["ret_1d"] = px.groupby(level="code")["close"].pct_change(fill_method=None)
    raw_prev = px.groupby(level="code")["raw_close"].shift(1)
    raw_chg = px["raw_close"] / raw_prev - 1

    codes_idx = px.index.get_level_values("code")
    dates_idx = px.index.get_level_values("date")
    lim = np.empty(len(px))
    for c in codes:
        m = codes_idx == c
        lim[m] = limit_pct(c, pd.Series(dates_idx[m]))
    px["suspended"] = px["volume"].isna() | (px["volume"] <= 0)
    px["limit_up"] = raw_chg >= lim - 0.002
    px["limit_down"] = raw_chg <= -lim + 0.002
    px["in_universe"] = [
        bool(member.at[d, c]) if d in member.index else False
        for d, c in zip(dates_idx, codes_idx)
    ]
    px.to_parquet(OUT / "prices.parquet")
    member.index.name = "date"
    member.to_parquet(OUT / "membership.parquet")

    idx = r.panel([INDEX], fields=["open", "close", "adjclose"], start=START).droplevel("code")
    idx = idx.rename(columns={"adjclose": "level"})
    idx["level_open"] = idx["level"] * idx["open"] / idx["close"]
    idx.to_parquet(OUT / "index.parquet")

    # ---- quality checks ----
    in_u = px[px["in_universe"]]
    per_day = in_u.groupby(level="date").size()
    big = in_u["ret_1d"].abs() > 0.25
    q = {
        "universe": UNIVERSE,
        "start": str(cal[0].date()),
        "end": str(cal[-1].date()),
        "trading_days": int(len(cal)),
        "unique_stocks_ever_in_universe": len(codes),
        "members_per_day_min_median_max": [int(per_day.min()), int(per_day.median()), int(per_day.max())],
        "member_days_missing_close_pct": round(100 * in_u["close"].isna().mean(), 3),
        "member_days_suspended_pct": round(100 * in_u["suspended"].mean(), 3),
        "member_days_limit_up_pct": round(100 * in_u["limit_up"].mean(), 3),
        "member_days_limit_down_pct": round(100 * in_u["limit_down"].mean(), 3),
        "member_days_abs_return_gt_25pct": int(big.sum()),
        "index_days": int(idx["close"].notna().sum()),
        "index_level_first_last": [round(float(idx["level"].iloc[0]), 2), round(float(idx["level"].iloc[-1]), 2)],
    }
    (OUT / "data_quality.json").write_text(json.dumps(q, indent=2))
    print(json.dumps(q, indent=2))
    if big.any():
        print(in_u.loc[big, ["ret_1d", "raw_close", "volume"]].head(10))


if __name__ == "__main__":
    main()
