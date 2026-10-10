"""Minimal reader for Qlib's binary data format (no qlib install needed).

Each <field>.day.bin is little-endian float32. The first value is the index of
the first trading day in calendars/day.txt; the rest are daily values.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FIELDS = ["open", "high", "low", "close", "volume", "amount", "vwap", "factor", "adjclose"]


class QlibReader:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.calendar = pd.DatetimeIndex(
            pd.read_csv(self.root / "calendars" / "day.txt", header=None)[0]
        )

    def instruments(self, universe: str) -> pd.DataFrame:
        """Point-in-time membership: one row per (code, start, end) spell."""
        df = pd.read_csv(
            self.root / "instruments" / f"{universe}.txt",
            sep="\t", header=None, names=["code", "start", "end"],
        )
        df["start"] = pd.to_datetime(df["start"])
        df["end"] = pd.to_datetime(df["end"])
        return df

    def series(self, code: str, field: str) -> pd.Series:
        path = self.root / "features" / code.lower() / f"{field}.day.bin"
        if not path.exists():
            return pd.Series(dtype="float32")
        arr = np.fromfile(path, dtype="<f4")
        if arr.size < 2:
            return pd.Series(dtype="float32")
        start = int(arr[0])
        vals = arr[1:]
        idx = self.calendar[start : start + len(vals)]
        return pd.Series(vals[: len(idx)], index=idx, name=code)

    def panel(self, codes, fields=FIELDS, start=None, end=None) -> pd.DataFrame:
        """Long panel indexed by (date, code) with one column per field."""
        frames = []
        for code in codes:
            cols = {f: self.series(code, f) for f in fields}
            df = pd.DataFrame(cols)
            if df.empty:
                continue
            if start is not None:
                df = df.loc[df.index >= pd.Timestamp(start)]
            if end is not None:
                df = df.loc[df.index <= pd.Timestamp(end)]
            df["code"] = code.upper()
            frames.append(df)
        out = pd.concat(frames)
        out.index.name = "date"
        return out.set_index("code", append=True).sort_index()
