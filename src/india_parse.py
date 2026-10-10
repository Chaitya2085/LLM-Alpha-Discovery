"""Read NSE files into tidy tables. Handles both bhavcopy formats.

Equity output columns (one row per stock; EQ series only unless all_series=True):
  date, symbol, isin, series, open, high, low, close, prevclose, volume, value
  (prices in rupees, unadjusted; volume in shares; value = traded value in rupees)

Index output columns: name, open, high, low, close
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd

# column names in the two formats -> our names
EQUITY_COLS = {
    "symbol": ["SYMBOL", "TckrSymb"],
    "series": ["SERIES", "SctySrs"],
    "open": ["OPEN", "OpnPric"],
    "high": ["HIGH", "HghPric"],
    "low": ["LOW", "LwPric"],
    "close": ["CLOSE", "ClsPric"],
    "prevclose": ["PREVCLOSE", "PrvsClsgPric"],
    "volume": ["TOTTRDQTY", "TtlTradgVol"],
    "value": ["TOTTRDVAL", "TtlTrfVal"],
    "isin": ["ISIN"],
    "date": ["TIMESTAMP", "TradDt"],
}
SEGMENT_COLS = ["Sgmt"]           # UDiFF: keep the cash market rows only
INSTRUMENT_COLS = ["FinInstrmTp"]  # UDiFF: STK = equity


def _pick(cols: list[str], names: list[str]) -> str | None:
    low = {c.strip().lower(): c for c in cols}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None


# equity series whose closes count for price continuity (EQ = normal, BE/BZ = trade-to-trade)
CONTINUITY_SERIES = ("EQ", "BE", "BZ")


def read_equity_bytes(raw: bytes, fallback_date: pd.Timestamp | None = None,
                      all_series: bool = False) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        df = pd.read_csv(z.open(name), dtype=str, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    seg = _pick(df.columns, SEGMENT_COLS)
    if seg:
        df = df[df[seg].str.strip().str.upper() == "CM"]
    ins = _pick(df.columns, INSTRUMENT_COLS)
    if ins:
        df = df[df[ins].str.strip().str.upper() == "STK"]
    out = pd.DataFrame()
    for ours, theirs in EQUITY_COLS.items():
        col = _pick(df.columns, theirs)
        if col is None:
            if ours in ("isin", "date"):
                out[ours] = None
                continue
            raise ValueError(f"column for '{ours}' not found; columns are {list(df.columns)[:12]}...")
        out[ours] = df[col].str.strip() if df[col].dtype == object else df[col]
    for c in ["open", "high", "low", "close", "prevclose", "volume", "value"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["series"] = out["series"].str.upper()
    keep = out["series"].isin(CONTINUITY_SERIES) if all_series else out["series"] == "EQ"
    out = out[keep].drop(columns=["date"]).copy()
    out["date"] = fallback_date  # the trading day comes from the file name
    return out.reset_index(drop=True)


def read_equity_file(path: str | Path, all_series: bool = False) -> pd.DataFrame:
    path = Path(path)
    stem = path.name.split(".")[0]
    fallback = pd.Timestamp(stem) if stem.isdigit() and len(stem) == 8 else None
    return read_equity_bytes(path.read_bytes(), fallback, all_series)


def read_index_file(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    name = _pick(df.columns, ["Index Name"])
    cols = {k: _pick(df.columns, v) for k, v in {
        "open": ["Open Index Value"], "high": ["High Index Value"],
        "low": ["Low Index Value"], "close": ["Closing Index Value"]}.items()}
    if name is None or cols["close"] is None:
        raise ValueError(f"unexpected index file columns: {list(df.columns)}")
    out = pd.DataFrame({"name": df[name].str.strip()})
    for k, c in cols.items():
        out[k] = pd.to_numeric(df[c].str.replace(",", ""), errors="coerce") if c else float("nan")
    return out


INDEX_ALIASES = {
    "Nifty 50": ["nifty 50", "cnx nifty", "s&p cnx nifty", "nifty"],
    "Nifty 200": ["nifty 200", "cnx 200", "s&p cnx 200"],
}


def canonical_index(name: str) -> str | None:
    n = name.strip().lower()
    for canon, aliases in INDEX_ALIASES.items():
        if n in aliases:
            return canon
    return None
