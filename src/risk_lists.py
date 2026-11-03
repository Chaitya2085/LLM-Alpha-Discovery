"""NSE lists the Risk Officer uses: surveillance flags per stock, and the sector of each Nifty 200 stock.

    python risk_lists.py --test      # check which NSE addresses work on your network
    python risk_lists.py             # download today's files

  data/raw/india/surveillance/REG_IND<DDMMYY>.csv   NSE's daily "regulatory indicators" file: for every
        stock, whether it is under GSM, long/short-term ASM, ESM, insolvency (IRP), high promoter pledge...
        (file layout: NSE's "REG_IND File Structure" document)
  data/raw/india/sectors/ind_nifty200list.csv       Nifty 200 members with their industry

NSE moves report files from time to time. If the download can't find REG_IND, download it by hand from
nseindia.com -> All Reports -> Capital Market (it is called "Regulatory Indicators" / REG_IND) and save
it in data/raw/india/surveillance/. The Risk Officer reads whatever is in that folder.
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

import india_download as nd

ROOT = Path(__file__).resolve().parents[1]
SURV = ROOT / "data" / "raw" / "india" / "surveillance"
SECTORS = ROOT / "data" / "raw" / "india" / "sectors"

# field positions in REG_IND (1-based in NSE's document) and what a value other than 100 means
FIELDS = {2: "symbol", 6: "GSM", 7: "long-term ASM", 9: "insolvency process (IRP)", 10: "short-term ASM",
          15: "high promoter pledge", 19: "ESM"}


def reg_ind_urls(d: date) -> list[str]:
    tag = f"{d:%d%m%y}"
    paths = [f"/content/equities/REG_IND{tag}.csv", f"/content/cm/REG_IND{tag}.csv",
             f"/content/equities/REG1_IND{tag}.csv"]
    return [b + p for p in paths for b in nd.BASES]


def sector_urls() -> list[str]:
    return [f"{b}/content/indices/ind_nifty200list.csv" for b in nd.BASES]


def parse_reg_ind(path: Path) -> dict[str, list[str]]:
    """symbol -> list of measures it is under (empty = clean). Reads by position, header or not."""
    raw = pd.read_csv(path, header=None, dtype=str, skipinitialspace=True).fillna("")
    first = raw.iloc[0].str.lower().tolist()
    if any("symbol" in x for x in first):
        raw = raw.iloc[1:]
    out: dict[str, list[str]] = {}
    for _, r in raw.iterrows():
        vals = [str(v).strip() for v in r.tolist()]
        if len(vals) < 19:
            continue
        sym = vals[1].upper()
        flags = []
        for pos, label in FIELDS.items():
            if label == "symbol":
                continue
            v = vals[pos - 1]
            if v and v != "100":
                flags.append(f"{label} (stage {v})" if label in ("GSM", "long-term ASM", "short-term ASM", "ESM")
                             else label)
        out[sym] = flags
    return out


def latest_reg_ind(folder: Path = SURV) -> tuple[Path | None, dict]:
    files = sorted(folder.glob("REG*_IND*.csv"), key=lambda f: f.stat().st_mtime) if folder.exists() else []
    if not files:
        return None, {}
    return files[-1], parse_reg_ind(files[-1])


def sectors(folder: Path = SECTORS) -> dict[str, str]:
    f = folder / "ind_nifty200list.csv"
    if not f.exists():
        return {}
    df = pd.read_csv(f, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]
    if "symbol" not in df.columns or "industry" not in df.columns:
        return {}
    return dict(zip(df["symbol"].str.strip().str.upper(), df["industry"].str.strip()))


def download(test: bool = False) -> dict:
    c = nd.Client(pause=0.5)
    c.warm_up()
    got = {"reg_ind": None, "sectors": None}
    for back in range(0, 7):
        d = date.today() - timedelta(days=back)
        if d.weekday() >= 5:
            continue
        for url in reg_ind_urls(d):
            try:
                body = c.get(url, retries=2)
            except (nd.Blocked, nd.Busy) as e:
                print(f"    {url}: {e}")
                continue
            if body and b"," in body[:500] and b"<html" not in body[:500].lower():
                SURV.mkdir(parents=True, exist_ok=True)
                (SURV / url.rsplit("/", 1)[-1]).write_bytes(body)
                got["reg_ind"] = url
                break
        if got["reg_ind"]:
            break
    for url in sector_urls():
        try:
            body = c.get(url, retries=2)
        except (nd.Blocked, nd.Busy):
            continue
        if body and b"Symbol" in body[:300]:
            SECTORS.mkdir(parents=True, exist_ok=True)
            (SECTORS / "ind_nifty200list.csv").write_bytes(body)
            got["sectors"] = url
            break
    if test or not all(got.values()):
        print(f"    surveillance file (REG_IND): {got['reg_ind'] or 'NOT FOUND - download it by hand, see the top of this file'}")
        print(f"    Nifty 200 sectors: {got['sectors'] or 'NOT FOUND'}")
    if got["reg_ind"]:
        _, flags = latest_reg_ind()
        n = sum(1 for v in flags.values() if v)
        print(f"    {len(flags)} stocks in the surveillance file, {n} under at least one measure")
    return got


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", action="store_true")
    download(ap.parse_args().test)
