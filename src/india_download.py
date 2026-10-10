"""Download NSE daily data: the equity "bhavcopy" and the index closing file.

    python src/india_download.py --test          # check two sample days first
    python src/india_download.py                 # 2014-01-01 to today (resumable)
    python src/india_download.py --start 2016-01-01 --workers 2

Files (saved exactly as NSE serves them):
  data/raw/india/equities/YYYY/YYYYMMDD.csv.zip   every listed stock that day
  data/raw/india/indices/YYYY/YYYYMMDD.csv        closing values of all NSE indices

Formats: until early July 2024 NSE published cmDDMONYYYYbhav.csv.zip; from 8 July 2024
the "UDiFF" file BhavCopy_NSE_CM_0_0_0_YYYYMMDD_F_0000.csv.zip, with renamed columns.
Both are handled. Weekends are skipped; days NSE has no file for (holidays) are
remembered so a re-run doesn't ask again. Already-downloaded days are skipped.

Be polite: NSE rate-limits heavy scraping, so the default is 2 workers with a pause.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import http.cookiejar
import io
import json
import os
import threading
import time
import urllib.error
import urllib.request
import zipfile
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "india"
BASES = [os.environ.get("NSE_BASE_URL", "https://nsearchives.nseindia.com"),
         *([] if os.environ.get("NSE_BASE_URL") else ["https://archives.nseindia.com"])]
HOME = os.environ.get("NSE_HOME_URL", "https://www.nseindia.com")
UDIFF_FROM = date(2024, 7, 8)
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}
MON = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


class Blocked(RuntimeError):
    """NSE refused us (401/403): stop the whole run."""


class Busy(RuntimeError):
    """NSE kept answering 'busy' (429/5xx) or the network kept failing: retry that day later."""


TRANSIENT = {429, 500, 502, 503, 504}
BACKOFF = [5, 15, 30, 60]       # seconds to wait before each retry when NSE is busy
MAX_PAUSE = 3.0                 # when NSE struggles, space requests out up to this many seconds
STOP_AFTER = 12                 # this many failed days in a row = NSE is down; stop and resume later


def equity_urls(d: date) -> list[str]:
    old = f"/content/historical/EQUITIES/{d.year}/{MON[d.month - 1]}/cm{d.day:02d}{MON[d.month - 1]}{d.year}bhav.csv.zip"
    new = f"/content/cm/BhavCopy_NSE_CM_0_0_0_{d:%Y%m%d}_F_0000.csv.zip"
    if d >= UDIFF_FROM + timedelta(days=7):
        paths = [new]
    elif d < UDIFF_FROM - timedelta(days=7):
        paths = [old]
    else:  # around the switch, try both
        paths = [new, old] if d >= UDIFF_FROM else [old, new]
    return [b + p for p in paths for b in BASES]


def index_urls(d: date) -> list[str]:
    return [f"{b}/content/indices/ind_close_all_{d:%d%m%Y}.csv" for b in BASES]


class Client:
    def __init__(self, pause: float = 0.35):
        self.jar = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self.jar))
        self.pause = self.base_pause = pause
        self.lock = threading.Lock()
        self.last = 0.0
        self.stop = threading.Event()   # set when NSE looks down: remaining days are skipped

    def warm_up(self) -> None:
        """Visit the NSE home page once to pick up cookies (some archive hosts want them)."""
        try:
            self.get(HOME, retries=1)
        except Exception:  # noqa: BLE001 - optional step
            pass

    def get(self, url: str, retries: int | None = None) -> bytes | None:
        """Return the body; None for 'no file' (404). Raise Blocked on 401/403, Busy when NSE
        stays busy (429/5xx) or the network keeps failing after several patient retries."""
        retries = len(BACKOFF) + 1 if retries is None else retries
        for attempt in range(retries):
            with self.lock:  # global pacing across workers
                wait = self.pause - (time.time() - self.last)
                if wait > 0:
                    time.sleep(wait)
                self.last = time.time()
            req = urllib.request.Request(url, headers=HEADERS)
            delay = BACKOFF[min(attempt, len(BACKOFF) - 1)] if BACKOFF else 0
            try:
                with self.opener.open(req, timeout=30) as r:
                    body = r.read()
                with self.lock:  # things are fine again: drift back to the normal pace
                    self.pause = max(self.base_pause, self.pause * 0.9)
                return body
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    return None
                if e.code in (401, 403):
                    raise Blocked(f"HTTP {e.code} from {url}") from None
                if e.code not in TRANSIENT:
                    raise Busy(f"HTTP {e.code} from {url}") from None
                why = f"HTTP {e.code}"
                ra = e.headers.get("Retry-After") if e.headers else None
                if ra and ra.strip().isdigit():
                    delay = max(delay, min(int(ra), 120))
                self._slow_down()
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                why = type(e).__name__
                self._slow_down()
            if attempt == retries - 1:
                raise Busy(f"{why} from {url} after {retries} tries")
            time.sleep(delay)
        return None

    def _slow_down(self) -> None:
        """NSE is struggling: space all requests out more (up to MAX_PAUSE seconds apart)."""
        with self.lock:
            self.pause = min(max(self.pause * 1.5, self.base_pause + 0.25), MAX_PAUSE)


def _valid_zip(b: bytes) -> bool:
    try:
        with zipfile.ZipFile(io.BytesIO(b)) as z:
            return any(n.lower().endswith(".csv") for n in z.namelist())
    except zipfile.BadZipFile:
        return False


def _valid_csv(b: bytes) -> bool:
    head = b[:300].decode(errors="ignore").lower()
    return "index" in head and "<html" not in head


def _first_file(client: Client, urls: list[str], valid) -> tuple[str, bytes | None]:
    """Try each URL. 'ok' + body, 'none' (no file anywhere), 'bad_file', or 'error' (NSE busy:
    we don't know whether the file exists, so the day must be retried, never saved as a holiday)."""
    status = "none"
    for url in urls:
        try:
            body = client.get(url)
        except Busy:
            status = "error"
            continue
        if body and valid(body):
            return "ok", body
        if body and status == "none":
            status = "bad_file"
    return status, None


def fetch_day(client: Client, d: date) -> dict:
    eq_path = RAW / "equities" / str(d.year) / f"{d:%Y%m%d}.csv.zip"
    ix_path = RAW / "indices" / str(d.year) / f"{d:%Y%m%d}.csv"
    out = {"date": d.isoformat(), "equity": "have" if eq_path.exists() else None,
           "index": "have" if ix_path.exists() else None}
    if client.stop.is_set():
        out["equity"] = "have" if eq_path.exists() else "error"
        out["index"] = out["index"] or "error"
        return out
    if not eq_path.exists():
        out["equity"], body = _first_file(client, equity_urls(d), _valid_zip)
        if body:
            eq_path.parent.mkdir(parents=True, exist_ok=True)
            eq_path.write_bytes(body)
    if out["equity"] in ("ok", "have") and not ix_path.exists():
        out["index"], body = _first_file(client, index_urls(d), _valid_csv)
        if body:
            ix_path.parent.mkdir(parents=True, exist_ok=True)
            ix_path.write_bytes(body)
    return out


def weekdays(start: date, end: date):
    d = start
    while d <= end:
        if d.weekday() < 5:
            yield d
        d += timedelta(days=1)


def run_test(client: Client) -> None:
    """Download two sample days (one per file format) and show what came back."""
    import pandas as pd
    from india_parse import read_equity_file, read_index_file

    today = date.today()
    samples = [date(2016, 6, 1), date(2023, 6, 1)]
    recent = [d for d in weekdays(today - timedelta(days=12), today - timedelta(days=1))]
    samples += recent[-3:]
    for d in samples:
        try:
            r = fetch_day(client, d)
        except Blocked as e:
            print(f"  {d}: BLOCKED ({e}). NSE refused the request; see README troubleshooting.")
            return
        eq = RAW / "equities" / str(d.year) / f"{d:%Y%m%d}.csv.zip"
        ix = RAW / "indices" / str(d.year) / f"{d:%Y%m%d}.csv"
        line = f"  {d} equity={r['equity']:<8} index={r['index'] or '-':<6}"
        if eq.exists():
            df = read_equity_file(eq)
            line += f" -> {len(df)} EQ stocks, e.g. " + ", ".join(df['symbol'].head(3))
        if ix.exists():
            ixd = read_index_file(ix)
            n50 = ixd[ixd['name'].str.lower() == 'nifty 50']
            if len(n50):
                line += f"; Nifty 50 close {float(n50['close'].iloc[0]):,.2f}"
        print(line)
    _ = pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2014-01-01")
    ap.add_argument("--end", default=None, help="default: yesterday")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--pause", type=float, default=0.35, help="seconds between requests")
    ap.add_argument("--test", action="store_true", help="download 5 sample days and report")
    a = ap.parse_args()

    client = Client(pause=a.pause)
    client.warm_up()
    if a.test:
        run_test(client)
        return

    start = date.fromisoformat(a.start)
    end = date.fromisoformat(a.end) if a.end else date.today() - timedelta(days=1)
    holidays_file = RAW / "no_file_days.json"
    no_file = set(json.loads(holidays_file.read_text())) if holidays_file.exists() else set()
    days = [d for d in weekdays(start, end)
            if d.isoformat() not in no_file
            and not ((RAW / "equities" / str(d.year) / f"{d:%Y%m%d}.csv.zip").exists()
                     and (RAW / "indices" / str(d.year) / f"{d:%Y%m%d}.csv").exists())]
    print(f"    {len(days)} weekdays to fetch ({start} to {end}); {len(no_file)} known no-file days skipped",
          flush=True)
    counts = {"ok": 0, "have": 0, "none": 0, "bad_file": 0, "error": 0}
    failed: list[str] = []
    t0 = time.time()
    done = 0
    in_a_row = 0
    try:
        with cf.ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
            futs = {ex.submit(fetch_day, client, d): d for d in days}
            for f in cf.as_completed(futs):
                d = futs[f]
                try:
                    r = f.result()
                except Blocked:
                    raise
                except Exception as e:  # noqa: BLE001 - one bad day must not kill a 40-minute run
                    r = {"equity": "error", "index": None, "why": str(e)}
                st = r["equity"]
                if r.get("index") == "error":
                    st = "error"      # the stock file came but the index file didn't: retry the day
                counts[st] = counts.get(st, 0) + 1
                if st == "error":
                    failed.append(d.isoformat())
                    in_a_row += 1
                    if in_a_row >= STOP_AFTER and not client.stop.is_set():
                        client.stop.set()
                        print(f"\n    NSE has been busy/unreachable for {in_a_row} days in a row; "
                              "pausing the download (finishing the requests in flight).", flush=True)
                elif r["equity"] in ("ok", "none", "have"):
                    in_a_row = 0
                # only remember a 'no file' day once it is safely in the past
                if st == "none" and d < date.today() - timedelta(days=5):
                    no_file.add(d.isoformat())
                done += 1
                if not client.stop.is_set() and (done % 100 == 0 or done == len(days)):
                    rate = done / max(time.time() - t0, 1e-9)
                    eta = (len(days) - done) / max(rate, 1e-9) / 60
                    extra = f" retry-later={counts['error']}" if counts["error"] else ""
                    print(f"    {done}/{len(days)} days  new={counts['ok']} no-file={counts['none']}{extra}  "
                          f"~{eta:.0f} min left", flush=True)
    except Blocked as e:
        print(f"\n    NSE blocked the requests: {e}\n    Wait 10-15 minutes and run the same command again "
              "(it resumes), or use --workers 1 --pause 1.0", flush=True)
        raise SystemExit(2)
    finally:
        RAW.mkdir(parents=True, exist_ok=True)
        # holidays are ~5% of weekdays; far more 'no file' answers means NSE was refusing us,
        # so don't record them as holidays (they'll be retried next run)
        new_none = counts.get("none", 0)
        if done > 50 and new_none / done > 0.25:
            print(f"    warning: {new_none} of {done} days had no file, far more than holidays explain. "
                  "NSE may be blocking; these days will be retried next run.", flush=True)
        else:
            holidays_file.write_text(json.dumps(sorted(no_file)))
    if counts.get("bad_file"):
        print(f"    note: {counts['bad_file']} days returned something that wasn't a data file (likely a "
              "block page); re-run to retry them", flush=True)
    print(f"    done: {counts['ok']} new days, {counts['none']} holidays/no-file days", flush=True)
    if failed:
        (RAW / "failed_days.json").write_text(json.dumps(sorted(failed)))
        print(f"\n    {len(failed)} days could not be downloaded because NSE was busy (HTTP 503/429) or the "
              f"network dropped (first: {min(failed)}, last: {max(failed)}).\n"
              "    Everything else is saved. Wait 10-15 minutes and run the same command again; it only "
              "fetches what's missing.\n    If it keeps happening, go slower:  --workers 1 --pause 1.0",
              flush=True)
        raise SystemExit(3)
    (RAW / "failed_days.json").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
