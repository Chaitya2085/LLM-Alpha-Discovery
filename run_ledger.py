"""The weekly prediction ledger (run once a week, after Friday's close).

    python run_ledger.py                  # download the latest NSE days, rebuild, record, reveal, score
    python run_ledger.py --skip-download  # use the data you already have

Each run fingerprints this week's predictions in ledger/commitments.csv (commit and push it: git's
timestamp proves the prediction came first), reveals entries older than 31 days into
ledger/revealed/, scores the revealed weeks, checks the whole chain, and updates the README.
The predictions themselves stay in ledger/private/, which is never committed.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"


def run(args: list[str], cwd: Path = SRC) -> None:
    t = time.time()
    r = subprocess.run([sys.executable, *args], cwd=cwd)
    if r.returncode != 0:
        sys.exit(r.returncode)
    print(f"    done in {time.time() - t:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    if not (ROOT / "config" / "phase6_frozen.json").exists():
        sys.exit("Freeze Phase 6 first:  python run_phase6.py --freeze")
    if not a.skip_download:
        print("\n=== Step 1/4: latest NSE data ===", flush=True)
        run(["india_download.py"])
        run(["build_india.py"])
        run(["build_features.py", "--market", "india"])
    print("\n=== Step 2/4: record this week's predictions ===", flush=True)
    run(["ledger.py", "record", "--threads", str(a.threads)])
    print("\n=== Step 3/4: reveal, score and check the ledger ===", flush=True)
    run(["ledger.py", "reveal"])
    run(["ledger.py", "score"])
    run(["ledger.py", "verify"])
    print("\n=== Step 4/4: update README ===", flush=True)
    run(["report.py"])
    print("\nDone. Commit and push ledger/commitments.csv now (the private folder stays on your computer):\n"
          "    git add ledger README.md && git commit -m \"Ledger: week of <date>\" && git push")


if __name__ == "__main__":
    main()
