"""Phase 3: run the whole pipeline on the Indian market (NSE) with one command.

    python run_india.py --test        # first: check that NSE downloads work on your network
    python run_india.py               # everything (first run downloads ~12 years: 30-60 min)

Options: --start 2014-01-01  --workers 2  --pause 0.35  --skip-download (use files already downloaded)

Steps:
  1. download NSE bhavcopies + index closes   -> data/raw/india/        (resumable)
  2. build the India dataset                   -> data/processed/india/
  3. Alpha158 features
  4. baselines: Nifty 200 buy-and-hold, classic factors, walk-forward LightGBM -> results/india/
  5. chart                                     -> figures/india_phase1_growth.png
  6. score every LLM factor on Indian data     -> results/india/phase2_*.csv
  7. charts                                    -> figures/india_phase2_*.png
  8. China vs India: do the factors travel?    -> figures/india_transfer.png
  9. fill the India tables in README.md
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
STEPS = 9


def step(n: int, title: str) -> None:
    print(f"\n=== Step {n}/{STEPS}: {title} ===", flush=True)


def run(args: list[str], check: bool = True) -> int:
    t = time.time()
    r = subprocess.run([sys.executable, *args], cwd=SRC)
    if check and r.returncode != 0:
        sys.exit(r.returncode)
    print(f"    done in {time.time() - t:.0f}s", flush=True)
    return r.returncode


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", action="store_true", help="only check NSE downloads on a few sample days")
    ap.add_argument("--start", default="2014-01-01")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--pause", type=float, default=0.35, help="seconds between NSE requests")
    ap.add_argument("--skip-download", action="store_true")
    a = ap.parse_args()
    if sys.version_info < (3, 10):
        sys.exit("Python 3.10 or newer is required.")

    if a.test:
        print("Checking NSE downloads on sample days (one per file format)...", flush=True)
        run(["india_download.py", "--test"])
        print("\nIf you see stock counts and a Nifty 50 close above, run:  python run_india.py")
        return

    step(1, "download NSE data (resumable; first run 30-60 min)")
    if a.skip_download:
        print("    skipped (--skip-download)")
    else:
        code = run(["india_download.py", "--start", a.start, "--workers", str(a.workers),
                    "--pause", str(a.pause)], check=False)
        if code != 0:
            print("\n    Download not complete yet (files so far are saved). Run  python run_india.py  again "
                  "in 10-15 minutes;\n    it picks up where it stopped.")
            sys.exit(code)

    step(2, "build the India dataset")
    run(["build_india.py", "--start", a.start])

    step(3, "compute Alpha158 features")
    run(["build_features.py", "--market", "india"])

    step(4, "run baselines (5-10 minutes)")
    run(["run_baselines.py", "--market", "india"])

    step(5, "draw chart")
    run(["plot_phase1.py", "--market", "india"])

    step(6, "score the LLM factors on Indian data")
    if not (ROOT / "factors" / "library.jsonl").exists():
        run(["generate_factors.py", "--provider", "replay"])
        if any((ROOT / "llm_runs").glob("*.json")):
            run(["generate_factors.py", "--provider", "replay", "--replay-dir", str(ROOT / "llm_runs")])
    run(["score_factors.py", "--market", "india"])

    step(7, "draw charts")
    run(["plot_phase2.py", "--market", "india"])

    step(8, "China vs India")
    if (ROOT / "results" / "phase2_factor_scores.csv").exists():
        run(["compare_markets.py"])
    else:
        print("    skipped: score the factors on China first (python run_phase2.py)")

    step(9, "update README tables")
    run(["report.py"])
    print("\nAll done. India results in results/india/, charts in figures/india_*.png, README updated.")


if __name__ == "__main__":
    main()
