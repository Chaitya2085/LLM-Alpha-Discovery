"""Phase 5: the prediction model, with one command (run run_india.py, and ideally run_phase4.py, first).

    python run_phase5.py                    # India (default): walk-forward test 2017-2020
    python run_phase5.py --market china

Steps:
  1. tests (Phase 5)
  2. train and test the weekly model, Alpha158 only and Alpha158 + LLM factors
                                     -> results/<market>/phase5_*.csv / .json / .parquet
  3. charts                          -> figures/<prefix>phase5_growth.png, phase5_calibration.png
  4. fill the Phase 5 tables in README.md

Nothing after 2020 is loaded: 2021 onward is kept for the Phase 6 final test.
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
    ap.add_argument("--market", default="india", choices=["india", "china"])
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    if sys.version_info < (3, 10):
        sys.exit("Python 3.10 or newer is required.")
    proc = ROOT / "data" / "processed" / ("india" if a.market == "india" else "")
    if not (proc / "alpha158.parquet").exists():
        sys.exit(f"No {a.market} features yet. Run first:  python {'run_india.py' if a.market == 'india' else 'run_phase1.py'}")

    print("\n=== Step 1/4: tests ===", flush=True)
    run([str(ROOT / "tests" / "test_phase5.py")], cwd=ROOT)
    print("\n=== Step 2/4: train and test the weekly model (5-15 minutes) ===", flush=True)
    run(["phase5_model.py", "--market", a.market, "--threads", str(a.threads)])
    print("\n=== Step 3/4: charts ===", flush=True)
    run(["plot_phase5.py", "--market", a.market])
    print("\n=== Step 4/4: update README tables ===", flush=True)
    run(["report.py"])
    print("\nAll done. Results in results/" + ("india/" if a.market == "india" else "") +
          "phase5_*, charts in figures/, README updated.")


if __name__ == "__main__":
    main()
