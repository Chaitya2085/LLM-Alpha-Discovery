"""Phase 4: the feedback loop, with one command (run run_india.py first).

    python run_phase4.py --providers gemini groq openrouter      # India (default), 3 rounds x 15 factors
    python run_phase4.py --providers gemini --rounds 4
    python run_phase4.py --market china --providers gemini
    python run_phase4.py                                          # no API calls: rebuild tables from the library

Each LLM is a Researcher. Round 0 is its Phase 2 factors; then, each round, it proposes
new factors, the weekly judge tests them (buy the top fifth, hold 5 days, after costs) and
the Critic writes feedback for the next round. A control arm asks the same LLM without
feedback, so we can see whether feedback really helps (--no-control halves the API calls).

Steps:
  1. tests (factor language, LLM client, Phase 4)
  2. feedback rounds for each provider      -> factors/library.jsonl, llm_runs/phase4/
  3. tables                                 -> results/<market>/phase4_*.csv / .json, phase4_feedback/
  4. chart                                  -> figures/<prefix>phase4_rounds.png
  5. fill the Phase 4 tables in README.md
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
from llm import PROVIDERS  # noqa: E402


def run(args: list[str], cwd: Path = SRC, check: bool = True) -> int:
    t = time.time()
    r = subprocess.run([sys.executable, *args], cwd=cwd)
    if check and r.returncode != 0:
        sys.exit(r.returncode)
    print(f"    done in {time.time() - t:.0f}s", flush=True)
    return r.returncode


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--market", default="india", choices=["india", "china"])
    ap.add_argument("--providers", nargs="*", default=[], choices=list(PROVIDERS))
    ap.add_argument("--model", default=None, help="model (only with a single provider)")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--per-round", type=int, default=15)
    ap.add_argument("--no-control", action="store_true")
    a = ap.parse_args()
    if sys.version_info < (3, 10):
        sys.exit("Python 3.10 or newer is required.")
    proc = ROOT / "data" / "processed" / ("india" if a.market == "india" else "")
    if not (proc / "prices.parquet").exists():
        sys.exit(f"No {a.market} data yet. Run first:  python {'run_india.py' if a.market == 'india' else 'run_phase1.py'}")

    print("\n=== Step 1/5: tests ===", flush=True)
    for t in ("test_dsl.py", "test_llm.py", "test_phase4.py"):
        run([str(ROOT / "tests" / t)], cwd=ROOT)

    print("\n=== Step 2/5 + 3/5: feedback rounds and tables ===", flush=True)
    cmd = ["feedback_loop.py", "--market", a.market, "--rounds", str(a.rounds), "--per-round", str(a.per_round)]
    if a.providers:
        cmd += ["--providers", *a.providers]
    else:
        cmd += ["--summarize-only"]
        print("    (no --providers given: no API calls, rebuilding tables from the library)")
    if a.model:
        cmd += ["--model", a.model]
    if a.no_control:
        cmd += ["--no-control"]
    run(cmd)

    print("\n=== Step 4/5: chart ===", flush=True)
    run(["plot_phase4.py", "--market", a.market])

    print("\n=== Step 5/5: update README tables ===", flush=True)
    run(["report.py"])
    print(f"\nAll done. Tables in results/{'india/' if a.market == 'india' else ''}phase4_*, "
          "the Critic's feedback in phase4_feedback/, chart in figures/, README updated.")


if __name__ == "__main__":
    main()
