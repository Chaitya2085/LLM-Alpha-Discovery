"""Phase 6: the final exam, with one command each step.

    python run_phase6.py --freeze     # 1. freeze every setting and the pass marks (then commit + push)
    python run_phase6.py              # 2. the exam on 2021 onward, the leakage test, chart, README

The exam runs once. It refuses to start until config/phase6_frozen.json is committed to git (so the
promise is timestamped before the result exists), and a second run must give a reason, which is
recorded: python run_phase6.py --rerun-reason "why"

Steps (2):
  1. tests (Phase 6)
  2. the exam: both frozen versions, each year predicted only from earlier years
                                  -> results/<market>/phase6_*.csv / .json
  3. leakage test: do LLM factors fade right after the model's knowledge cutoff?
                                  -> results/<market>/leakage*.csv / .json
  4. chart                        -> figures/<prefix>phase6_exam.png
  5. fill the Phase 6 tables in README.md
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
FROZEN = ROOT / "config" / "phase6_frozen.json"


def run(args: list[str], cwd: Path = SRC, check: bool = True) -> int:
    t = time.time()
    r = subprocess.run([sys.executable, *args], cwd=cwd)
    if check and r.returncode != 0:
        sys.exit(r.returncode)
    print(f"    done in {time.time() - t:.0f}s", flush=True)
    return r.returncode


def frozen_is_committed() -> tuple[bool, str]:
    try:
        tracked = subprocess.run(["git", "log", "-1", "--format=%h %cd", "--", str(FROZEN)], cwd=ROOT,
                                 capture_output=True, text=True)
        dirty = subprocess.run(["git", "status", "--porcelain", "--", str(FROZEN)], cwd=ROOT,
                               capture_output=True, text=True)
    except FileNotFoundError:
        return True, "git not found; skipping the check"
    if tracked.returncode != 0:
        return True, "not a git folder; skipping the check"
    if not tracked.stdout.strip():
        return False, "config/phase6_frozen.json has never been committed"
    if dirty.stdout.strip():
        return False, "config/phase6_frozen.json has changes that are not committed"
    return True, f"frozen settings committed in {tracked.stdout.strip()}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--market", default="india", choices=["india", "china"])
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--rerun-reason", default=None)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    if sys.version_info < (3, 10):
        sys.exit("Python 3.10 or newer is required.")

    if a.freeze:
        run(["phase6_exam.py", "--market", a.market, "--freeze"])
        return
    if not FROZEN.exists():
        sys.exit("Freeze first:  python run_phase6.py --freeze   (then commit and push the frozen file)")
    ok, why = frozen_is_committed()
    print(f"    {why}")
    if not ok:
        sys.exit("Commit the frozen settings before the exam, so they are timestamped first:\n"
                 "    git add config/phase6_frozen.json && git commit -m \"Phase 6: freeze settings\" && git push")

    print("\n=== Step 1/5: tests ===", flush=True)
    run([str(ROOT / "tests" / "test_phase6.py")], cwd=ROOT)
    print("\n=== Step 2/5: the final exam (10-20 minutes) ===", flush=True)
    cmd = ["phase6_exam.py", "--market", a.market, "--run", "--threads", str(a.threads)]
    if a.rerun_reason:
        cmd += ["--rerun-reason", a.rerun_reason]
    run(cmd)
    print("\n=== Step 3/5: leakage test ===", flush=True)
    run(["leakage_test.py", "--market", a.market])
    print("\n=== Step 4/5: chart ===", flush=True)
    run(["plot_phase6.py", "--market", a.market])
    print("\n=== Step 5/5: update README tables ===", flush=True)
    run(["report.py"])
    print("\nAll done. Results in results/" + ("india/" if a.market == "india" else "") + "phase6_*, leakage_*, "
          "chart in figures/, README updated. Next: start the weekly ledger (python run_ledger.py).")


if __name__ == "__main__":
    main()
