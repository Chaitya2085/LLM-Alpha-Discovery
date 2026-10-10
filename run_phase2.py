"""Run all of Phase 2 with one command (run Phase 1 first).

    python run_phase2.py --provider gemini                 # one free provider
    python run_phase2.py --provider gemini groq cerebras   # several, compared side by side
    python run_phase2.py                                   # no API calls: re-score what's in the library
    python run_phase2.py --provider ollama --model qwen3:8b

Options:
    --batches 3 --per-batch 20   how many factors to ask each provider for
    --model NAME                 model for the provider (only with a single --provider)
    --no-seed                    leave out the 40 seed factors written in the chat session

Steps:
  1. run the tests (factor language + LLM client)
  2. load the saved seed batch, then ask each provider for new factors
  3. score every valid factor on 2014-2020 only     -> results/phase2_*.csv
  4. draw the charts                                -> figures/phase2_*.png
  5. fill the Phase 2 tables in README.md
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
    ap.add_argument("--provider", nargs="*", default=[], choices=list(PROVIDERS))
    ap.add_argument("--model", default=None)
    ap.add_argument("--batches", type=int, default=3)
    ap.add_argument("--per-batch", type=int, default=20)
    ap.add_argument("--no-seed", action="store_true")
    a = ap.parse_args()
    if a.model and len(a.provider) != 1:
        sys.exit("--model can only be used with exactly one --provider")

    if not (ROOT / "data" / "processed" / "alpha158.parquet").exists():
        sys.exit("Phase 1 data not found. Run first:  python run_phase1.py")

    print("\n=== Step 1/5: tests ===", flush=True)
    run([str(ROOT / "tests" / "test_dsl.py")], cwd=ROOT)
    run([str(ROOT / "tests" / "test_llm.py")], cwd=ROOT)

    print("\n=== Step 2/5: generate factors ===", flush=True)
    if not a.no_seed:
        run(["generate_factors.py", "--provider", "replay"])
    if any((ROOT / "llm_runs").glob("*.json")):
        # re-read saved replies not yet in the library (e.g. ones recovered by a newer reader)
        run(["generate_factors.py", "--provider", "replay", "--replay-dir", str(ROOT / "llm_runs")])
    failed = []
    for prov in a.provider:
        cmd = ["generate_factors.py", "--provider", prov, "--batches", str(a.batches),
               "--per-batch", str(a.per_batch)]
        if a.model:
            cmd += ["--model", a.model]
        if run(cmd, check=False) != 0:
            failed.append(prov)
    if not a.provider:
        print("    (no --provider given: no new factors requested, re-scoring the library)")

    print("\n=== Step 3/5: score factors on 2014-2020 (about 2 s per factor) ===", flush=True)
    run(["score_factors.py"] + (["--no-seed"] if a.no_seed else []))

    print("\n=== Step 4/5: draw charts ===", flush=True)
    run(["plot_phase2.py"])

    print("\n=== Step 5/5: update README tables ===", flush=True)
    run(["report.py"])

    print("\nAll done. Results in results/, charts in figures/, README tables updated.")
    if failed:
        print(f"Note: these providers failed and were skipped: {', '.join(failed)} "
              "(run `python src/llm.py check` to see why)")


if __name__ == "__main__":
    main()
