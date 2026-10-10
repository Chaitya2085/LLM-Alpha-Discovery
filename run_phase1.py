"""Run all of Phase 1 with one command (Windows, macOS, Linux):

    python run_phase1.py

Steps (each is skipped if its output already exists; use --force to redo):
  1. download the Qlib China A-share dataset (~570 MB)  -> data/raw/qlib_bin/
  2. build the CSI300 point-in-time dataset              -> data/processed/
  3. compute the 158 Alpha158 features                    -> data/processed/alpha158.parquet
  4. run baselines (walk-forward LightGBM + factors)      -> results/
  5. draw the headline chart                              -> figures/phase1_growth.png
  6. fill the Phase 1 table in README.md from the results
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
SRC = ROOT / "src"
DATA_URL = "https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz"


def step(n: int, title: str) -> None:
    print(f"\n=== Step {n}/6: {title} ===", flush=True)


def run(script: str) -> None:
    t = time.time()
    subprocess.run([sys.executable, script], cwd=SRC, check=True)
    print(f"    done in {time.time() - t:.0f}s", flush=True)


def download(force: bool) -> None:
    target = RAW / "qlib_bin"
    if (target / "calendars" / "day.txt").exists() and not force:
        print("    data already downloaded, skipping")
        return
    RAW.mkdir(parents=True, exist_ok=True)
    tgz = RAW / "qlib_bin.tar.gz"
    if not tgz.exists() or force:
        print(f"    downloading {DATA_URL}")

        def hook(blocks, bs, total):
            if total > 0 and blocks % 400 == 0:
                print(f"    {min(blocks * bs / total, 1):6.1%}", end="\r", flush=True)

        urllib.request.urlretrieve(DATA_URL, tgz, hook)
        print()
    print("    extracting (takes a minute) ...")
    if target.exists():
        shutil.rmtree(target)
    with tarfile.open(tgz) as tf:
        try:
            tf.extractall(RAW, filter="data")
        except TypeError:  # Python < 3.12
            tf.extractall(RAW)
    tgz.unlink()  # free ~570 MB; the extracted folder is what we use


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="redo every step")
    a = ap.parse_args()

    if sys.version_info < (3, 10):
        sys.exit("Python 3.10 or newer is required.")

    step(1, "download data")
    download(a.force)

    step(2, "build CSI300 dataset")
    if (PROC / "prices.parquet").exists() and not a.force:
        print("    already built, skipping")
    else:
        run("build_dataset.py")

    step(3, "compute Alpha158 features")
    if (PROC / "alpha158.parquet").exists() and not a.force:
        print("    already built, skipping")
    else:
        run("build_features.py")

    step(4, "run baselines (about 5-10 minutes)")
    run("run_baselines.py")

    step(5, "draw chart")
    run("plot_phase1.py")

    step(6, "update README tables")
    run("report.py")

    print(f"\nAll done. Results in {RES.name}/, chart in figures/, README tables updated.")


if __name__ == "__main__":
    main()
