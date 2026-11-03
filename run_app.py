"""Phase 8a: open the app on your computer.

    python run_app.py            # builds the page the first time (needs Node.js), then opens it
    python run_app.py --build    # rebuild the page after an update
    python run_app.py --no-open  # don't open the browser

The app runs only on this computer (http://127.0.0.1:8000) because it shows stock names. Stop it with
Ctrl+C. While it runs you can still use run_team.py in another Terminal window.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"


def build() -> None:
    npm = shutil.which("npm")
    if not npm:
        sys.exit("Node.js is needed to build the page once: install it from https://nodejs.org (LTS), then run again.")
    print("Building the page (first time takes about a minute)...", flush=True)
    for args in (["install", "--no-audit", "--no-fund"], ["run", "build"]):
        if subprocess.run([npm, *args], cwd=WEB).returncode != 0:
            sys.exit(f"npm {args[0]} failed; send the error above.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    try:
        import fastapi  # noqa: F401
        import uvicorn
    except ImportError:
        sys.exit("Install the backend packages first:  pip install fastapi uvicorn")
    if a.build or not (WEB / "dist" / "index.html").exists():
        build()
    sys.path.insert(0, str(ROOT))
    from app.server import create_app
    url = f"http://127.0.0.1:{a.port}"
    print(f"\nLLM Alpha Discovery is running at {url}  (Ctrl+C to stop)\n", flush=True)
    if not a.no_open:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(create_app(), host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
