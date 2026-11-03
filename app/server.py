"""Phase 8a: the local web app's backend. It only reads what the team, the exam and the ledger wrote,
plus three actions: answer an approval card, ask the Chief Strategist, and start a team run.

It listens on 127.0.0.1 only (your computer), because Personal mode shows stock names. Requests that
don't come from this computer's own address are refused.

    python run_app.py          # builds the web page the first time, then opens http://127.0.0.1:8000
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import ledger  # noqa: E402
import markets  # noqa: E402
from team import chief, compliance  # noqa: E402
from team.core import Approvals, EventLog, Journal  # noqa: E402

DIST = ROOT / "web" / "dist"
ALLOWED_HOSTS = {"127.0.0.1", "localhost"}

ROSTER = [
    ("Chief Strategist", "Head office", "Runs the team in order and stops the line when something is wrong"),
    ("Data Collector", "Market data", "Downloads the latest NSE files after the close"),
    ("Data Checker", "Market data", "Catches stale, missing or odd data before anyone uses it"),
    ("Regime Watcher", "Model room", "Labels the market mood and notes when it changes"),
    ("Model Trainer", "Model room", "Records the frozen model's weekly predictions in the ledger"),
    ("Risk Officer", "Risk & compliance", "Flags NSE surveillance, crashes, thin trading and crowding"),
    ("Compliance Officer", "Risk & compliance", "Blocks advice wording and checks every report"),
    ("Explainer", "Communications", "Turns the model's reasons into plain English"),
    ("Reporter", "Communications", "Drafts the personal and shareable reports"),
    ("Research lead", "Research lab", "Asks before running a monthly research round"),
]


class Answer(BaseModel):
    approve: bool
    note: str = ""


class Question(BaseModel):
    question: str
    private: bool = False


class RunRequest(BaseModel):
    download: bool = False
    weekly: bool = False


def clean(x):
    """JSON-safe: NaN/inf -> None."""
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [clean(v) for v in x]
    return x


def create_app(cfg=None, ledger_paths=None, team_runner=None) -> FastAPI:
    cfg = cfg or markets.get(os.environ.get("LLM_ALPHA_MARKET", "india"))
    folder = chief.team_folder(cfg)
    folder.mkdir(parents=True, exist_ok=True)
    lp = ledger_paths or ledger.paths()
    log = EventLog(folder, "-")
    approvals = Approvals(folder)
    run_state = {"proc": None, "started": None, "args": None}
    runner = team_runner or (lambda args: subprocess.Popen([sys.executable, str(ROOT / "run_team.py"), *args],
                                                           cwd=ROOT, stdout=subprocess.DEVNULL,
                                                           stderr=subprocess.DEVNULL))
    app = FastAPI(title="LLM Alpha Discovery", docs_url=None, redoc_url=None)

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        host = (request.headers.get("host") or "").rsplit(":", 1)[0].strip("[]")
        client = request.client.host if request.client else ""
        if host not in ALLOWED_HOSTS or client not in {"127.0.0.1", "::1", "testclient"}:
            return JSONResponse({"detail": "this app only answers requests from this computer"}, 403)
        return await call_next(request)

    # ------------------------------------------------------------ reading

    def running() -> bool:
        p = run_state["proc"]
        return p is not None and p.poll() is None

    @app.get("/api/overview")
    def overview():
        evs = log.read()
        last = {}
        for e in evs:
            last[e["agent"]] = e
        runs = [e for e in evs if e["agent"] == "Chief Strategist" and e["action"] in ("start", "finish")]
        last_start = next((e for e in reversed(runs) if e["action"] == "start"), None)
        last_finish = next((e for e in reversed(runs) if e["action"] == "finish"), None)
        as_of = next((e["data"]["as_of"] for e in reversed(evs) if e["agent"] == "Data Checker"
                      and e.get("data", {}).get("as_of")), None)
        regime = next((e["data"]["regime"] for e in reversed(evs) if e["agent"] == "Regime Watcher"
                       and e.get("data", {}).get("regime")), None)
        agents = []
        for name, desk, job in ROSTER:
            e = last.get(name)
            agents.append({"name": name, "desk": desk, "job": job,
                           "status": e["status"] if e else "idle", "message": e["message"] if e else None,
                           "time": e["time"] if e else None})
        commits = ledger.read_commits(lp)
        return clean({
            "market": cfg.name, "as_of": as_of, "regime": regime, "agents": agents,
            "last_run": {"started": last_start and last_start["time"], "finished": last_finish and last_finish["time"],
                         "status": last_finish and last_finish["status"], "message": last_finish and last_finish["message"]},
            "running": running(),
            "pending": sum(1 for c in approvals.all() if c["status"] == "pending"),
            "ledger_weeks": len({c["as_of"] for c in commits}),
            "disclaimer": compliance.DISCLAIMER,
        })

    @app.get("/api/events")
    def events(last: int = 60, private: bool = False):
        evs = log.read(last=last)
        if private:
            mine = [{**e, "private": True} for e in log.read(private=True, last=last)]
            evs = sorted(evs + mine, key=lambda e: e["time"])[-last:]
        return list(reversed(evs))

    @app.get("/api/approvals")
    def get_approvals():
        return list(reversed(approvals.all()))

    @app.get("/api/journal")
    def journal():
        return Journal(folder).recent(30)

    @app.get("/api/track")
    def track():
        res = cfg.results
        out = {"exam": None, "growth": None, "ledger": [], "scores": []}
        s6 = res / "phase6_summary.json"
        if s6.exists():
            s = json.loads(s6.read_text().replace("NaN", "null"))
            out["exam"] = {"years": s.get("years"), "variants": {
                v: {"book": d["book"], "verdict": d["verdict"], "by_year": d.get("by_year"),
                    "rank_ic": d.get("weekly_rank_ic"), "calibration_gap": d.get("calibration", {}).get("ece")}
                for v, d in s["variants"].items()}}
        w6 = res / "phase6_weekly.csv"
        if w6.exists():
            w = pd.read_csv(w6, parse_dates=["date"])
            growth = {}
            for v, g in w.groupby("variant"):
                g = g.sort_values("date")
                growth[v] = {"dates": g["date"].dt.strftime("%Y-%m-%d").tolist(),
                             "model": (1 + g["net"]).cumprod().round(4).tolist(),
                             "index": (1 + g["bench"]).cumprod().round(4).tolist(),
                             "equal": (1 + g["universe"]).cumprod().round(4).tolist()
                             if "universe" in g else None}
            out["growth"] = growth
        now = datetime.now(timezone.utc)
        for c in ledger.read_commits(lp):
            created = datetime.fromisoformat(c["created_at"])
            opens = created + timedelta(days=ledger.REVEAL_AFTER_DAYS)
            out["ledger"].append({"as_of": c["as_of"], "variant": c["variant"], "fingerprint": c["sha256"][:12],
                                  "created_at": c["created_at"], "reveal_on": opens.date().isoformat(),
                                  "revealed": (lp["revealed"] / c["file"]).exists()})
        if lp["scores"].exists():
            out["scores"] = pd.read_csv(lp["scores"]).to_dict("records")
        return clean(out)

    @app.get("/api/week")
    def week():
        files = sorted((folder / "private").glob("explanations_*.json")) if (folder / "private").exists() else []
        if not files:
            return {"as_of": None, "variant": None, "stocks": []}
        ex = json.loads(files[-1].read_text())
        stocks = [{"code": k, **v} for k, v in ex["stocks"].items()]
        return clean({"as_of": ex["as_of"], "variant": ex["variant"], "stocks": stocks})

    # ------------------------------------------------------------ actions

    @app.post("/api/approvals/{card_id}")
    def answer(card_id: str, a: Answer):
        try:
            c = approvals.answer(card_id, a.approve, a.note)
        except SystemExit as e:
            raise HTTPException(400, str(e))
        Journal(folder).add("You", f"{c['status']} card {c['id']} in the app: {c['question']}")
        return c

    llm_holder = {"llm": None, "tried": False}

    @app.post("/api/ask")
    def ask(q: Question):
        if not q.question.strip():
            raise HTTPException(400, "ask a question")
        if not llm_holder["tried"]:
            llm_holder["tried"] = True
            try:
                llm_holder["llm"] = chief.make_llm()
            except Exception:  # noqa: BLE001 - no LLM: the answer comes from the log
                llm_holder["llm"] = None
        ctx = chief.new_context(cfg, llm=llm_holder["llm"], folder=folder, ledger_paths=lp)
        return {"answer": chief.ask(ctx, q.question.strip()[:500], private=q.private)}

    lock = threading.Lock()

    @app.post("/api/run")
    def run(r: RunRequest):
        with lock:
            if running():
                raise HTTPException(409, "the team is already running")
            args = ([] if r.download else ["--skip-download"]) + (["--weekly"] if r.weekly else [])
            run_state.update(proc=runner(args), started=datetime.now(timezone.utc).isoformat(), args=args)
        return {"started": run_state["started"], "args": args}

    @app.get("/api/run")
    def run_status():
        p = run_state["proc"]
        return {"running": running(), "started": run_state["started"],
                "exit_code": None if p is None or p.poll() is None else p.returncode}

    # ------------------------------------------------------------ the page
    if DIST.exists():
        app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

        @app.get("/{path:path}")
        def page(path: str):
            f = DIST / path
            if path and f.is_file() and DIST in f.resolve().parents:
                return FileResponse(f)
            return FileResponse(DIST / "index.html")

    return app
