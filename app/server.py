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


class LabRequest(BaseModel):
    idea: str = ""
    expression: str = ""


class LabSave(BaseModel):
    job: str
    index: int


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

    @app.post("/api/ask")
    def ask(q: Question):
        if not q.question.strip():
            raise HTTPException(400, "ask a question")
        ctx = chief.new_context(cfg, llm=get_llm(), folder=folder, ledger_paths=lp)
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

    # ------------------------------------------------------------ stocks (personal mode: names stocks)
    market_cache = {}

    def market():
        if "m" not in market_cache:
            from data import load_market
            market_cache["m"] = load_market(cfg.name)
        return market_cache["m"]

    def latest_rank(tag: str):
        files = sorted(lp["private"].glob(f"*_{tag}.csv")) if lp["private"].exists() else []
        return (pd.read_csv(files[-1]), files[-1]) if files else (None, None)

    def risk_info():
        import risk_lists
        try:
            _, surv = risk_lists.latest_reg_ind()
        except Exception:  # noqa: BLE001 - a malformed file just means no flags
            surv = {}
        return surv, risk_lists.sectors()

    @app.get("/api/stocks")
    def stocks():
        a158, _ = latest_rank("a158")
        if a158 is None:
            return {"as_of": None, "stocks": []}
        llm, _ = latest_rank("llm")
        llm_in = set(llm.loc[llm["in_book"], "code"]) if llm is not None else set()
        surv, sect = risk_info()
        rows = []
        for i, r in enumerate(a158.itertuples(), 1):
            code = str(r.code)
            rows.append({"rank": i, "code": code, "p": float(r.p), "in_book": bool(r.in_book),
                         "in_book_llm": code in llm_in, "sector": sect.get(code.upper()),
                         "flags": surv.get(code.upper(), [])})
        return clean({"as_of": str(a158["as_of"].iloc[0]), "count": len(rows), "stocks": rows})

    @app.get("/api/stocks/{code}")
    def stock(code: str):
        from generate_factors import load_library
        from team import explain
        a158, f = latest_rank("a158")
        if a158 is None:
            raise HTTPException(404, "no weekly ranking yet: run the team's weekly jobs first")
        codes = a158["code"].astype(str).tolist()
        if code not in codes:
            raise HTTPException(404, f"{code} is not in this week's top-200 universe")
        i = codes.index(code)
        row = a158.iloc[i]
        as_of = str(row["as_of"])
        out = {"code": code, "as_of": as_of, "rank": i + 1, "of": len(codes), "p": float(row["p"]),
               "in_book": bool(row["in_book"])}
        llm, _ = latest_rank("llm")
        if llm is not None and code in set(llm["code"].astype(str)):
            lr = llm.reset_index(drop=True)
            j = lr.index[lr["code"].astype(str) == code][0]
            out["llm"] = {"rank": int(j) + 1, "p": float(lr.loc[j, "p"]), "in_book": bool(lr.loc[j, "in_book"])}
        surv, sect = risk_info()
        out["sector"], out["flags"] = sect.get(code.upper()), list(surv.get(code.upper(), []))
        why_f = lp["private"] / f"{as_of}_a158_why.csv"
        out["reasons"] = []
        if why_f.exists():
            why = pd.read_csv(why_f)
            lib = {r["id"]: r for r in load_library()}
            for w in why[why["code"].astype(str) == code].itertuples():
                out["reasons"].append({"feature": w.feature, "contribution": float(w.contribution),
                                       "fact": explain.fact(w.feature, w.rank_pct, w.contribution, lib)})
        notes = notes_file()
        out["note"] = notes.get(f"{as_of}:{code}")
        # price history: one year, indexed to 100, against the Nifty 200
        m = market()
        if code in m.close.columns:
            c = m.close[code].dropna().iloc[-252:]
            idx = m.index_close.reindex(c.index).ffill()
            out["prices"] = {"dates": [d.strftime("%Y-%m-%d") for d in c.index],
                             "stock": (c / c.iloc[0] * 100).round(2).tolist(),
                             "index": (idx / idx.iloc[0] * 100).round(2).tolist()}
            last = m.close[code].dropna()
            out["returns"] = {k: float(last.iloc[-1] / last.iloc[-1 - n] - 1) if len(last) > n else None
                              for k, n in (("week", 5), ("month", 21), ("quarter", 63), ("year", 252))}
        # past weeks in the ledger
        hist = []
        for bf in sorted(lp["private"].glob("*_a158.csv")) if lp["private"].exists() else []:
            b = pd.read_csv(bf)
            cs = b["code"].astype(str).tolist()
            if code in cs:
                hist.append({"as_of": str(b["as_of"].iloc[0]), "rank": cs.index(code) + 1,
                             "p": float(b["p"].iloc[cs.index(code)]), "in_book": bool(b["in_book"].iloc[cs.index(code)])})
        out["history"] = hist[::-1]
        return clean(out)

    def notes_file() -> dict:
        nf = folder / "private" / "stock_notes.json"
        return json.loads(nf.read_text()) if nf.exists() else {}

    @app.post("/api/stocks/{code}/explain")
    def stock_explain(code: str):
        from team.agents import EXPLAIN_SYSTEM
        from llm import extract_json
        s = stock(code)
        if not s["reasons"]:
            raise HTTPException(400, "no reasons saved for this stock yet")
        facts = [r["fact"] for r in s["reasons"] if not r["feature"].startswith("mkt_")][:4]
        llm = get_llm()
        if llm is None:
            raise HTTPException(400, "no LLM key found; the facts above are the explanation")
        try:
            obj = extract_json(llm(EXPLAIN_SYSTEM.replace("ranked one stock highly", "ranked one stock where it did"),
                                   json.dumps({"stock": code, "rank": f"{s['rank']} of {s['of']}", "facts": facts,
                                               "risk_flags": s["flags"]})) or "", "summary")
        except Exception as e:  # noqa: BLE001
            raise HTTPException(502, f"the LLM didn't answer: {str(e)[:120]}")
        note = {"summary": str(obj.get("summary", "")).strip(), "watch": str(obj.get("watch", "")).strip()}
        if not note["summary"] or compliance.check(note["summary"] + " " + note["watch"]):
            raise HTTPException(422, "the LLM's wording failed the compliance check; the facts above still stand")
        notes = notes_file()
        notes[f"{s['as_of']}:{code}"] = note
        (folder / "private").mkdir(exist_ok=True)
        (folder / "private" / "stock_notes.json").write_text(json.dumps(notes, indent=1))
        EventLog(folder, "app").write("Explainer", "explain", f"Explained {code} in plain words", private=True)
        return note

    # ------------------------------------------------------------ research lab
    import research_lab
    lab_jobs = research_lab.Jobs(cfg, lambda: get_llm(),
                                 log=lambda agent, action, msg, status: EventLog(folder, "lab").write(agent, action, msg, status))

    @app.post("/api/lab")
    def lab_start(r: LabRequest):
        idea, expr = r.idea.strip(), r.expression.strip()
        if not idea and not expr:
            raise HTTPException(400, "describe an idea or write a formula")
        try:
            job = lab_jobs.start(idea=idea or None, expression=expr or None)
        except RuntimeError as e:
            raise HTTPException(409, str(e))
        return {"id": job["id"]}

    @app.get("/api/lab/{job_id}")
    def lab_job(job_id: str):
        j = lab_jobs.get(job_id)
        if not j:
            raise HTTPException(404, "no such lab run (the app may have restarted)")
        return clean({k: v for k, v in j.items() if k != "started"})

    @app.post("/api/lab-save")
    def lab_save(r: LabSave):
        j = lab_jobs.get(r.job)
        if not j or j["status"] != "done" or not (0 <= r.index < len(j["results"])):
            raise HTTPException(404, "nothing to save")
        rec = j["results"][r.index]
        if rec.get("status") != "valid":
            raise HTTPException(400, "only a valid, new formula can be saved")
        try:
            new = research_lab.save_to_library(rec)
        except ValueError as e:
            raise HTTPException(409, str(e))
        rec["saved_as"] = new["id"]
        Journal(folder).add("You", f"saved {new['id']} \"{new.get('name')}\" from the research lab ({rec.get('verdict')})")
        return {"id": new["id"]}

    @app.get("/api/grammar")
    def grammar():
        from dsl import grammar_doc
        return {"grammar": grammar_doc()}

    # ------------------------------------------------------------ strategy library
    @app.get("/api/strategies")
    def strategies():
        from generate_factors import load_library
        lib = load_library()
        p4, p2 = {}, {}
        f4, f2 = cfg.results / "phase4_factors.csv", cfg.results / "phase2_factor_scores.csv"
        if f4.exists():
            for r in pd.read_csv(f4).to_dict("records"):
                p4.setdefault(r["id"], r)
        if f2.exists():
            for r in pd.read_csv(f2).to_dict("records"):
                p2[r["id"]] = r
        rows = []
        for r in lib:
            w = p4.get(r["id"]) or {}
            lab = r.get("lab") or {}
            d = p2.get(r["id"]) or {}
            who = "you" if r.get("provider") == "you" else (
                "Claude (seed)" if r.get("provider") == "claude-in-chat" else f"{r.get('provider')}: {r.get('model')}")
            if r.get("phase") == "lab":
                who = f"lab · {r.get('provider')}"
            rows.append({
                "id": r["id"], "name": r.get("name"), "category": r.get("category"), "expression": r.get("expression")
                or r.get("raw_expression"), "hypothesis": r.get("hypothesis"), "researcher": who,
                "phase": r.get("phase", 2), "status": r.get("status"), "error": r.get("error"),
                "verdict": w.get("verdict") or lab.get("verdict") or (r.get("status") if r.get("status") != "valid" else None),
                "reason": w.get("reason") or lab.get("reason"),
                "net_excess_ann": w.get("net_excess_ann", lab.get("net_excess_ann")),
                "hit_rate": w.get("hit_rate", lab.get("hit_rate")),
                "weekly_ic_t": w.get("weekly_ic_t", lab.get("weekly_ic_t")),
                "daily_rank_ic": d.get("rank_ic"), "shortlist": bool(d.get("shortlist")) if d else False,
                "created_at": r.get("created_at"), "idea": r.get("idea"),
            })
        return clean({"count": len(rows), "strategies": rows})

    llm_cache = {}

    def get_llm():
        if "llm" not in llm_cache:
            try:
                llm_cache["llm"] = chief.make_llm()
            except Exception:  # noqa: BLE001
                llm_cache["llm"] = None
        return llm_cache["llm"]

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
