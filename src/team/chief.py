"""The Chief Strategist: runs the team in order, stops the line when an agent says STOP, and answers
questions from the activity log.

  every run      Data Collector -> Data Checker -> Regime Watcher
  weekly         Model Trainer -> Risk Officer -> Explainer -> Reporter -> Compliance Officer
  monthly        Research lead (asks for approval first; runs only after you approve)

The weekly jobs run when you pass --weekly, or automatically when the last ledger entry is 7+ days old.
"""
from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

from .agents import (ComplianceOfficer, DataChecker, DataCollector, Explainer, ModelTrainer, RegimeWatcher,
                     Reporter, ResearchLead, RiskOfficer)
from .core import OK, STOP, WARN, Agent, Approvals, Context, EventLog, Journal, Result, new_run_id

LLM_ORDER = ["gemini", "groq"]


def make_llm(order: list[str] = LLM_ORDER):
    """A callable (system, user) -> text that tries Gemini, then Groq. None if no key is set
    (the team then writes template text, which is always available)."""
    from llm import LLMClient, load_dotenv
    load_dotenv()
    clients = []
    for name in order:
        try:
            clients.append(LLMClient(name, timeout=90, max_retries=2))
        except SystemExit:
            continue
    if not clients:
        return None

    def call(system: str, user: str) -> str:
        last = None
        for c in list(clients):
            try:
                return c.complete(system, user, max_tokens=600).text
            except Exception as e:  # noqa: BLE001 - try the next provider
                last = e
                clients.remove(c)         # don't keep hitting a provider that failed this run
        raise RuntimeError(f"no LLM answered: {last}")

    call.providers = [c.provider for c in clients]
    return call


def team_folder(cfg) -> Path:
    return cfg.results / "team"


def weekly_due(ctx: Context, today: date) -> bool:
    import ledger
    commits = ledger.read_commits(ctx.ledger_paths or ledger.paths())
    if not commits:
        return True
    return (today - date.fromisoformat(commits[-1]["as_of"])).days >= 7


class ChiefStrategist(Agent):
    name, desk = "Chief Strategist", "Chief"

    def __init__(self, team: dict[str, Agent] | None = None):
        self.team = team or {
            "collect": DataCollector(), "check": DataChecker(), "regime": RegimeWatcher(),
            "model": ModelTrainer(), "risk": RiskOfficer(), "explain": Explainer(), "report": Reporter(),
            "comply": ComplianceOfficer(), "research": ResearchLead()}

    def plan(self, ctx: Context, today: date) -> list[str]:
        steps = ["collect", "check", "regime"]
        if ctx.weekly or weekly_due(ctx, today):
            ctx.weekly = True
            steps += ["model", "risk", "explain", "report", "comply"]
        steps.append("research")
        return steps

    def run(self, ctx: Context, today: date | None = None) -> Result:
        today = today or ctx.shared.get("today") or date.today()
        ctx.shared["today"] = today
        steps = self.plan(ctx, today)
        llm = getattr(ctx.llm, "providers", None) or (["an LLM"] if ctx.llm else None)
        self.say(ctx, "start", f"Run started ({'weekly' if ctx.weekly else 'daily'} jobs; words by "
                 f"{' then '.join(llm) if llm else 'templates (no LLM)'})", steps=steps)
        results = ctx.shared.setdefault("results", [])
        worst = OK
        for key in steps:
            agent = self.team[key]
            try:
                r = agent.run(ctx)
            except (Exception, SystemExit) as e:  # noqa: BLE001 - one broken agent must not hide what happened
                r = Result(STOP, f"crashed: {type(e).__name__}: {str(e)[:200]}")
                self.say(ctx, "crash", f"{agent.name} {r.summary}", STOP)
            results.append({"agent": agent.name, "status": r.status, "summary": r.summary})
            if r.status == WARN and worst == OK:
                worst = WARN
            if r.status == STOP:
                worst = STOP
                if key in ("collect", "check", "model"):   # bad data or a broken ledger: stop the line
                    ctx.journal.add(self.name, f"Stopped the run: {agent.name} said '{r.summary}'")
                    self.say(ctx, "halt", f"Stopped after {agent.name}: {r.summary}", STOP)
                    break
        done = len(results)
        self.say(ctx, "finish", f"Run finished: {done}/{len(steps)} steps, status {worst}", worst)
        return Result(worst, f"{done}/{len(steps)} steps", {"results": results})


def new_context(cfg, *, llm=None, mode="personal", weekly=False, skip_download=False, folder=None,
                ledger_paths=None) -> Context:
    folder = Path(folder or team_folder(cfg))
    folder.mkdir(parents=True, exist_ok=True)
    return Context(cfg=cfg, folder=folder, log=EventLog(folder, new_run_id()), approvals=Approvals(folder),
                   journal=Journal(folder), llm=llm, mode=mode, weekly=weekly, skip_download=skip_download,
                   ledger_paths=ledger_paths)


# ---------------------------------------------------------------- questions

ASK_SYSTEM = (
    "You are the Chief Strategist of a stock-research team. Answer the question using ONLY the activity "
    "log, journal and pending approvals given. If the answer is not there, say so. Plain English, under "
    "120 words. Never tell anyone to buy, sell or hold anything, never give price targets or predictions.")


def ask(ctx: Context, question: str, private: bool = False) -> str:
    """Answer a question about what the team did. Without an LLM, shows the most relevant log lines."""
    from . import compliance
    evs = ctx.log.read(last=60) + (ctx.log.read(private=True, last=40) if private else [])
    pending = [c for c in ctx.approvals.all() if c["status"] == "pending"]
    words = {w for w in question.lower().split() if len(w) > 3}
    hits = [e for e in evs if words & set(f"{e['agent']} {e['message']}".lower().split())] or evs[-10:]
    lines = [f"{e['time'][:16]} {e['agent']}: {e['message']}" for e in hits[-15:]]
    if ctx.llm is not None:
        user = json.dumps({"question": question, "log": [f"{e['time'][:16]} {e['agent']} [{e['status']}]: "
                                                         f"{e['message']}" for e in evs],
                           "journal": ctx.journal.recent(15), "pending_approvals": pending})
        try:
            ans = (ctx.llm(ASK_SYSTEM, user) or "").strip()
            if ans and not compliance.check(ans):
                return ans
        except Exception:  # noqa: BLE001 - fall back to the log lines
            pass
    head = "From the activity log:" if lines else "The activity log is empty: run the team first."
    tail = [f"Waiting for you: {c['question']} (card {c['id']})" for c in pending]
    return "\n".join([head, *lines, *tail])
