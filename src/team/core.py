"""The plumbing every agent shares: the activity log, approval cards, the research journal, and the
context an agent works in.

Where things are written (results/<market>/team/):
  events.jsonl         the public activity log: what each agent did, never a current stock name
  private/events.jsonl events that name current stocks (risk flags, explanations): never committed
  approvals.json       decisions waiting for a person (approve / reject from the command line)
  journal.md           the research journal: decisions, approvals, lessons, in plain words
  reports/             weekly evidence reports (personal_*.md stays private; learn_*.md can be shared)
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

OK, WARN, STOP = "ok", "warning", "stop"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class EventLog:
    def __init__(self, folder: Path, run_id: str):
        self.folder = folder
        self.run_id = run_id
        (folder / "private").mkdir(parents=True, exist_ok=True)

    def write(self, agent: str, action: str, message: str, status: str = OK, private: bool = False,
              **data) -> dict:
        ev = {"time": now(), "run": self.run_id, "agent": agent, "action": action, "status": status,
              "message": message, **({"data": data} if data else {})}
        f = self.folder / ("private/events.jsonl" if private else "events.jsonl")
        with f.open("a") as fh:
            fh.write(json.dumps(ev, default=str) + "\n")
        return ev

    def read(self, private: bool = False, run: str | None = None, last: int | None = None) -> list[dict]:
        f = self.folder / ("private/events.jsonl" if private else "events.jsonl")
        if not f.exists():
            return []
        evs = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        if run:
            evs = [e for e in evs if e["run"] == run]
        return evs[-last:] if last else evs


class Approvals:
    """Decisions the team may not take alone. A card is pending until a person answers it."""

    def __init__(self, folder: Path):
        self.file = folder / "approvals.json"

    def all(self) -> list[dict]:
        return json.loads(self.file.read_text()) if self.file.exists() else []

    def _save(self, cards: list[dict]) -> None:
        self.file.write_text(json.dumps(cards, indent=2))

    def ask(self, agent: str, kind: str, question: str, detail: str = "") -> dict:
        cards = self.all()
        for c in cards:                       # never ask the same open question twice
            if c["kind"] == kind and c["status"] == "pending":
                return c
        card = {"id": uuid.uuid4().hex[:6], "agent": agent, "kind": kind, "question": question,
                "detail": detail, "status": "pending", "asked_at": now(), "answered_at": None}
        cards.append(card)
        self._save(cards)
        return card

    def answer(self, card_id: str, approve: bool, note: str = "") -> dict:
        cards = self.all()
        for c in cards:
            if c["id"] == card_id:
                if c["status"] != "pending":
                    raise SystemExit(f"card {card_id} was already {c['status']}")
                c.update(status="approved" if approve else "rejected", answered_at=now(), note=note)
                self._save(cards)
                return c
        raise SystemExit(f"no approval card with id {card_id}")

    def approved(self, kind: str) -> dict | None:
        """The oldest approved card of this kind that hasn't been acted on yet."""
        for c in self.all():
            if c["kind"] == kind and c["status"] == "approved" and not c.get("done_at"):
                return c
        return None

    def mark_done(self, card_id: str, outcome: str) -> None:
        cards = self.all()
        for c in cards:
            if c["id"] == card_id:
                c.update(done_at=now(), outcome=outcome)
        self._save(cards)


class Journal:
    def __init__(self, folder: Path):
        self.file = folder / "journal.md"

    def add(self, agent: str, text: str) -> None:
        new = not self.file.exists()
        with self.file.open("a") as fh:
            if new:
                fh.write("# Research journal\n\nDecisions, approvals and lessons, newest last.\n\n")
            fh.write(f"- {now()[:16].replace('T', ' ')} UTC · **{agent}**: {text}\n")

    def recent(self, n: int = 20) -> list[str]:
        if not self.file.exists():
            return []
        return [x for x in self.file.read_text().splitlines() if x.startswith("- ")][-n:]


@dataclass
class Result:
    status: str = OK
    summary: str = ""
    data: dict = field(default_factory=dict)


@dataclass
class Context:
    """What an agent can see and use during one run of the team."""
    cfg: object                       # markets.MarketConfig
    folder: Path                      # results/<market>/team
    log: EventLog
    approvals: Approvals
    journal: Journal
    llm: object = None                # a callable (system, user) -> str, or None for template text only
    mode: str = "personal"            # "personal" (local, names stocks) or "learn" (30-day-old data, no names)
    weekly: bool = False              # run the weekly jobs this time
    skip_download: bool = False
    ledger_paths: dict | None = None
    shared: dict = field(default_factory=dict)   # results agents pass to each other in one run
    _market: object = None

    def market(self):
        if self._market is None:
            import phase5_model as p5
            self._market = p5.load_cut(self.cfg, "2262-01-01")
        return self._market


class Agent:
    name = "agent"
    desk = ""

    def run(self, ctx: Context) -> Result:  # pragma: no cover - each agent overrides
        raise NotImplementedError

    def say(self, ctx: Context, action: str, message: str, status: str = OK, private: bool = False, **data):
        return ctx.log.write(self.name, action, message, status, private, **data)


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:4]
