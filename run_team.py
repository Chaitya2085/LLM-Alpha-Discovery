"""Phase 7: run the AI research team.

    python run_team.py                    # download, check the data, and (if a week has passed) run the weekly jobs
    python run_team.py --skip-download    # use the data already on this computer
    python run_team.py --weekly           # force the weekly jobs (ledger, risk, explanations, reports)
    python run_team.py --no-llm           # template words only, no API calls

    python run_team.py log                # what the team did (add --private to include stock names)
    python run_team.py approvals          # decisions waiting for you
    python run_team.py approve <id>       # say yes to a card    (reject <id> to say no)
    python run_team.py ask "why did the run stop?"

Everything is written to results/india/team/. The private/ folder there (stock names, personal reports)
is never committed. learn_*.md reports in reports/ use 30-day-old data and can be shared.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import markets  # noqa: E402
from team import chief  # noqa: E402
from team.core import Approvals, EventLog, Journal  # noqa: E402

ICON = {"ok": " ", "warning": "!", "stop": "X"}


def show(ev: dict) -> str:
    return f"  {ICON.get(ev['status'], '?')} {ev['time'][5:16].replace('T', ' ')}  {ev['agent']:<19} {ev['message']}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", default="run",
                    choices=["run", "log", "approvals", "approve", "reject", "ask"])
    ap.add_argument("arg", nargs="?", default=None, help="card id (approve/reject) or question (ask)")
    ap.add_argument("--market", default=os.environ.get("LLM_ALPHA_MARKET", "india"))
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--weekly", action="store_true")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--private", action="store_true", help="log/ask: include events that name stocks")
    ap.add_argument("--last", type=int, default=40)
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    cfg = markets.get(a.market)
    folder = chief.team_folder(cfg)
    folder.mkdir(parents=True, exist_ok=True)

    if a.command == "log":
        log = EventLog(folder, "-")
        evs = log.read(last=a.last)
        if a.private:
            evs = sorted(evs + log.read(private=True, last=a.last), key=lambda e: e["time"])[-a.last:]
        print("\n".join(show(e) for e in evs) or "  (empty: run the team first)")
        return
    if a.command == "approvals":
        cards = Approvals(folder).all()
        for c in cards[-20:]:
            print(f"  [{c['id']}] {c['status']:<9} {c['agent']}: {c['question']}"
                  + (f"  ({c['detail']})" if c.get("detail") else "")
                  + (f"  -> {c['outcome']}" if c.get("outcome") else ""))
        if not cards:
            print("  no approval cards yet")
        elif any(c["status"] == "pending" for c in cards):
            print("\n  Answer with:  python run_team.py approve <id>   or   python run_team.py reject <id>")
        return
    if a.command in ("approve", "reject"):
        if not a.arg:
            sys.exit("give the card id, e.g.  python run_team.py approve 3f9a1c")
        c = Approvals(folder).answer(a.arg, a.command == "approve", a.note)
        Journal(folder).add("You", f"{c['status']} card {c['id']}: {c['question']}")
        print(f"  card {c['id']} {c['status']}. It runs the next time the team runs.")
        return

    llm = None if a.no_llm else chief.make_llm()
    if a.command == "ask":
        if not a.arg:
            sys.exit('ask a question, e.g.  python run_team.py ask "what happened this week?"')
        print(chief.ask(chief.new_context(cfg, llm=llm), a.arg, private=a.private))
        return

    ctx = chief.new_context(cfg, llm=llm, weekly=a.weekly, skip_download=a.skip_download)
    print(f"\n=== AI research team: run {ctx.log.run_id} ===", flush=True)
    r = chief.ChiefStrategist().run(ctx)
    print()
    for e in ctx.log.read(run=ctx.log.run_id):
        print(show(e))
    pending = [c for c in ctx.approvals.all() if c["status"] == "pending"]
    for c in pending:
        print(f"\n  Waiting for you [{c['id']}]: {c['question']}\n    python run_team.py approve {c['id']}")
    for f in ctx.shared.get("saved_reports", []):
        print(f"\n  Report: {Path(f).relative_to(ROOT)}")
    sys.exit(3 if r.status == "stop" else 0)


if __name__ == "__main__":
    main()
