"""Tests for Phase 8a (the local app's backend). No internet, no Node.js needed.

Run:  python tests/test_phase8.py   (or pytest)
"""
import dataclasses
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "src"))
from fastapi.testclient import TestClient  # noqa: E402

import ledger  # noqa: E402
import markets  # noqa: E402
from app.server import create_app  # noqa: E402
from team.core import Approvals, EventLog  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="phase8_test_"))


class FakeProc:
    def __init__(self):
        self.done = False
        self.returncode = 0

    def poll(self):
        return 0 if self.done else None


def _app():
    cfg = dataclasses.replace(markets.MARKETS["india"], name="app8", results=TMP / "results")
    team = cfg.results / "team"
    log = EventLog(team, "r1")
    log.write("Chief Strategist", "start", "Run started")
    log.write("Data Checker", "check", "Data through 2026-11-02: 200 stocks", as_of="2026-11-02")
    log.write("Regime Watcher", "label", "Market mood: calm, rising", regime="calm, rising")
    log.write("Risk Officer", "flag", "ABC: under GSM", "warning", private=True)
    log.write("Chief Strategist", "finish", "Run finished: 4/4 steps, status ok")
    card = Approvals(team).ask("Research lead", "research-round", "Run a research round?")
    procs = []

    def runner(args):
        procs.append((args, FakeProc()))
        return procs[-1][1]

    client = TestClient(create_app(cfg, ledger_paths=ledger.paths(TMP / "ledger"), team_runner=runner),
                        base_url="http://127.0.0.1")
    return client, card, procs, team


def test_overview_and_privacy():
    c, card, _, _ = _app()
    o = c.get("/api/overview").json()
    assert o["as_of"] == "2026-11-02" and o["regime"] == "calm, rising" and o["pending"] == 1
    assert {a["name"] for a in o["agents"]} >= {"Chief Strategist", "Explainer", "Research lead"}
    assert next(a for a in o["agents"] if a["name"] == "Explainer")["status"] == "idle"
    public = c.get("/api/events").json()
    assert not any("ABC" in e["message"] for e in public)                 # stock names stay private
    private = c.get("/api/events?private=true").json()
    assert any(e.get("private") and "ABC" in e["message"] for e in private)


def test_only_this_computer():
    c, *_ = _app()
    assert c.get("/api/overview", headers={"host": "evil.example.com"}).status_code == 403


def test_approve_from_the_app():
    c, card, _, team = _app()
    r = c.post(f"/api/approvals/{card['id']}", json={"approve": True})
    assert r.status_code == 200 and r.json()["status"] == "approved"
    assert c.post(f"/api/approvals/{card['id']}", json={"approve": True}).status_code == 400   # only once
    assert "approved card" in (team / "journal.md").read_text()


def test_run_button_starts_one_team_at_a_time():
    c, _, procs, _ = _app()
    assert c.post("/api/run", json={}).json()["args"] == ["--skip-download"]
    assert c.get("/api/overview").json()["running"] is True
    assert c.post("/api/run", json={"download": True}).status_code == 409
    procs[-1][1].done = True
    assert c.post("/api/run", json={"download": True, "weekly": True}).json()["args"] == ["--weekly"]


def test_ask_track_and_week_without_results():
    c, *_ = _app()
    import team.chief as chief
    chief.make_llm = lambda *a, **k: None                                   # no API calls in tests
    ans = c.post("/api/ask", json={"question": "what did the data checker find?"}).json()["answer"]
    assert "Data Checker" in ans
    t = c.get("/api/track").json()
    assert t["exam"] is None and t["ledger"] == []
    assert c.get("/api/week").json()["stocks"] == []


def test_week_reads_explanations():
    c, _, _, team = _app()
    (team / "private").mkdir(exist_ok=True)
    (team / "private" / "explanations_2026-11-02.json").write_text(json.dumps(
        {"as_of": "2026-11-02", "variant": "Alpha158 only",
         "stocks": {"XYZ": {"confidence": 0.55, "facts": ["f"], "summary": "s", "watch": "w", "risk": []}}}))
    w = c.get("/api/week").json()
    assert w["as_of"] == "2026-11-02" and w["stocks"][0]["code"] == "XYZ"


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\nall {len(tests)} tests passed")
