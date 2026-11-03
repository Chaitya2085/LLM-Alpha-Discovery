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
sys.path.insert(0, str(HERE))
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


def _market_app():
    """An app on a small synthetic market with one recorded week and a factor library."""
    import time as _t
    import pandas as pd
    import generate_factors
    from synth_market import make_market
    root = TMP / "mk"
    cfg = make_market(root, name="app8m", n=40, seed=2, start="2013-06-01", end="2021-03-31")
    generate_factors.LIB = root / "library.jsonl"
    generate_factors.append_library(generate_factors.ingest(json.dumps({"factors": [
        {"name": "busy", "category": "volume", "expression": "Mean(volume, 20)", "hypothesis": "h"}]}),
        {"provider": "fake", "model": "m", "prompt_version": "t", "run_file": "t", "created_at": "t"}, []))
    lp = ledger.paths(root / "ledger")
    lp["private"].mkdir(parents=True, exist_ok=True)
    codes = [f"S{i:03d}" for i in range(40)]
    pd.DataFrame({"as_of": "2021-03-31", "code": codes, "p": [0.55 - i * 0.002 for i in range(40)],
                  "rank_pct": [1 - i / 40 for i in range(40)], "in_book": [i < 10 for i in range(40)]}).to_csv(
        lp["private"] / "2021-03-31_a158.csv", index=False)
    pd.DataFrame({"code": ["S005", "S005"], "feature": ["a158_ROC5", "a158_STD20"], "rank_pct": [0.9, 0.2],
                  "contribution": [0.1, -0.05]}).to_csv(lp["private"] / "2021-03-31_a158_why.csv", index=False)
    import team.chief as chief
    chief.make_llm = lambda *a, **k: (lambda system, user: json.dumps({"factors": [
        {"name": "fell_quietly", "category": "reversal", "expression": "-Pct(close, 20) * Mean(volume, 60) / Mean(volume, 20)",
         "hypothesis": "quiet falls recover"}]}))
    c = TestClient(create_app(cfg, ledger_paths=lp, team_runner=lambda a: FakeProc()), base_url="http://127.0.0.1")
    return c, _t


def test_stock_list_and_page():
    c, _ = _market_app()
    s = c.get("/api/stocks").json()
    assert s["count"] == 40 and s["stocks"][0]["code"] == "S000" and s["stocks"][0]["in_book"]
    p = c.get("/api/stocks/S005").json()
    assert p["rank"] == 6 and p["in_book"] and len(p["prices"]["stock"]) > 200
    assert [r["contribution"] > 0 for r in p["reasons"]] == [True, False] and "ranks above 90%" in p["reasons"][0]["fact"]
    assert c.get("/api/stocks/NOPE").status_code == 404


def test_lab_formula_and_idea_then_save():
    c, t = _market_app()
    def wait(jid):
        for _ in range(240):
            j = c.get(f"/api/lab/{jid}").json()
            if j["status"] in ("done", "failed"):
                return j
            t.sleep(0.5)
        raise AssertionError("lab run did not finish")
    j = wait(c.post("/api/lab", json={"expression": "Mean(volume, 20)"}).json()["id"])
    assert j["status"] == "done", j
    assert j["results"][0]["status"] == "duplicate" and "already in the library" in j["results"][0]["reason"]
    j = wait(c.post("/api/lab", json={"idea": "stocks that fell quietly"}).json()["id"])
    assert j["status"] == "done", j
    r = j["results"][0]
    assert r["status"] == "valid" and r["verdict"] and r["curve"]["dates"][-1] <= "2020-12-31"   # never the exam years
    saved = c.post("/api/lab-save", json={"job": j["id"], "index": 0}).json()["id"]
    assert c.post("/api/lab-save", json={"job": j["id"], "index": 0}).status_code == 409
    lib = {x["id"]: x for x in c.get("/api/strategies").json()["strategies"]}
    assert lib[saved]["verdict"] == r["verdict"] and lib[saved]["idea"] == "stocks that fell quietly"


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\nall {len(tests)} tests passed")
