"""Tests for Phase 7 (the AI research team), on a synthetic market. No internet or API keys needed.
The LLM is a fake: one that writes good notes and one that tries to give buy calls.

Run:  python tests/test_phase7.py   (or pytest)
"""
import json
import sys
import tempfile
from datetime import date
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))
import generate_factors  # noqa: E402
import ledger  # noqa: E402
import risk_lists  # noqa: E402
from synth_market import make_market  # noqa: E402
from team import agents, chief, compliance, explain  # noqa: E402
from team.core import OK, STOP, WARN  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="phase7_test_"))
agents.DataChecker.MIN_STOCKS = 40               # the test market has 60 stocks, not 200


def good_llm(system, user):
    facts = json.loads(user).get("facts", ["x"]) if user.startswith("{") else ["x"]
    return json.dumps({"summary": "The model liked this stock's recent trading pattern. " + facts[0][:80],
                       "watch": "The view changes if the pattern reverses."})


def bad_llm(system, user):
    return '{"summary": "Strong buy, target price 500. It will double.", "watch": "none"}'


def broken_llm(system, user):
    raise RuntimeError("quota exceeded")


def _setup(name, end="2023-06-30"):
    root = TMP / name
    cfg = make_market(root, name=name, n=60, seed=3, start="2016-01-01", end=end)
    generate_factors.LIB = root / "library.jsonl"
    recs = generate_factors.ingest(json.dumps({"factors": [
        {"name": "liquidity", "category": "volume", "expression": "Mean(volume, 20)",
         "hypothesis": "busy stocks drift up"}]}),
        {"provider": "fake", "model": "m", "prompt_version": "t", "run_file": "t", "created_at": "t"}, [])
    generate_factors.append_library(recs)
    frozen = {"variants": ["LLM factors only"], "designs": {"LLM factors only": {"stocks": 10, "keep_zone": 30}},
              "llm_factors": recs, "fingerprint": "test"}
    (root / "config").mkdir(parents=True, exist_ok=True)
    ledger.FROZEN = root / "config" / "phase6_frozen.json"
    ledger.FROZEN.write_text(json.dumps(frozen, default=str))
    return root, cfg


def _ctx(root, cfg, llm=None, weekly=True):
    ctx = chief.new_context(cfg, llm=llm, weekly=weekly, skip_download=True, folder=root / "team",
                            ledger_paths=ledger.paths(root / "ledger"))
    ctx.shared["today"] = date(2023, 7, 3)
    return ctx


def _research(root):
    calls = []
    return agents.ResearchLead(runner=lambda prov: calls.append(prov) or 0, runs_dir=root / "runs"), calls


def test_compliance_rules():
    assert compliance.check("Strong buy with a target price of 500")
    assert compliance.check("This stock will double")
    assert compliance.check("It is guaranteed")
    assert compliance.check("will give 20% returns")
    assert not compliance.check("The model ranked it in the top 5% on momentum.")
    t = date(2023, 7, 1)
    assert any("data-age" in x for x in compliance.check("ok", "learn", as_of=date(2023, 6, 20), today=t))
    assert any("current-pick" in x for x in compliance.check("S001 did well", "learn", date(2023, 5, 1), t, {"S001"}))
    assert any("disclaimer" in x for x in compliance.check("ok", require_disclaimer=True))
    assert not compliance.check("ok\n" + compliance.DISCLAIMER, require_disclaimer=True)


def test_explainer_facts_are_plain_and_true():
    assert "fell more than most" in explain.fact("a158_ROC5", 0.95, 0.1)
    assert "ranks above 95%" in explain.fact("a158_ROC5", 0.95, 0.1) and "raised" in explain.fact("a158_ROC5", 0.95, 0.1)
    assert "ranks below 90%" in explain.fact("a158_STD20", 0.10, -0.1)
    assert "Market backdrop" in explain.fact("mkt_vol20", 0.2, 0.05)
    f = explain.fact("llm_abc", 0.9, 0.2, {"abc": {"name": "busy_stocks", "hypothesis": "volume leads."}})
    assert "busy stocks" in f and "volume leads" in f
    assert not compliance.check(explain.template_note([f])["summary"])


def test_reg_ind_parser():
    f = TMP / "REG_IND300623.csv"
    row = lambda sym, gsm="100", asm="100", esm="100": ",".join(  # noqa: E731
        ["EQ", sym, "", "", "", gsm, asm, "", "100", "100", "", "", "", "", "100", "", "", "", esm])
    f.write_text("Series,Symbol,a,b,c,GSM,LTASM,d,IRP,STASM,e,f,g,h,Pledge,i,j,k,ESM\n"
                 + "\n".join([row("CLEAN"), row("RISKY", gsm="2"), row("ODD", asm="1", esm="2")]))
    flags = risk_lists.parse_reg_ind(f)
    assert flags["CLEAN"] == [] and flags["RISKY"] == ["GSM (stage 2)"]
    assert flags["ODD"] == ["long-term ASM (stage 1)", "ESM (stage 2)"]


def test_data_checker_stops_on_bad_data():
    root, cfg = _setup("bad7")
    ctx = _ctx(root, cfg)
    assert agents.DataChecker().run(ctx).status == OK
    assert agents.DataChecker().run(ctx, today=date(2023, 7, 10)).status == WARN      # 10 days old
    assert agents.DataChecker().run(ctx, today=date(2023, 8, 30)).status == STOP      # stale
    m = ctx.market()
    m.close.iloc[-1, :20] = float("nan")                                            # a third of prices missing
    assert agents.DataChecker().run(ctx).status == STOP
    r = chief.ChiefStrategist().run(ctx)                                            # the line stops
    assert r.status == STOP and [x["agent"] for x in r.data["results"]][-1] == "Data Checker"
    assert any(e["action"] == "halt" for e in ctx.log.read())


def test_weekly_run_good_llm_end_to_end():
    root, cfg = _setup("week7")
    lead, calls = _research(root)
    boss = chief.ChiefStrategist()
    boss.team["research"] = lead
    ctx = _ctx(root, cfg, llm=good_llm)
    r = boss.run(ctx)
    assert r.status in (OK, WARN), r.data
    names = [x["agent"] for x in r.data["results"]]
    assert names[:3] == ["Data Collector", "Data Checker", "Regime Watcher"] and "Compliance Officer" in names
    # ledger written, reasons saved, explanations by the LLM
    p = ledger.paths(root / "ledger")
    assert len(ledger.read_commits(p)) == 1 and list(p["private"].glob("*_why.csv"))
    ex = json.loads(next((root / "team" / "private").glob("explanations_*.json")).read_text())
    assert len(ex["stocks"]) == 10 and all(v["summary"] for v in ex["stocks"].values())
    # a book recorded before reasons were saved: the Explainer rebuilds them, identical to the originals
    why_f = next(p["private"].glob("*_why.csv"))
    original = pd.read_csv(why_f)
    why_f.unlink()
    assert agents.Explainer().run(_ctx(root, cfg)).status == OK
    assert pd.read_csv(why_f).equals(original)
    assert any("10 written by the LLM" in e["message"] for e in ctx.log.read())
    # reports: personal stays private, learn is shareable and names no current stock
    assert list((root / "team" / "private").glob("personal_*.md"))
    learn = next((root / "team" / "reports").glob("learn_*.md")).read_text()
    assert compliance.DISCLAIMER in learn
    book = pd.read_csv(sorted(p["private"].glob("*_llm.csv"))[-1])
    assert not any(c in learn for c in book.loc[book["in_book"], "code"])
    # privacy: the public log never names a stock in the book
    public = (root / "team" / "events.jsonl").read_text()
    assert not any(f" {c}" in public or f"{c}:" in public for c in book.loc[book["in_book"], "code"])
    # research needs approval: asked once, runs only after "approve"
    cards = ctx.approvals.all()
    assert len(cards) == 1 and cards[0]["status"] == "pending" and calls == []
    ctx.approvals.answer(cards[0]["id"], True)
    assert lead.run(ctx).status == OK and calls == [["gemini", "groq"]]
    assert ctx.approvals.all()[0]["outcome"] == "finished"
    assert "Research round finished" in ctx.journal.file.read_text()
    # a second run the same day: nothing new to record, ledger still verifies
    ctx2 = _ctx(root, cfg)
    boss.run(ctx2)
    assert len(ledger.read_commits(p)) == 1 and ledger.verify(p) == []
    assert len(ctx2.approvals.all()) == 1                    # just ran research: no new card this month
    assert any(e["action"] == "wait" for e in ctx2.log.read(run=ctx2.log.run_id))
    # a run file dated by its name, not by when git wrote it
    lead2 = agents.ResearchLead(runner=lambda p: 0, runs_dir=root / "runs2")
    (root / "runs2").mkdir()
    (root / "runs2" / "20230601T000000Z_gemini_m_feedback_r1.json").write_text("{}")
    ctx3 = _ctx(root, cfg)
    ctx3.approvals.file = root / "approvals3.json"
    lead2.run(ctx3, today=date(2023, 6, 20))
    assert ctx3.approvals.all() == []                        # 19 days ago: not due
    lead2.run(ctx3, today=date(2023, 7, 3))
    assert len(ctx3.approvals.all()) == 1                    # 32 days ago: asks


def test_bad_or_broken_llm_falls_back_to_templates():
    root, cfg = _setup("llm7")
    for llm in (bad_llm, broken_llm):
        ctx = _ctx(root, cfg, llm=llm)
        boss = chief.ChiefStrategist()
        boss.team["research"] = _research(root)[0]
        boss.run(ctx)
        ex = json.loads(sorted((root / "team" / "private").glob("explanations_*.json"))[-1].read_text())
        text = json.dumps(ex)
        assert "target" not in text.lower() and "buy" not in text.lower()
        assert all(not compliance.check(v["summary"] + " " + v["watch"]) for v in ex["stocks"].values())
    assert any(e["action"] == "reject" for e in ctx.log.read(private=True))
    assert any(e["action"] == "llm-error" for e in ctx.log.read())


def test_compliance_officer_blocks_a_bad_report():
    root, cfg = _setup("block7")
    ctx = _ctx(root, cfg)
    ctx.shared["drafts"] = [{"mode": "learn", "file": "learn_x.md", "as_of": date(2023, 5, 1),
                             "text": "Buy S001 now.\n" + compliance.DISCLAIMER, "current_names": set()}]
    assert agents.ComplianceOfficer().run(ctx).status == STOP
    assert not (root / "team" / "reports" / "learn_x.md").exists()
    assert (root / "team" / "private" / "BLOCKED_learn_x.md").exists()


def test_ask_without_llm_uses_the_log():
    root, cfg = _setup("ask7")
    ctx = _ctx(root, cfg)
    agents.DataChecker().run(ctx)
    ans = chief.ask(ctx, "what did the data checker find?")
    assert "Data Checker" in ans
    assert chief.ask(ctx, "anything?", ) and "buy" not in chief.ask(_ctx(root, cfg, llm=bad_llm), "x").lower()


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\nall {len(tests)} tests passed")
