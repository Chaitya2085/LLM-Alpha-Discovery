"""Tests for Phase 4 (weekly judge, Critic, feedback loop), on a synthetic market with one
planted effect and a fake LLM. No internet or API keys needed.

The planted effect: heavily traded stocks drift up. So `Mean(volume, 20)` works, and
`-Mean(volume, 20)` works in reverse. The fake LLM proposes the reversed idea first and,
only if the Critic's feedback says it worked in reverse, flips it. The control arm never
sees feedback, so it never flips.

Run:  python tests/test_phase4.py   (or pytest)
"""
import dataclasses
import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))
import critic  # noqa: E402
import feedback_loop as fl  # noqa: E402
import generate_factors  # noqa: E402
import weekly  # noqa: E402
from data import load_market  # noqa: E402
from synth_market import make_market  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="phase4_test_"))
START, END = "2016-01-01", "2019-12-31"


CFG = make_market(TMP)
generate_factors.LIB = TMP / "library.jsonl"
fl.RUNS4 = TMP / "llm_runs" / "phase4"


@dataclasses.dataclass
class Resp:
    provider: str
    model: str
    text: str
    usage: dict = None


class FakeLLM:
    """Proposes the reversed liquidity idea; flips anything the feedback calls REVERSE."""
    provider, model = "fake", "m1"

    def __init__(self):
        self.prompts = []

    def complete(self, system, user):
        self.prompts.append(user)
        rnd = int(re.search(r"Round (\d+)", user).group(1))
        facs = []
        flipped = re.findall(r"Worked in REVERSE.*?\n((?:- .*\n?)+)", user)
        if flipped:
            for line in flipped[0].strip().splitlines():
                expr = line.split(":", 1)[1].split("->")[0].strip()
                facs.append({"name": f"flip_r{rnd}", "category": "volume", "expression": f"-({expr})",
                             "hypothesis": "the opposite of what worked in reverse"})
        facs += [
            {"name": f"noise_a_r{rnd}", "category": "momentum", "expression": f"Delta(close, {rnd})",
             "hypothesis": "noise"},
            {"name": f"broken_r{rnd}", "category": "other", "expression": "Bogus(close)", "hypothesis": "typo"},
            {"name": f"illiquid_r{rnd}", "category": "volume", "expression": f"-Mean(volume, {20 + rnd})",
             "hypothesis": "illiquid stocks earn a premium"},
        ]
        return Resp(self.provider, self.model, json.dumps({"factors": facs}))


def _lab():
    return fl.Lab(CFG, START, END)


def test_weekly_judge_finds_planted_effect_and_counts_costs():
    m = load_market("synth")
    ws = weekly.setup(m, START, END)
    assert len(ws.days) > 150 and ws.days[-1] <= pd.Timestamp(END)
    # holding period never crosses into the held-out period
    pos = {d: i for i, d in enumerate(m.dates)}
    assert m.dates[pos[ws.days[-1]] + weekly.HOLD + 1] <= pd.Timestamp(END)
    good = weekly.judge(m.volume.rolling(20).mean(), ws)
    bad = weekly.judge(-m.volume.rolling(20).mean(), ws)
    noise = weekly.judge(pd.DataFrame(np.random.default_rng(0).normal(size=m.close.shape),
                                      index=m.dates, columns=m.close.columns), ws)
    assert good["weekly_ic_t"] > 3 and good["net_excess_ann"] > 0.03 and good["hit_rate"] > 0.55
    assert bad["weekly_ic_t"] < -3 and bad["net_excess_ann"] < 0
    assert abs(noise["weekly_ic_t"]) < 3 and noise["turnover"] > 0.7      # random picks churn
    assert good["turnover"] < 0.3 and good["cost_ann"] < noise["cost_ann"]
    assert abs(noise["cost_ann"] - noise["turnover"] * 52 * (m.buy_cost + m.sell_cost)) < 0.01


def test_critic_verdicts():
    base = {"status": "valid", "judge_status": "scored", "years_same_sign": 0.9, "hit_rate": 0.6,
            "turnover": 0.2, "gross_excess_ann": 0.06, "max_corr_known": 0.3, "max_corr_prev": 0.1}
    cases = [
        ({"weekly_ic_t": 4, "net_excess_ann": 0.05}, "star"),
        ({"weekly_ic_t": 2.5, "net_excess_ann": 0.01}, "keep"),
        ({"weekly_ic_t": 3, "net_excess_ann": -0.01}, "costly"),
        ({"weekly_ic_t": -4, "net_excess_ann": -0.05}, "backwards"),
        ({"weekly_ic_t": 1, "net_excess_ann": 0.01}, "weak"),
        ({"weekly_ic_t": 3, "net_excess_ann": 0.02, "years_same_sign": 0.4}, "unstable"),
        ({"weekly_ic_t": 5, "net_excess_ann": 0.05, "max_corr_known": 0.9, "closest_known": "STD20"}, "copy"),
        ({"status": "invalid", "error": "unknown operator"}, "broken"),
        ({"status": "duplicate"}, "duplicate"),
    ]
    for extra, want in cases:
        got, why = critic.verdict({**base, **extra})
        assert got == want, (extra, got, why)


def test_feedback_loop_feedback_beats_control_and_is_market_blind():
    lib = TMP / "library.jsonl"
    # Phase 2 history of this model: the reversed idea (works backwards) -> round-0 feedback
    seed = generate_factors.ingest(json.dumps({"factors": [
        {"name": "illiquid_premium", "category": "volume", "expression": "-Mean(volume, 20)", "hypothesis": "x"},
        {"name": "short_rev", "category": "reversal", "expression": "-Delta(close, 3)", "hypothesis": "x"}]}),
        {"provider": "fake", "model": "m1", "prompt_version": "v1.1", "run_file": "seed", "created_at": "x"}, [])
    generate_factors.append_library(seed)
    lab = _lab()
    llm = FakeLLM()
    fl.run_researcher(llm, lab, rounds=2, per_round=4, arms=["feedback", "control"], stamp="T", log=lambda *a: None)

    recs = [json.loads(x) for x in lib.read_text().splitlines()]
    p4 = [r for r in recs if r.get("phase") == 4]
    assert {r["arm"] for r in p4} == {"feedback", "control"} and {r["round"] for r in p4} == {1, 2}
    assert len(list(fl.RUNS4.glob("*.json"))) == 4
    # the feedback arm saw the Critic's notes, the control arm didn't
    fb_prompts = [p for p in llm.prompts if "Results of your previous round" in p]
    assert len(fb_prompts) == 2 and "REVERSE" in fb_prompts[0]
    for p in llm.prompts:
        assert not re.search(r"\b(19|20)\d\d\b", p), "prompts must not contain years"
        assert not re.search(r"india|china|nse|nifty|csi", p, re.I), "prompts must be market-blind"

    s = fl.summarize(lab)
    rounds = pd.read_csv(CFG.results / "phase4_rounds.csv")
    fb = rounds[rounds["arm"] == "feedback"]["useful"].sum()
    ct = rounds[rounds["arm"] == "control"]["useful"].sum()
    assert fb >= 1 and ct == 0, (fb, ct)
    assert s["useful_pct_feedback"] > s["useful_pct_control"]
    assert s["best"] and s["best"][0]["arm"] == "feedback"
    f = pd.read_csv(CFG.results / "phase4_factors.csv")
    assert set(f["verdict"]) >= {"broken", "copy", "star", "weak"}
    assert (CFG.results / "phase4_feedback").is_dir()

    # the Phase 2 scorer ignores Phase 4 factors (they have their own tables)
    assert sum(r.get("phase", 2) != 4 for r in recs) == 2


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\nall {len(tests)} tests passed")
