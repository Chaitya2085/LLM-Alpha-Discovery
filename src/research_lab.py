"""The research lab: test one idea the way every factor in this project is tested.

    idea in plain English -> a Researcher (LLM) writes 1-3 formulas in the factor language
    or your own formula    -> checked by the same parser
    -> the weekly judge on 2014-2020 only (top fifth, held a week, after Indian costs,
       against holding all stocks equally) -> the Critic's verdict and reason

Nothing here looks at 2021 or later: those years belong to the final exam. Results are not saved
to the factor library unless you ask (save_to_library).
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from datetime import datetime, timezone

import numpy as np

import critic
import weekly
from generate_factors import append_library, ingest, load_library
from prompts import SYSTEM_WEEKLY

LAB_PROMPT_VERSION = "lab-v1"
MAX_IDEA = 600

_lab = None
_lab_lock = threading.Lock()


def get_lab(cfg):
    """The data and the weekly judge, loaded once (about 10-30 seconds the first time)."""
    global _lab
    with _lab_lock:
        if _lab is None or _lab.cfg.name != cfg.name:
            from feedback_loop import Lab
            _lab = Lab(cfg)
        return _lab


def idea_prompt(idea: str, n: int = 3) -> str:
    return (f"A person described this stock-picking idea in plain words:\n\n\"{idea.strip()[:MAX_IDEA]}\"\n\n"
            f"Write {n} different factor expressions that test this idea as faithfully as possible "
            "(different windows or formulations of the same idea, not unrelated ideas). If part of the idea "
            "needs data the language doesn't have (news, earnings, sectors), test the closest part that can "
            "be measured and say so in the hypothesis. Check every expression against the language rules "
            "(operator names, argument counts, integer windows 1-60).")


def curve(lab, expression: str) -> dict:
    """Growth of 1 in the factor's top fifth (after costs) and in all stocks equally, 2014-2020."""
    ws = lab.ws
    f = lab.ev(expression)
    lab.ev.clear_cache()
    s = f.reindex(index=ws.days, columns=ws.member.columns).where(ws.member)
    top = (s.rank(axis=1, pct=True) > 1 - weekly.TOP) & ws.buyable
    n = top.sum(axis=1)
    gross = ws.fwd.where(top).mean(axis=1)
    prev = top.shift(1, fill_value=False)
    turnover = (1 - (top & prev).sum(axis=1) / n.where(n > 0)).fillna(1.0)
    net = (gross - turnover * ws.cost_round_trip).fillna(0)
    univ = ws.univ.fillna(0)
    return {"dates": [d.strftime("%Y-%m-%d") for d in ws.days],
            "factor": np.round(np.cumprod(1 + net.to_numpy()), 4).tolist(),
            "equal": np.round(np.cumprod(1 + univ.to_numpy()), 4).tolist()}


def evaluate(cfg, reply_text: str, source: dict, with_curve: bool = True) -> list[dict]:
    """Parse a reply ({"factors": [...]}) and judge every formula in it (no saving)."""
    from feedback_loop import assess
    library = load_library()
    recs = ingest(reply_text, source, library)
    if not recs:
        return []
    lab = get_lab(cfg)
    by_id = {r["id"]: r for r in recs}
    reviewed = [{**by_id.get(r["id"], {}), **r} for r in assess(lab, recs, history={})]   # keep provider, idea...
    known = {r["expression"]: r for r in library if r.get("status") == "valid"}
    for r in reviewed:
        if r.get("status") == "duplicate" and r.get("expression") in known:
            k = known[r["expression"]]
            r["reason"] = f"already in the library as {k['id']} \"{k.get('name', '')}\""
        if with_curve and r.get("judge_status") == "scored":
            try:
                r["curve"] = curve(lab, r["expression"])
            except Exception:  # noqa: BLE001 - the verdict still stands without a chart
                r["curve"] = None
    return [_clean(r) for r in reviewed]


def _clean(r: dict) -> dict:
    out = {}
    for k, v in r.items():
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            v = None
        elif isinstance(v, (np.floating, np.integer)):
            v = v.item()
        out[k] = v
    return out


def save_to_library(rec: dict) -> dict:
    """Add one tested formula to the factor library (with its lab verdict)."""
    library = load_library()
    if any(r.get("expression") == rec.get("expression") and r.get("status") == "valid" for r in library):
        raise ValueError("this formula is already in the library")
    keep = {k: rec.get(k) for k in ("name", "category", "raw_expression", "expression", "hypothesis", "provider",
                                     "model", "prompt_version", "n_nodes", "depth", "lookback", "fields", "idea")}
    new = {"id": f"F{len(library) + 1:04d}", **keep, "phase": "lab", "status": "valid", "error": None,
           "run_file": "app research lab", "created_at": datetime.now(timezone.utc).isoformat(),
           "lab": {k: rec.get(k) for k in ("verdict", "reason", "weekly_ic", "weekly_ic_t", "net_excess_ann",
                                            "gross_excess_ann", "hit_rate", "turnover", "years_same_sign")}}
    append_library([new])
    return new


class Jobs:
    """Lab runs happen in the background; the page polls for the result."""

    def __init__(self, cfg, llm_factory, log=None):
        self.cfg, self.llm_factory, self.log = cfg, llm_factory, log
        self.jobs: dict[str, dict] = {}
        self.lock = threading.Lock()

    def start(self, idea: str | None = None, expression: str | None = None) -> dict:
        jid = uuid.uuid4().hex[:8]
        job = {"id": jid, "idea": idea, "expression": expression, "status": "queued",
               "step": "Starting", "results": [], "error": None, "started": time.time()}
        with self.lock:
            busy = [j for j in self.jobs.values() if j["status"] in ("queued", "running")]
            if busy:
                raise RuntimeError("the lab is already testing an idea; wait for it to finish")
            self.jobs[jid] = job
        threading.Thread(target=self._run, args=(job,), daemon=True).start()
        return job

    def get(self, jid: str) -> dict | None:
        return self.jobs.get(jid)

    def _say(self, msg: str, status: str = "ok") -> None:
        if self.log:
            self.log("Research lead", "lab", msg, status)

    def _run(self, job: dict) -> None:
        job["status"] = "running"
        try:
            if job["expression"]:
                reply = json.dumps({"factors": [{"name": "my_formula", "category": "other",
                                                 "expression": job["expression"], "hypothesis": "written by you"}]})
                source = {"provider": "you", "model": "app research lab", "prompt_version": LAB_PROMPT_VERSION,
                          "run_file": "app research lab", "created_at": datetime.now(timezone.utc).isoformat()}
                self._say("Testing a formula you wrote (2014-2020, weekly, after costs)")
            else:
                job["step"] = "A Researcher is turning your idea into formulas"
                llm = self.llm_factory()
                if llm is None:
                    raise RuntimeError("no LLM key found: add GEMINI_API_KEY or GROQ_API_KEY to .env, "
                                       "or write the formula yourself")
                self._say("Turning an idea from you into formulas")
                try:
                    reply = llm(SYSTEM_WEEKLY, idea_prompt(job["idea"]), max_tokens=4000)
                except TypeError:                       # a simple (system, user) callable
                    reply = llm(SYSTEM_WEEKLY, idea_prompt(job["idea"]))
                provider = (getattr(llm, "used", None) or (getattr(llm, "providers", None) or ["llm"])[0])
                source = {"provider": provider, "model": "app research lab", "prompt_version": LAB_PROMPT_VERSION,
                          "run_file": "app research lab", "created_at": datetime.now(timezone.utc).isoformat(),
                          "idea": job["idea"][:MAX_IDEA]}
            job["step"] = "Loading 2014-2020 data and the weekly judge (first time takes a little while)"
            get_lab(self.cfg)
            job["step"] = "The weekly judge and the Critic are testing each formula"
            res = evaluate(self.cfg, reply, source)
            if not res:
                raise RuntimeError("the Researcher's reply had no formulas in it; try rewording the idea")
            job["results"] = res
            good = sum(r["verdict"] in critic.USEFUL for r in res)
            self._say(f"Lab result: {good} of {len(res)} formula(s) work after costs", "ok" if good else "warning")
            job["status"] = "done"
        except Exception as e:  # noqa: BLE001 - shown to the person on the page
            job["status"], job["error"] = "failed", str(e)[:300]
            self._say(f"Lab run failed: {str(e)[:120]}", "warning")
        job["step"] = None
