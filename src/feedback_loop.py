"""Phase 4: the feedback loop. Researcher agents propose factors, the weekly judge tests them,
the Critic explains the results, and the Researcher tries again with that feedback.

    python feedback_loop.py --providers gemini groq --rounds 3 --per-round 15      # India (default)
    python feedback_loop.py --market china --providers gemini
    python feedback_loop.py --summarize-only        # no API calls: rebuild tables from the library

For each Researcher (one per LLM):
  round 0   its Phase 2 factors, judged weekly -> first feedback
  rounds 1+ two arms, so we can tell whether feedback really helps:
              feedback  the prompt includes the Critic's feedback on the previous round
              control   the same prompt without feedback (just "propose new factors")

Every prompt and reply is saved under llm_runs/phase4/, every factor goes into the library
(tagged phase 4, round, arm), and all tables are rebuilt from the library at the end, so a
run can be reproduced with --summarize-only. Judging uses 2014-2020 only.

Writes results/<market>/phase4_rounds.csv, phase4_factors.csv, phase4_summary.json and the
feedback texts in results/<market>/phase4_feedback/.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import critic
import markets
import weekly
from dsl import Evaluator
from generate_factors import RUNS, append_library, ingest, load_library
from llm import PROVIDERS, LLMClient, LLMError
from prompts import PROMPT_VERSION_P4, SYSTEM_WEEKLY, round_prompt

ROOT = Path(__file__).resolve().parents[1]
RUNS4 = RUNS / "phase4"
SEED_PROVIDER = "claude-in-chat"
N_SAMPLE = 60


def model_label(r: dict) -> str:
    if r.get("provider") == SEED_PROVIDER:
        return "seed: claude (chat)"
    return f"{r.get('provider')}: {r.get('model')}"


def _ranks(mat: np.ndarray) -> np.ndarray:
    r = pd.DataFrame(mat).rank(axis=1, pct=True).to_numpy()
    return np.nan_to_num(r - np.nanmean(r, axis=1, keepdims=True))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    num = (a * b).sum(1)
    den = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    ok = den > 0
    return float(np.mean(num[ok] / den[ok])) if ok.any() else 0.0


class Lab:
    """Evaluates factors weekly and checks them against known features and earlier factors."""

    def __init__(self, cfg: markets.MarketConfig, start: str = weekly.DISC_START, end: str = weekly.DISC_END):
        from data import load_market
        self.cfg = cfg
        self.m = load_market(cfg.name)
        self.ws = weekly.setup(self.m, start, end)
        self.ev = Evaluator(self.m)
        days = self.ws.days
        self.sample = days[np.linspace(0, len(days) - 1, min(N_SAMPLE, len(days))).astype(int)]
        self.mem_s = self.m.member.loc[self.sample]
        self.known = self._known_features()
        self.cache: dict[str, dict] = {}

    def _known_features(self) -> dict[str, np.ndarray]:
        path = self.cfg.proc / "alpha158.parquet"
        if not path.exists():
            return {}
        try:
            X = pd.read_parquet(path, filters=[("date", "in", list(self.sample))])
        except Exception:  # noqa: BLE001 - older pyarrow: read all, then filter
            X = pd.read_parquet(path)
            X = X[X.index.get_level_values("date").isin(self.sample)]
        out = {}
        for col in X.columns:
            w = X[col].unstack("code").reindex(index=self.sample, columns=self.m.member.columns)
            out[col] = _ranks(w.astype("float64").where(self.mem_s).to_numpy())
        return out

    def measure(self, expression: str) -> dict:
        """Weekly judge + novelty vs known features for one expression (cached)."""
        if expression in self.cache:
            return self.cache[expression]
        try:
            f = self.ev(expression)
            j = weekly.judge(f, self.ws)
            ranks = _ranks(f.reindex(index=self.sample, columns=self.m.member.columns)
                           .where(self.mem_s).to_numpy())
        except Exception as e:  # noqa: BLE001 - a factor that can't be computed is just 'broken'
            j, ranks = {"status": "eval_error", "error": str(e)[:200]}, None
        finally:
            self.ev.clear_cache()
        res = {"judge": j, "ranks": ranks, "known": (0.0, "")}
        if ranks is not None and self.known:
            corrs = {k: abs(_corr(ranks, v)) for k, v in self.known.items()}
            best = max(corrs, key=corrs.get)
            res["known"] = (corrs[best], best)
        self.cache[expression] = res
        return res


def assess(lab: Lab, recs: list[dict], history: dict[str, tuple[str, np.ndarray]]) -> list[dict]:
    """Judge a round of records. `history` holds earlier factors with a real signal (any
    direction) from the same researcher+arm, for the 'copy of an earlier factor' check
    (same direction only: the flipped version of a backwards idea is new). Updated in place."""
    out = []
    for r in recs:
        row = {k: r.get(k) for k in ("id", "name", "category", "expression", "raw_expression", "hypothesis",
                                       "status", "error")}
        if r.get("status") == "valid":
            ms = lab.measure(r["expression"])
            j = ms["judge"]
            row["judge_status"] = j["status"]
            row["judge_error"] = j.get("error")
            row.update({k: v for k, v in j.items() if k not in ("status", "error")})
            row["max_corr_known"], row["closest_known"] = ms["known"]
            prev_best, prev_name = 0.0, ""
            if ms["ranks"] is not None:
                for name, rk in history.values():
                    c = _corr(ms["ranks"], rk)   # signed: flipping a backwards idea is not a copy
                    if c > prev_best:
                        prev_best, prev_name = c, name
                if j["status"] == "scored" and abs(j["weekly_ic_t"]) >= critic.T_SIGNAL:
                    history[r["id"]] = (r.get("name") or r["id"], ms["ranks"])
            row["max_corr_prev"], row["closest_prev"] = prev_best, prev_name
        out.append(row)
    return critic.review(out)


# ---------------------------------------------------------------- the loop

def researcher_of(r: dict) -> str:
    return r.get("researcher") or model_label(r)


def phase2_of(library: list[dict], label: str) -> list[dict]:
    """A researcher's Phase 2 factors. 'openrouter: auto-free' picks whichever free model is
    available, so it inherits all earlier OpenRouter factors."""
    prov, _, model = label.partition(": ")
    if model == "auto-free":
        return [r for r in library if r.get("phase", 2) != 4 and r.get("provider") == prov]
    return [r for r in library if r.get("phase", 2) != 4 and model_label(r) == label]


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def _safe(s: str) -> str:
    return s.replace("/", "_").replace(":", "_").replace(" ", "_")


def run_researcher(client, lab: Lab, rounds: int, per_round: int, arms: list[str],
                   stamp: str, log=print) -> None:
    """Ask one LLM for `rounds` rounds per arm; everything is saved to the library."""
    library = load_library()
    label = f"{client.provider}: {client.model}"
    phase2 = phase2_of(library, label)
    base = [r for r in library if r.get("phase", 2) != 4 and r["status"] == "valid"]
    hist0: dict = {}
    reviewed0 = assess(lab, phase2, hist0) if phase2 else []
    first_feedback = critic.feedback_text(reviewed0) if reviewed0 else None
    if reviewed0:
        s0 = critic.round_stats(reviewed0)
        log(f"  {label} round 0 (its Phase 2 factors): {s0['useful']}/{s0['proposed']} work weekly after costs")
    RUNS4.mkdir(parents=True, exist_ok=True)
    for arm in arms:
        feedback = first_feedback if arm == "feedback" else None
        history = dict(hist0)
        own: list[str] = []
        for rnd in range(1, rounds + 1):
            existing = [r["expression"] for r in base] + own
            prompt = round_prompt(per_round, rnd, feedback, existing)
            t = time.time()
            log(f"  {label} [{arm}] round {rnd}/{rounds}: asking for {per_round} factors ...")
            try:
                resp = client.complete(SYSTEM_WEEKLY, prompt)
            except LLMError as e:
                log(f"    failed: {e}; stopping this arm")
                break
            run = {"provider": resp.provider, "model": resp.model, "prompt_version": PROMPT_VERSION_P4,
                   "created_at": datetime.now(timezone.utc).isoformat(), "phase": 4, "round": rnd, "arm": arm,
                   "market": lab.cfg.name, "system": SYSTEM_WEEKLY, "user": prompt, "feedback": feedback,
                   "reply": resp.text, "usage": getattr(resp, "usage", None), "seconds": round(time.time() - t, 1)}
            fp = RUNS4 / f"{stamp}_{_safe(resp.provider)}_{_safe(resp.model)}_{arm}_r{rnd}.json"
            fp.write_text(json.dumps(run, indent=2))
            library = load_library()
            src = {"provider": resp.provider, "model": resp.model, "prompt_version": PROMPT_VERSION_P4,
                   "run_file": _rel(fp), "created_at": run["created_at"],
                   "phase": 4, "round": rnd, "arm": arm, "market": lab.cfg.name, "researcher": label}
            # duplicates: checked against Phase 2 factors and this arm's own, so the two arms
            # don't penalise each other for proposing the same idea
            pool = [r for r in library if r.get("phase", 2) != 4
                    or (researcher_of(r) == label and r.get("arm") == arm and r.get("market") == lab.cfg.name)]
            recs = ingest(resp.text, src, library, dedup_against=pool)
            append_library(recs)
            own += [r["expression"] for r in recs if r["status"] == "valid"]
            reviewed = assess(lab, recs, history)
            st = critic.round_stats(reviewed)
            log(f"    {st['proposed']} proposed: {st['useful']} work after costs ({st['stars']} strongly), "
                f"{st['verdicts'].get('backwards', 0)} backwards, {st['verdicts'].get('copy', 0)} copies, "
                f"{st['verdicts'].get('broken', 0)} broken")
            if arm == "feedback":
                feedback = critic.feedback_text(reviewed)


def summarize(lab: Lab) -> dict:
    """Rebuild every Phase 4 table from the library (no API calls)."""
    library = load_library()
    p4 = [r for r in library if r.get("phase") == 4 and r.get("market", lab.cfg.name) == lab.cfg.name]
    res = lab.cfg.results
    fb_dir = res / "phase4_feedback"
    fb_dir.mkdir(parents=True, exist_ok=True)
    rounds_rows, factor_rows = [], []
    for label in sorted({researcher_of(r) for r in p4}):
        phase2 = phase2_of(library, label)
        hist0: dict = {}
        rev0 = assess(lab, phase2, hist0) if phase2 else []
        if rev0:
            rounds_rows.append({"researcher": label, "arm": "both", "round": 0, **_flat(critic.round_stats(rev0))})
            (fb_dir / f"{_safe(label)}_r0.txt").write_text(critic.feedback_text(rev0))
        for arm in ("feedback", "control"):
            history = dict(hist0)
            mine = [r for r in p4 if researcher_of(r) == label and r.get("arm") == arm]
            for rnd in sorted({r["round"] for r in mine}):
                rev = assess(lab, [r for r in mine if r["round"] == rnd], history)
                rounds_rows.append({"researcher": label, "arm": arm, "round": rnd, **_flat(critic.round_stats(rev))})
                if arm == "feedback":
                    (fb_dir / f"{_safe(label)}_r{rnd}.txt").write_text(critic.feedback_text(rev))
                for x in rev:
                    factor_rows.append({"researcher": label, "arm": arm, "round": rnd, **x})
    rounds = pd.DataFrame(rounds_rows)
    factors = pd.DataFrame(factor_rows)
    res.mkdir(parents=True, exist_ok=True)
    rounds.to_csv(res / "phase4_rounds.csv", index=False)
    factors.to_csv(res / "phase4_factors.csv", index=False)
    summary = _summary(rounds, factors, lab)
    (res / "phase4_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _flat(st: dict) -> dict:
    v = st.pop("verdicts")
    return {**st, **{f"n_{k}": v.get(k, 0) for k in critic.ORDER}}


def _summary(rounds: pd.DataFrame, factors: pd.DataFrame, lab: Lab) -> dict:
    out = {"market": lab.cfg.name, "judging": "weekly, long-only top fifth, after costs, vs equal-weight universe",
           "period": [str(lab.ws.days[0].date()), str(lab.ws.days[-1].date())],
           "researchers": int(rounds["researcher"].nunique()) if len(rounds) else 0,
           "factors_proposed": int(len(factors)), "per_researcher": [], "best": []}
    if not len(rounds):
        return out
    for label, g in rounds.groupby("researcher"):
        row = {"researcher": label}
        r0 = g[g["round"] == 0]
        row["round0_useful_pct"] = float(r0["useful_pct"].iloc[0]) if len(r0) else None
        for arm in ("feedback", "control"):
            ga = g[g["arm"] == arm].sort_values("round")
            if len(ga):
                row[f"{arm}_first_useful_pct"] = float(ga["useful_pct"].iloc[0])
                row[f"{arm}_last_useful_pct"] = float(ga["useful_pct"].iloc[-1])
                row[f"{arm}_mean_useful_pct"] = float(ga["useful_pct"].mean())
                row[f"{arm}_rounds"] = int(len(ga))
                row[f"{arm}_backwards_pct"] = float(ga["backwards_pct"].mean())
        out["per_researcher"].append(row)
    if len(factors):
        good = factors[factors["verdict"].isin(critic.USEFUL)].sort_values("net_excess_ann", ascending=False)
        out["useful_total"] = int(len(good))
        out["useful_by_arm"] = {k: int(v) for k, v in good["arm"].value_counts().items()}
        out["best"] = [{"id": r.id, "name": r.name, "researcher": r.researcher, "arm": r.arm, "round": int(r.round),
                        "expression": r.expression, "net_excess_ann": round(float(r.net_excess_ann), 4),
                        "hit_rate": round(float(r.hit_rate), 3), "weekly_ic_t": round(float(r.weekly_ic_t), 2),
                        "turnover": round(float(r.turnover), 3), "verdict": r.verdict}
                       for r in good.head(10).itertuples()]
        fb = rounds[rounds["arm"] == "feedback"]["useful_pct"].mean()
        ct = rounds[rounds["arm"] == "control"]["useful_pct"].mean()
        out["useful_pct_feedback"] = None if pd.isna(fb) else round(float(fb), 1)
        out["useful_pct_control"] = None if pd.isna(ct) else round(float(ct), 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--market", default=os.environ.get("LLM_ALPHA_MARKET", "india"))
    ap.add_argument("--providers", nargs="*", default=[], choices=list(PROVIDERS))
    ap.add_argument("--model", default=None, help="model (only with a single provider)")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--per-round", type=int, default=15)
    ap.add_argument("--no-control", action="store_true", help="skip the no-feedback control arm (half the calls)")
    ap.add_argument("--summarize-only", action="store_true")
    a = ap.parse_args()
    if a.model and len(a.providers) != 1:
        raise SystemExit("--model can only be used with exactly one provider")
    cfg = markets.get(a.market)
    os.environ["LLM_ALPHA_MARKET"] = cfg.name
    t0 = time.time()
    print(f"  loading {cfg.name} data and the weekly judge ...", flush=True)
    lab = Lab(cfg)
    print(f"  {len(lab.ws.days)} weekly rebalances, {lab.ws.days[0].date()} to {lab.ws.days[-1].date()}", flush=True)
    arms = ["feedback"] if a.no_control else ["feedback", "control"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    failed = []
    if not a.summarize_only:
        for prov in a.providers:
            try:
                client = LLMClient(prov, a.model)
            except (LLMError, SystemExit) as e:
                print(f"  {prov}: skipped ({e})")
                failed.append(prov)
                continue
            run_researcher(client, lab, a.rounds, a.per_round, arms, stamp)
    s = summarize(lab)
    print(json.dumps({k: v for k, v in s.items() if k != "best"}, indent=2))
    for b in s.get("best", [])[:5]:
        print(f"  best: {b['name']} ({b['researcher']}, {b['arm']} r{b['round']}): {b['net_excess_ann']:+.1%}/yr "
              f"after costs, hit {b['hit_rate']:.0%}  {b['expression']}")
    print(f"  done in {time.time() - t0:.0f}s")
    if failed:
        print(f"  skipped providers: {', '.join(failed)} (run `python src/llm.py check`)")


if __name__ == "__main__":
    main()
