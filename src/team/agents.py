"""The team. Each agent has one job, runs code we have already tested, and logs what it did.

Only three of them use an LLM (the Explainer, the Research lead's Researchers, and the Chief Strategist
when you ask it something), and only to put given facts into words. Every number comes from code.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from . import compliance, explain
from .core import OK, STOP, WARN, Agent, Context, Result

SRC = Path(__file__).resolve().parents[1]
ROOT = SRC.parent


def _py(args: list[str]) -> int:
    return subprocess.run([sys.executable, *args], cwd=SRC).returncode


def _frozen() -> dict | None:
    import ledger
    return json.loads(ledger.FROZEN.read_text()) if ledger.FROZEN.exists() else None


def _tag(variant: str) -> str:
    return "llm" if "LLM" in variant else "a158"


def latest_book(ctx: Context, variant: str) -> tuple[str | None, pd.DataFrame | None]:
    import ledger
    p = ctx.ledger_paths or ledger.paths()
    files = sorted(p["private"].glob(f"*_{_tag(variant)}.csv")) if p["private"].exists() else []
    if not files:
        return None, None
    df = pd.read_csv(files[-1])
    return str(df["as_of"].iloc[0]), df


# ---------------------------------------------------------------- market data desk

class DataCollector(Agent):
    name, desk = "Data Collector", "Market data"

    def run(self, ctx: Context) -> Result:
        if ctx.skip_download:
            self.say(ctx, "skip", "Using the data already on this computer (download skipped)")
            return Result(OK, "skipped")
        self.say(ctx, "start", "Downloading the latest NSE files after the close")
        for step in (["india_download.py"], ["build_india.py"], ["build_features.py", "--market", "india"]):
            code = _py(step)
            if code != 0:
                self.say(ctx, "fail", f"{step[0]} stopped (code {code}); NSE may be busy, try again later", STOP)
                return Result(STOP, f"{step[0]} failed")
        if _py(["risk_lists.py"]) != 0:
            self.say(ctx, "warn", "Could not refresh NSE's surveillance list; using the last one saved", WARN)
        self.say(ctx, "done", "Latest NSE data downloaded and rebuilt")
        return Result(OK, "downloaded")


class DataChecker(Agent):
    name, desk = "Data Checker", "Market data"
    STALE_WARN, STALE_STOP = 6, 14            # calendar days since the last trading day
    MIN_STOCKS = 150                          # the universe is the top 200

    def run(self, ctx: Context, today: date | None = None) -> Result:
        m = ctx.market()
        today = today or ctx.shared.get("today") or date.today()
        last = m.dates[-1]
        age = (today - last.date()).days
        problems, status = [], OK
        if age > self.STALE_STOP:
            problems.append(f"the latest data is {age} days old")
            status = STOP
        elif age > self.STALE_WARN:
            problems.append(f"the latest data is {age} days old")
            status = WARN
        members = m.member.loc[last]
        n = int(members.sum())
        missing = float(m.close.loc[last][members].isna().mean()) if n else 1.0
        if n < self.MIN_STOCKS:
            problems.append(f"only {n} stocks in the universe on {last.date()}")
            status = STOP
        if missing > 0.05:
            problems.append(f"{missing:.0%} of universe stocks have no price on {last.date()}")
            status = STOP if missing > 0.2 else max(status, WARN, key=[OK, WARN, STOP].index)
        if pd.isna(m.index_close.loc[last]):
            problems.append(f"no Nifty 200 close on {last.date()}")
            status = STOP
        r5 = (m.close.iloc[-5:] / m.close.ffill().shift(1).iloc[-5:] - 1).where(m.member.iloc[-5:])
        big = r5.abs().stack()
        big = big[big > 0.25]
        if len(big):
            problems.append(f"{len(big)} one-day moves over 25% in the last week (check corporate_actions.csv)")
            if status == OK:
                status = WARN
            for (d, code), v in big.items():
                self.say(ctx, "flag", f"{code} moved {r5.at[d, code]:+.0%} on {d.date()}", WARN, private=True)
        ctx.shared["as_of"] = last.date()
        msg = (f"Data through {last.date()}: {n} stocks, all checks passed" if not problems
               else f"Data through {last.date()}: " + "; ".join(problems))
        self.say(ctx, "check", msg, status, as_of=str(last.date()), stocks=n)
        return Result(status, msg, {"as_of": str(last.date())})


# ---------------------------------------------------------------- model room

class RegimeWatcher(Agent):
    name, desk = "Regime Watcher", "Model room"

    def run(self, ctx: Context) -> Result:
        import phase5_model as p5
        reg = p5.regimes(ctx.market())["regime"].dropna()
        if reg.empty:
            self.say(ctx, "skip", "Not enough history to label the market mood yet", WARN)
            return Result(WARN, "no regime")
        now_, before = reg.iloc[-1], reg.iloc[-6] if len(reg) > 5 else reg.iloc[0]
        ctx.shared["regime"] = now_
        if now_ != before:
            msg = f"Market mood changed this week: {before} -> {now_}"
            ctx.journal.add(self.name, msg)
            self.say(ctx, "change", msg, WARN, regime=now_)
        else:
            msg = f"Market mood: {now_} (unchanged this week)"
            self.say(ctx, "label", msg, regime=now_)
        return Result(OK, msg, {"regime": now_})


class ModelTrainer(Agent):
    """Weekly: records the frozen model's predictions in the ledger, reveals and scores old ones, and
    watches whether the live record is drifting from what the final exam showed."""
    name, desk = "Model Trainer", "Model room"

    def run(self, ctx: Context) -> Result:
        import ledger
        p = ctx.ledger_paths or ledger.paths()
        if not _frozen():
            self.say(ctx, "skip", "Phase 6 settings are not frozen yet; nothing to record", WARN)
            return Result(WARN, "not frozen")
        if ctx.weekly:
            commits = ledger.read_commits(p)
            as_of = str(ctx.market().dates[-1].date())
            if commits and commits[-1]["as_of"] >= as_of:
                self.say(ctx, "skip", f"This week's predictions are already in the ledger ({as_of})")
            else:
                self.say(ctx, "train", "Retraining the frozen model on all data up to the latest close")
                rows = ledger.record(ctx.cfg, p=p, log=lambda *a: None)
                for r in rows:
                    self.say(ctx, "record", f"{r['variant']}: predictions fingerprinted for the week after "
                             f"{r['as_of']} ({r['sha256'][:12]})", fingerprint=r["sha256"])
            revealed = ledger.reveal(p)
            if revealed:
                self.say(ctx, "reveal", f"Revealed {len(revealed)} prediction file(s) older than 31 days")
        problems = ledger.verify(p)
        if problems:
            self.say(ctx, "verify", "Ledger check FAILED: " + problems[0], STOP)
            return Result(STOP, problems[0])
        scores = ledger.score(ctx.cfg, p) if (p["revealed"].exists() and any(p["revealed"].glob("*.csv"))) else pd.DataFrame()
        health = {"revealed_weeks": int(len(scores))}
        status, msg = OK, "Ledger checks out"
        if len(scores) >= 8:
            last = scores.sort_values("as_of").groupby("variant").tail(13)
            for v, g in last.groupby("variant"):
                hit, ex = g["beat_index"].mean(), g["excess"].mean() * 52
                health[v] = {"weeks": len(g), "hit_rate": float(hit), "excess_ann": float(ex)}
                if hit < 0.4 and ex < 0:
                    status = WARN
                    msg = f"{v} beat the index in only {hit:.0%} of the last {len(g)} revealed weeks"
        else:
            msg += f"; {len(scores)} revealed week(s) so far (health checks start at 8)"
        ctx.shared["model_health"] = health
        self.say(ctx, "health", msg, status, **health)
        return Result(status, msg, health)


# ---------------------------------------------------------------- risk & compliance

class RiskOfficer(Agent):
    name, desk = "Risk Officer", "Risk & compliance"
    CROWDED = 0.30

    def run(self, ctx: Context) -> Result:
        import risk_lists
        frozen = _frozen()
        if not frozen:
            return Result(OK, "no book yet")
        m = ctx.market()
        _, surv = risk_lists.latest_reg_ind()
        sect = risk_lists.sectors()
        value = (m.volume * m.vwap).iloc[-20:].median()
        thin_cut = value[m.member.iloc[-1]].quantile(0.10)
        r5 = m.close.iloc[-1] / m.close.iloc[-6] - 1
        locked = (m.high.iloc[-1] == m.low.iloc[-1]) & ((m.close.iloc[-1] / m.close.iloc[-2] - 1).abs() >= 0.019)
        all_flags, counts = {}, {}
        for variant in frozen["variants"]:
            as_of, book = latest_book(ctx, variant)
            if book is None:
                continue
            held = book.loc[book["in_book"], "code"].tolist()
            flags = {}
            for c in held:
                f = list(surv.get(str(c).upper(), []))
                if bool(locked.get(c, False)):
                    f.append("locked at its price limit on the last day")
                if r5.get(c, 0) <= -0.15:
                    f.append(f"fell {r5[c]:.0%} in the last week")
                if value.get(c, np.inf) < thin_cut:
                    f.append("thinly traded (bottom 10% of the universe)")
                if f:
                    flags[c] = f
            secs = pd.Series([sect.get(str(c).upper(), "Unknown") for c in held])
            share = secs[secs != "Unknown"].value_counts() / max(len(held), 1)
            crowded = {k: float(v) for k, v in share.items() if v > self.CROWDED}
            for c, f in flags.items():
                self.say(ctx, "flag", f"{variant}: {c}: " + "; ".join(f), WARN, private=True)
            for k, v in crowded.items():
                self.say(ctx, "crowding", f"{variant}: {v:.0%} of the book is in {k}", WARN)
            all_flags[variant] = flags
            counts[variant] = len(flags)
        ctx.shared["risk"] = all_flags
        n = sum(counts.values())
        notes = []
        if not surv:
            notes.append("no surveillance file yet (python src/risk_lists.py --test)")
        if not sect:
            notes.append("no sector list yet")
        msg = (f"{n} risk flag(s) across the books" + (f"; {'; '.join(notes)}" if notes else ""))
        self.say(ctx, "review", msg, WARN if n else OK, flagged=counts)
        return Result(WARN if n else OK, msg, {"flagged": counts})


class ComplianceOfficer(Agent):
    """Checks every draft report; only clean drafts are saved where they can be shared."""
    name, desk = "Compliance Officer", "Risk & compliance"

    def run(self, ctx: Context) -> Result:
        drafts = ctx.shared.get("drafts", [])
        out = ctx.folder / "reports"
        out.mkdir(parents=True, exist_ok=True)
        blocked = 0
        for d in drafts:
            probs = compliance.check(d["text"], d["mode"], d.get("as_of"), current_names=d.get("current_names"),
                                     require_disclaimer=True)
            if probs:
                blocked += 1
                (ctx.folder / "private").mkdir(exist_ok=True)
                (ctx.folder / "private" / f"BLOCKED_{d['file']}").write_text(
                    "Blocked by the Compliance Officer:\n- " + "\n- ".join(probs) + "\n\n" + d["text"])
                self.say(ctx, "block", f"Blocked {d['file']}: " + "; ".join(probs[:3]), STOP)
            else:
                folder = ctx.folder / ("private" if d["mode"] == "personal" else "reports")
                folder.mkdir(exist_ok=True)
                (folder / d["file"]).write_text(d["text"])
                ctx.shared.setdefault("saved_reports", []).append(str(folder / d["file"]))
                self.say(ctx, "approve", f"Cleared {d['file']} ({d['mode']} mode)")
        return Result(STOP if blocked else OK, f"{len(drafts) - blocked} cleared, {blocked} blocked")


# ---------------------------------------------------------------- communications

EXPLAIN_SYSTEM = (
    "You write short, plain-English notes about why a stock-ranking model ranked one stock highly this week. "
    "Use ONLY the facts given. Do not add numbers, company news or opinions. Never tell anyone to buy, sell or "
    "hold, never give price targets, never promise or predict returns. Reply with only a JSON object: "
    '{"summary": "<two short sentences, under 60 words>", "watch": "<one sentence: what would make this view wrong>"}')


class Explainer(Agent):
    name, desk = "Explainer", "Communications"
    TOP = 10

    def run(self, ctx: Context) -> Result:
        import ledger
        from generate_factors import load_library
        frozen = _frozen()
        if not frozen:
            return Result(OK, "no book yet")
        variant = frozen["variants"][0]
        as_of, book = latest_book(ctx, variant)
        p = ctx.ledger_paths or ledger.paths()
        why_f = p["private"] / f"{as_of}_{_tag(variant)}_why.csv" if as_of else None
        if book is None:
            self.say(ctx, "skip", "No book to explain yet", WARN)
            return Result(WARN, "nothing to explain")
        if not why_f.exists():                # recorded before reasons were saved (update 17 or earlier)
            self.say(ctx, "rebuild", f"Rebuilding the reasons for the book of {as_of} (a few minutes)")
            ledger.rebuild_reasons(ctx.cfg, as_of, p=p, log=lambda *a: None)
            if not why_f.exists():
                self.say(ctx, "skip", f"Can't rebuild the reasons for {as_of}: newer data has arrived since. "
                         "Explanations start with next week's book", WARN)
                return Result(WARN, "nothing to explain")
        why = pd.read_csv(why_f)
        lib = {r["id"]: r for r in load_library()}
        top = book[book["in_book"]].sort_values("p", ascending=False).head(self.TOP)
        risk = ctx.shared.get("risk", {}).get(variant, {})
        notes, used_llm = {}, 0
        for r in top.itertuples():
            g = why[(why["code"] == r.code) & ~why["feature"].str.startswith("mkt_")]   # stock-specific reasons
            rows = g.loc[g["contribution"].abs().sort_values(ascending=False).index].head(3)
            facts = [explain.fact(w.feature, w.rank_pct, w.contribution, lib) for w in rows.itertuples()]
            note = self._words(ctx, r.code, facts, risk.get(r.code, []))
            used_llm += note.pop("_llm", 0)
            notes[r.code] = {"confidence": float(r.p), "facts": facts, **note, "risk": risk.get(r.code, [])}
        out = ctx.folder / "private"
        out.mkdir(exist_ok=True)
        (out / f"explanations_{as_of}.json").write_text(json.dumps({"as_of": as_of, "variant": variant,
                                                                    "stocks": notes}, indent=2))
        ctx.shared["explanations"] = {"as_of": as_of, "variant": variant, "stocks": notes}
        how = f"{used_llm} written by the LLM, {len(notes) - used_llm} from templates"
        self.say(ctx, "explain", f"Explained the top {len(notes)} stocks of {variant} ({how})")
        return Result(OK, f"{len(notes)} explained", {"llm": used_llm})

    def _words(self, ctx: Context, code: str, facts: list[str], risks: list[str]) -> dict:
        if ctx.llm is not None:
            from llm import extract_json
            user = json.dumps({"stock": code, "facts": facts, "risk_flags": risks})
            try:
                obj = extract_json(ctx.llm(EXPLAIN_SYSTEM, user) or "")
                s, w = str(obj.get("summary", "")).strip(), str(obj.get("watch", "")).strip()
                if s and w and len(s) < 600 and not compliance.check(s + " " + w):
                    return {"summary": s, "watch": w, "_llm": 1}
                self.say(ctx, "reject", f"LLM note for {code} failed the checks; using the template", WARN,
                         private=True)
            except Exception as e:  # noqa: BLE001 - any failure falls back to the template
                self.say(ctx, "llm-error", f"LLM unavailable ({str(e)[:80]}); using the template", WARN)
                ctx.llm = None
        return {**explain.template_note(facts), "_llm": 0}


class Reporter(Agent):
    """Drafts the weekly evidence reports; the Compliance Officer decides whether they are saved."""
    name, desk = "Reporter", "Communications"

    def run(self, ctx: Context, today: date | None = None) -> Result:
        import ledger
        today = today or ctx.shared.get("today") or date.today()
        as_of = ctx.shared.get("as_of") or ctx.market().dates[-1].date()
        frozen = _frozen() or {"variants": []}
        p = ctx.ledger_paths or ledger.paths()
        commits = ledger.read_commits(p)
        health = ctx.shared.get("model_health", {})
        regime = ctx.shared.get("regime", "unknown")
        current = set()
        for v in frozen["variants"]:
            _, b = latest_book(ctx, v)
            if b is not None:
                current |= set(b.loc[b["in_book"], "code"].astype(str))
        drafts = [self._personal(ctx, as_of, regime, commits, health),
                  self._learn(ctx, today, commits, health, p)]
        drafts[0]["current_names"] = None
        drafts[1]["current_names"] = current
        ctx.shared["drafts"] = drafts
        self.say(ctx, "draft", f"Drafted this week's personal and shareable reports (data through {as_of})")
        return Result(OK, "drafted")

    def _personal(self, ctx, as_of, regime, commits, health) -> dict:
        ex = ctx.shared.get("explanations", {})
        lines = [f"# Weekly evidence report, data through {as_of}", "",
                 "*Personal mode: stays on this computer.*", "", "## This week", "",
                 f"- Market mood: **{regime}**",
                 f"- Ledger: {len({c['as_of'] for c in commits})} week(s) of predictions fingerprinted "
                 f"({len(commits)} books); "
                 f"{health.get('revealed_weeks', 0)} revealed week(s) scored"]
        for k, v in health.items():
            if isinstance(v, dict):
                lines.append(f"- {k}: beat the index in {v['hit_rate']:.0%} of the last {v['weeks']} revealed "
                             f"weeks ({v['excess_ann']:+.1%} a year)")
        for r in ctx.shared.get("results", []):
            if r["status"] != OK:
                lines.append(f"- {r['agent']}: {r['summary']}")
        if ex.get("stocks"):
            lines += ["", f"## Why the model ranks these stocks highly ({ex['variant']})", "",
                      "| Stock | Chance of beating the Nifty 200 next week | Why | What would change the view | Risk flags |",
                      "|---|---|---|---|---|"]
            for code, n in ex["stocks"].items():
                lines.append(f"| {code} | {n['confidence']:.0%} | {n['summary']} | {n['watch']} | "
                             f"{'; '.join(n['risk']) or 'none'} |")
            lines += ["", "A confidence near 50% is normal: one stock in one week is close to a coin flip. "
                          "The model's edge comes from ranking many stocks, not from certainty about one."]
        lines += ["", "---", compliance.DISCLAIMER]
        return {"mode": "personal", "file": f"personal_{as_of}.md", "text": "\n".join(lines), "as_of": None}

    def _learn(self, ctx, today, commits, health, p) -> dict:
        import phase5_model as p5
        m = ctx.market()
        cutoff = pd.Timestamp(today) - pd.Timedelta(days=compliance.LEARN_LAG_DAYS + 1)
        old = m.dates[m.dates <= cutoff]
        as_of = old[-1].date() if len(old) else None
        reg = p5.regimes(m)["regime"].dropna()
        reg_old = reg[reg.index <= cutoff]
        lines = [f"# LLM Alpha Discovery: research update, data through {as_of}", "",
                 f"*Learn mode: uses only data at least {compliance.LEARN_LAG_DAYS} days old and names no stock "
                 "in a current book.*", "", "## The record", "",
                 f"- Weeks of predictions fingerprinted so far: **{len({c['as_of'] for c in commits})}** "
                 f"({len(commits)} books, each locked before its week "
                 "began; see ledger/commitments.csv)",
                 f"- Revealed and scored: **{health.get('revealed_weeks', 0)}** week(s)"]
        if len(reg_old):
            lines.append(f"- Market mood on {reg_old.index[-1].date()}: {reg_old.iloc[-1]}")
        scores = pd.read_csv(p["scores"]) if p["scores"].exists() else pd.DataFrame()
        if len(scores):
            scores = scores[pd.to_datetime(scores["as_of"]) <= cutoff].sort_values("as_of")
            lines += ["", "## Revealed weeks (each book was fingerprinted before its week began)", "",
                      "| Week after | Book | Book return | Nifty 200 | Difference |", "|---|---|---|---|---|"]
            for r in scores.tail(8).itertuples():
                lines.append(f"| {r.as_of} | {r.variant} | {r.book_return:+.2%} | {r.index_return:+.2%} | "
                             f"{r.excess:+.2%} |")
            for v, g in scores.groupby("variant"):
                lines.append(f"\n{v}: beat the index in {g['beat_index'].mean():.0%} of {len(g)} revealed weeks.")
        lines += ["", "---", compliance.DISCLAIMER]
        return {"mode": "learn", "file": f"learn_{as_of}.md", "text": "\n".join(lines), "as_of": as_of}


# ---------------------------------------------------------------- research lab

class ResearchLead(Agent):
    """Once a month, asks permission to run a round of the Phase 4 feedback loop (it spends API calls
    and adds factors to the library). Runs it only after a person approves."""
    name, desk = "Research lead", "Research lab"
    EVERY_DAYS = 30
    PROVIDERS = ["gemini", "groq"]

    def __init__(self, runner=None, runs_dir: Path | None = None):
        self.runs_dir = runs_dir or ROOT / "llm_runs" / "phase4"
        self.runner = runner or (lambda providers: _py(["feedback_loop.py", "--providers", *providers,
                                                        "--rounds", "3", "--per-round", "15"]))

    def run(self, ctx: Context, today: date | None = None) -> Result:
        today = today or ctx.shared.get("today") or date.today()
        card = ctx.approvals.approved("research-round")
        if card:
            self.say(ctx, "start", "Approved: running a research round (Researchers propose, Critic judges)")
            code = self.runner(self.PROVIDERS)
            outcome = "finished" if code == 0 else f"stopped (code {code})"
            ctx.approvals.mark_done(card["id"], outcome)
            ctx.journal.add(self.name, f"Research round {outcome} (approved card {card['id']})")
            self.say(ctx, "done", f"Research round {outcome}", OK if code == 0 else WARN)
            return Result(OK if code == 0 else WARN, outcome)
        # the date is read from the file name (20261012T...Z_gemini_...json), not the file's modified
        # time, because git sets that to the moment you cloned or pulled
        seen = []
        for fp in (self.runs_dir.glob("*.json") if self.runs_dir.exists() else []):
            try:
                seen.append(datetime.strptime(fp.name[:8], "%Y%m%d").date())
            except ValueError:
                continue
        seen += [date.fromisoformat(c["done_at"][:10]) for c in ctx.approvals.all()
                 if c["kind"] == "research-round" and c.get("done_at")]
        last = max(seen) if seen else None
        if last is None or (today - last).days >= self.EVERY_DAYS:
            c = ctx.approvals.ask(self.name, "research-round",
                                  "Run a monthly research round? (Gemini and Groq, 3 rounds, about 12 free API calls)",
                                  f"last round: {last or 'never'}")
            self.say(ctx, "ask", f"Waiting for approval to run a research round (card {c['id']})")
            return Result(OK, f"approval requested ({c['id']})")
        due = last + timedelta(days=self.EVERY_DAYS)
        msg = f"Last research round was on {last}; the next one is due on {due} (no approval needed yet)"
        self.say(ctx, "wait", msg)
        return Result(OK, msg)
