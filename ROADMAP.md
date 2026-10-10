# Roadmap

Where the project is going, phase by phase: what gets built, how we know it worked,
and what each phase needs from us.

**End goal:** a system where AI agents discover, test and combine stock-picking
signals, a prediction model ranks stocks with a confidence score, and a high-tech web
app shows it all live: today's predictions, why the model made them, how accurate it
has been, and the agents at work.

| Phase | What | Status |
|---|---|---|
| 1 | Data, backtester, human-designed baselines | **Done** |
| 2 | LLMs write factors; scored on 2014–2020 | **Done** |
| 3 | Feedback loop: LLM agents learn from their results | **Next** |
| 4 | Prediction model: combine factors, confidence scores, market regimes | |
| 5 | Honest test on 2021–2026 + leakage experiment | |
| 6 | Multi-agent system that runs the whole pipeline | |
| 7 | High-tech web app (backend API + dashboard) | |
| 8 | Deploy, demo video, final report | |

---

## What "accurate" means here

Stock returns are mostly noise, so no honest model is right about individual stocks
"most of the time". Professional funds are happy when their picks beat the market
55–60% of the time, because small, consistent edges compound. A backtest showing
80–90% accuracy is almost always a sign of a mistake such as look-ahead or leakage,
which is exactly what this project is built to catch.

So we measure accuracy in ways that are meaningful and achievable, and we push each
one as high as the honest test allows:

| Metric | What it means | Today | Target (on held-out 2021–2026) |
|---|---|---|---|
| **Weekly hit rate** | Share of weeks our top 50 beat the index | measured in Phase 4 | **55–60%+** |
| **High-confidence hit rate** | Same, but only on weeks/stocks the model is confident about | measured in Phase 4 | **60%+** on fewer, stronger calls |
| **Rank IC** | How well the ranking matches what actually happens | 0.045 (baseline model) | **0.06–0.08** |
| **Return after costs vs index** | Does it make money after trading costs? | ≈ +0.7%/yr (baseline) | **+5%/yr or more** |
| **Information ratio** | Excess return per unit of risk | 0.13 (baseline) | **0.7+** |
| **Calibration** | When it says "70% confident", is it right ~70% of the time? | n/a | within ±5 points |

The biggest accuracy gains in this kind of work come from five things, and Phases 3–5
use all of them:

1. **Predict weekly, not daily.** Day-to-day moves are close to random and cost a lot
   to trade; 5-day returns are more predictable and cheaper to act on.
2. **Combine many weak signals.** One factor is weak; dozens of different ones,
   combined by a model, are much stronger (an ensemble).
3. **Abstain when unsure.** A confidence score lets the system act only on its
   strongest calls, which raises the hit rate on the calls it does make.
4. **Adapt to the market regime.** Signals that work in calm markets often fail in
   crashes; detecting the regime and re-weighting helps.
5. **Learn from mistakes.** The feedback loop tells the LLMs what failed (for example
   wrong-signed momentum ideas), so later factors are better.

---

## Phase 3: Feedback loop (next)

LLMs stop guessing blind. After each round, every model gets its results back and
writes the next batch with that knowledge.

- **Generator agent** (one per LLM): proposes factors, as in Phase 2.
- **Critic agent**: turns scores into plain feedback, e.g. "your 5 momentum ideas
  worked backwards: this market reverts", "`X` duplicates baseline feature ROC5",
  "`Y` flips every day, so trading costs kill it", "these 3 worked: build on them".
- **Stricter judging**: factors must hold up **long-only, after costs, at a weekly
  horizon**, not just predict on average.
- Several rounds per model; we track whether each round's factors get better.

**Done when:** each model's later rounds beat its first round on the shortlist rate,
and the run is reproducible from `llm_runs/`.
**Needs from us:** the same free API keys; about 30–60 minutes of runs.

## Phase 4: Prediction model (accuracy)

Turn the best factors into one prediction per stock, with a confidence score.

- **Ensemble model**: LightGBM ranking model on the shortlisted LLM factors plus
  Alpha158, trained walk-forward (only on the past), predicting **5-day** returns.
- **Confidence score**: from agreement between models and how far a stock's score is
  from the crowd; calibrated so "70%" means 70%.
- **Regime detection**: a simple, explainable model labels each period calm,
  trending or turbulent; factor weights adapt to the regime.
- **Portfolio rules**: hold the top stocks, trade only when the edge beats the cost.
- Everything still chosen on 2014–2020 only.

**Done when:** on 2014–2020 cross-validation, the ensemble clearly beats the best
single factor and the Phase 1 baseline on every metric in the table above.

## Phase 5: Honest test

The moment of truth, run once on the untouched **2021–2026** data.

- Walk-forward results with costs, China trading rules, and every metric above.
- **Leakage experiment**: the same pipeline with LLMs whose training data ends
  *before* the test years vs *after* (older open models via Ollama), and with tickers
  and dates hidden. If results collapse, the "alpha" was memory.
- **Deflated Sharpe ratio**: corrects for how many ideas we tried.
- Comparison with the paper's claimed +53.17% on its test window.

**Done when:** we can state, with evidence, how accurate the system really is.
**Needs from us:** Ollama installed, plus one or two older open models (I'll list
which).

## Phase 6: Multi-agent system

The pipeline becomes a team of AI agents that work together, each with one job and
tools to do it:

| Agent | Job | Tools |
|---|---|---|
| **Orchestrator** | Plans the run, assigns work, decides when to stop | all agents |
| **Researcher** | Proposes new factor ideas (several LLMs) | factor language, library |
| **Critic** | Reviews results, explains failures, gives feedback | scorer, backtester |
| **Risk manager** | Flags crowded, unstable or costly signals; checks regime | regime model, cost model |
| **Portfolio agent** | Builds today's ranked list with confidence | prediction model |
| **Explainer** | Writes plain-English reasons for each prediction | factor values, LLM |

Built as small, testable Python classes with a shared message log, so every agent
decision is saved and can be replayed (and shown live in the UI). We may adopt an agent
framework such as LangGraph if it makes this simpler; the decision is made when we get
there.

**Done when:** one command runs a full research round end to end with agents, and
every step is logged.

## Phase 7: High-tech web app

A fast, modern dashboard, dark-themed, with live updates and smooth animation.

**Screens**
1. **Command centre**: today's top-ranked stocks with confidence, market regime,
   recent accuracy, and a live feed of agent activity.
2. **Stock view**: price chart with the model's past predictions overlaid, and the
   Explainer's reasons ("ranked high because…").
3. **Factor lab**: type an idea in plain English ("stocks recovering on rising
   volume"); the Researcher agent writes the formula, tests it, and shows the
   results in seconds.
4. **Agent console**: watch agents propose, critique and refine factors in real time.
5. **Model arena**: compare LLMs head to head (the Phase 2 comparison, live).
6. **Track record**: hit rate, calibration and returns over time, honestly reported.

**Tech** (decided when we start; all have free tiers)
- Backend: **FastAPI** (Python) wrapping the existing code, with live updates over
  WebSockets.
- Frontend: **Next.js + React + Tailwind**, interactive financial charts, smooth
  animations.
- Daily refresh: a scheduled job pulls the newest data and updates predictions.

**Done when:** the app runs locally with one command and shows real predictions.
**Needs from us:** Node.js installed (I'll guide), a free Vercel account (frontend) and
a free backend host (Render, Railway or Hugging Face Spaces; chosen in Phase 8).

## Phase 8: Deploy and present

- Put the app online (free tiers), with the API keys stored safely on the hosts.
- Final report: method, honest results, what worked, what didn't.
- 3-minute demo video and a polished README with screenshots.

---

*Research and education only; not financial advice.*
