# Roadmap

**Product goal:** an AI research team for the **Indian stock market**. A team of AI
agents discovers, tests and combines stock-picking signals on NSE stocks, a prediction
model ranks stocks with a confidence score, and a high-tech "trading floor" web app
lets people watch the agents work, ask them questions, and see honest results. It
works in English and Hindi, phone first.

**Research goal:** test, honestly, whether LLM-discovered signals are real or just
memory. China's CSI300 (the market used in the original paper) remains our research
benchmark; India is the product market.

| Phase | What | Status |
|---|---|---|
| 1 | China data, backtester, baselines (paper's market) | **Done** |
| 2 | LLMs write factors: 4 models, 191 factors, scored | **Done** |
| 3 | **Indian market data**: NSE pipeline, Nifty baselines, Indian costs and rules | **Built, run on your Mac** |
| 4 | Feedback loop: LLM agents learn from their results each round | next |
| 5 | Prediction model: combined factors, confidence scores, market regimes | |
| 6 | Honest test on held-out years + leakage experiment | |
| 7 | AI agent team: the "trading floor" | |
| 8 | Web app: war room, research lab, track record (English and Hindi) | |
| 9 | Compliance review, deploy, demo video, final report | |

---

## Rules for an Indian product (SEBI)

This shapes the product, so it comes first. *Not legal advice: get a SEBI compliance
professional to review before launching publicly.*

- Giving specific **buy/sell/hold calls on named stocks** to the public is investment
  advice or research. It needs **SEBI registration** as a Research Analyst or
  Investment Adviser. SEBI actively acts against unregistered "finfluencers".
- **From 1 July 2026**, educational content may only use stock price data that is
  **at least 30 days old**, and may not present securities in a way that suggests
  future moves (SEBI circular of 8 May 2026).
- Unregistered people may not make performance or return claims.

So the product has three modes:

| Mode | Who | What it shows | Registration |
|---|---|---|---|
| **Learn** (public) | Everyone | Agents at work, factor lab, backtests, model accuracy, all on data **30+ days old**; no current calls on named stocks | Not needed |
| **Personal** (local) | You, on your own computer | Full live predictions for your own research | Not needed (not published) |
| **Pro** (future) | Subscribers | Live ranked picks with confidence and reasons | **Needs SEBI RA registration, or a registered partner** |

A **Compliance agent** (Phase 7) checks every output: data age in Learn mode,
disclaimers, and no advice-style wording.

---

## What "accurate" means here

No honest model is right about individual stocks most of the time. Funds are happy when
picks beat the market 55–60% of the time, because small steady edges compound. A
backtest showing 80–90% accuracy almost always means look-ahead or leakage, which is
what this project is built to catch. So we measure accuracy in meaningful ways and push
each as high as the honest test allows:

| Metric | What it means | Target (held-out years) |
|---|---|---|
| **Weekly hit rate** | Share of weeks our top picks beat the index | **55–60%+** |
| **High-confidence hit rate** | Same, on the model's confident calls only | **60%+** |
| **Rank IC** | How well the ranking matches what happens | **0.06–0.08** |
| **Return after costs vs index** | Makes money after Indian trading costs? | **+5%/yr or more** |
| **Information ratio** | Excess return per unit of risk | **0.7+** |
| **Calibration** | "70% confident" is right about 70% of the time | within ±5 points |

How we raise accuracy: predict **weekly** instead of daily; **combine many factors** in an
ensemble; **abstain when unsure** (confidence scores); **adapt to the market regime**;
and **feed mistakes back** to the LLMs.

---

## Phase 3: Indian market data (built)

- **Prices:** NSE's official daily bhavcopy archive (every listed stock, including
  ones later delisted, so no survivorship bias), with Yahoo Finance (`.NS` tickers) as
  a fallback. Downloaded on your computer; the code is market-agnostic, so everything
  from Phases 1–2 reruns on India.
- **Universe:** a point-in-time list of the ~200 most liquid NSE stocks each month (by
  traded value), roughly Nifty 200 size, without needing historical index lists.
- **Benchmark:** Nifty 50 and Nifty 200 index levels.
- **Indian rules in the backtester:** price bands (circuit limits 2/5/10/20%), T+1
  settlement, NSE holidays.
- **Indian costs:** about 0.12% to buy and 0.10% to sell for delivery (STT 0.1% each
  way, stamp duty 0.015% on buys, exchange and GST charges), plus ₹15.93 DP charge per
  stock sold.
- **Baselines:** Nifty buy-and-hold, classic factors, LightGBM on Alpha158, the same
  table as Phase 1, for India.
- Re-score all 191 LLM factors on Indian data: do they travel from China to India?

**Needs from you:** run the downloader on your Mac (about 20–40 minutes the first
time).

## Phase 4: Feedback loop

- **Researcher agents** (one per LLM) propose factors; the **Critic agent** turns scores
  into plain feedback ("your momentum ideas worked backwards", "this copies ROC5",
  "this flips daily so costs kill it", "build on these 3").
- Judged **weekly, long-only, after costs**. Several rounds; we track whether each round
  improves.

## Phase 5: Prediction model (accuracy)

- **Ensemble**: LightGBM ranking model on shortlisted LLM factors plus Alpha158, trained
  walk-forward, predicting **5-day** returns.
- **Confidence scores**, calibrated. **Regime detection** (calm, trending, turbulent) with
  regime-aware weights. Trade only when the edge beats the cost.

## Phase 6: Honest test

- Run once on untouched later years, for India and China, with costs and local rules.
- **Leakage experiment**: LLMs trained before vs after the test years (older open models
  via Ollama), plus hidden tickers and dates. **Deflated Sharpe** for the number of ideas
  tried. Compare with the paper's +53.17% claim.

## Phase 7: AI agent team, the "trading floor"

A team of agents, each with one job, working like a research firm. Every action is
logged, so it can be replayed and shown live in the app.

| Desk | Agent | Job |
|---|---|---|
| Head office | **Chief Strategist** | Plans each day, assigns work, asks you to approve big decisions |
| Market data | **Data Collector** | Pulls NSE data after the 3:30 pm IST close |
| | **Data Checker** | Catches missing or odd data before anyone uses it |
| Research lab | **Researchers** (Gemini, gpt-oss, Nemotron, …) | Propose new factors; each LLM sits at its own desk |
| | **Critic** | Reviews results, explains failures, gives feedback |
| Model room | **Model Trainer** | Retrains the ensemble, checks accuracy and calibration |
| | **Regime Watcher** | Labels the market mood; flags regime changes |
| Risk & compliance | **Risk Officer** | Flags crowded, unstable or costly signals |
| | **Compliance Officer** | Enforces SEBI rules: data age, disclaimers, wording |
| Communications | **Explainer** | Plain-English and Hindi reasons for every ranking |
| | **Reporter** | Evening digest after market close |

## Phase 8: Web app

Inspired by "AI office" dashboards: an **isometric trading floor** where you watch the
agents work.

- **War room** (home): an animated isometric office with a desk per agent, live status
  bubbles ("Critic: reviewing 20 factors"), a wall screen with key numbers (Nifty, factors
  tested today, hit rate), a **"Chief Strategist asks"** panel with approve/hold cards, a
  **"What your agents just did"** feed, and an **"Ask the Chief Strategist anything"** chat.
- **Research lab:** type an idea in plain English or Hindi; a Researcher writes the
  formula, tests it, and shows results in seconds.
- **Model arena:** LLMs compared head to head.
- **Stock view:** price chart, past model rankings, reasons (data age depends on mode).
- **Track record:** hit rate, calibration and returns over time, honestly reported.
- **Learn:** how markets and factors work, using lagged data, in English and Hindi.

**Design:** light "office" theme like the reference plus a dark mode; phone-first, since
most Indian investors use phones; ₹ amounts in lakh/crore; IST times.
**Tech:** FastAPI backend with live updates (WebSockets); Next.js + React + Tailwind
frontend; isometric scene as SVG with animated overlays. All on free tiers to start.
**Needs from you:** Node.js (I'll guide), free Vercel account, free backend host.

## Phase 9: Compliance, deploy, present

- Compliance review of Learn mode before going public; decide on the Pro route
  (registration or partner).
- Deploy, final report (method, honest results, India vs China), 3-minute demo video.

---

*Research and education only; not investment advice.*
