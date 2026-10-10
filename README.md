# LLM Alpha Discovery: honest evaluation

[![tests](https://github.com/Chaitya2085/LLM-Alpha-Discovery/actions/workflows/tests.yml/badge.svg)](https://github.com/Chaitya2085/LLM-Alpha-Discovery/actions/workflows/tests.yml)

Large language models invent their own stock-picking signals ("alpha factors"), a
filtering system tests them, and the whole thing is checked rigorously enough to
tell genuine discovery from an LLM simply remembering what happened in the market.

It builds on *Automate Strategy Finding with LLM in Quant Investment* (Kou et al.,
EMNLP 2025, [arXiv:2409.06289](https://arxiv.org/abs/2409.06289)), which reports
+53.17% on China's SSE50 from Jan 2023 to Jan 2024. Our version adds:

- **Several LLMs, mostly free** (Gemini, Groq, Cerebras, OpenRouter, Mistral, local
  Ollama, plus paid Claude/OpenAI), compared side by side
- **A safe factor language**: LLMs write formulas, not code, so nothing they write
  can run anything harmful, and no formula can peek at future prices
- **A held-out test period**: factors are chosen on 2014–2020 only; 2021 onward is
  untouched until the final test
- **Realistic trading**: point-in-time index membership, China's price-limit rules,
  and trading costs
- **A leakage test** (Phase 6): do models that were trained *before* the test years
  do as well as models trained *after*?
- **Built for India** (Phase 3 on): NSE stocks, Indian trading costs and rules, and a
  product designed around SEBI's rules for research and investor education
- **An AI agent team and a web app** (Phases 7–8): agents run the research on a live
  "trading floor", with rankings, confidence scores and plain-English/Hindi reasons

| Phase | What | Status |
|---|---|---|
| 1 | China data, backtester, baselines (the paper's market) | **Done** |
| 2 | LLM factor generation (many providers) and scoring | **Done** |
| 3 | Indian market data: NSE pipeline, Nifty baselines, Indian costs and rules | **Built, run on your Mac** |
| 4 | Feedback loop: LLM agents learn from their results | next |
| 5 | Prediction model: combined factors, confidence scores, market regimes | |
| 6 | Honest test on held-out years + leakage experiment | |
| 7 | AI agent team: the "trading floor" | |
| 8 | Web app: war room, research lab, track record (English and Hindi) | |
| 9 | Compliance review, deploy, demo video, final report | |

The full plan, with accuracy targets, SEBI rules and what each phase delivers, is in
**[ROADMAP.md](ROADMAP.md)**.

---

## Quick start

Needs Python 3.10+, about 8 GB RAM and 2 GB free disk.

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python run_phase1.py                   # ~10 min: data, features, baselines, chart, README table
cp .env.example .env                   # then paste at least one free API key into .env
python src/llm.py check                # confirms which keys work
python run_phase2.py --provider gemini groq    # LLMs write factors; scored, charted, README updated

python run_india.py --test             # check NSE downloads work on your network
python run_india.py                    # India: download (30-60 min first time), build, score, compare
```

Every chart in this README is produced by code into `figures/`, and every results
table is filled in by `src/report.py` from the files in `results/`.

## LLM providers

All free options below need no credit card. Free tiers and model names change, so
`python src/llm.py models --provider <name>` lists what your key can use, and any
model can be chosen with `--model` or `<NAME>_MODEL` in `.env`.

| Provider | Free? | Get a key | `.env` variable | Default model |
|---|---|---|---|---|
| `gemini` (Google AI Studio) | Yes | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) | `GEMINI_API_KEY` | `gemini-3.8-flash` |
| `groq` | Yes | [console.groq.com/keys](https://console.groq.com/keys) | `GROQ_API_KEY` | `openai/gpt-oss-120b` |
| `cerebras` | Yes | [cloud.cerebras.ai](https://cloud.cerebras.ai) | `CEREBRAS_API_KEY` | `qwen-3.8-27b` |
| `openrouter` (`:free` models) | Yes, 50 requests/day | [openrouter.ai/keys](https://openrouter.ai/keys) | `OPENROUTER_API_KEY` | picks a currently free model automatically |
| `mistral` | Yes, low limits | [console.mistral.ai](https://console.mistral.ai/api-keys) | `MISTRAL_API_KEY` | `mistral-small-latest` |
| `ollama` (your computer) | Yes, no key | [ollama.com/download](https://ollama.com/download) | none | `qwen3:8b` |
| `anthropic` (Claude) | Paid | [console.anthropic.com](https://console.anthropic.com) | `ANTHROPIC_API_KEY` | `claude-sonnet-5-5` |
| `openai` | Paid | [platform.openai.com](https://platform.openai.com/api-keys) | `OPENAI_API_KEY` | set with `--model` |

Notes: thinking models on free tiers are slow (a batch of 20 factors can take a few
minutes), so the client waits up to 15 minutes per request. On Google's free tier, prompts may be used to improve Google's products
(fine here, since nothing private is sent). Hugging Face no longer includes free
inference credits for free accounts, so it isn't listed; its open models can be run
locally through Ollama instead. Every LLM reply is saved in `llm_runs/`, so any run
can be re-scored later without calling the API again.

---

## Phase 1: data, backtester and baselines

![Growth of 1 since 2017](figures/phase1_growth.png)

Daily prices for every stock that was in China's CSI300 index since 2014 (700
different stocks, kept even after they left the index), a day-by-day trading
simulator, and the strategies professionals already use.

<!-- AUTO:phase1_table -->
| Strategy | Rank IC | ICIR | Gross / yr | Net / yr | Excess vs index / yr | Info ratio | Max drawdown | Costs / yr | Paper window* |
|---|---|---|---|---|---|---|---|---|---|
| Reversal 5d | 0.021 | 0.10 | 1.1% | −3.3% | −5.5% | −0.41 | −57.4% | 4.4% | −36.2% |
| Momentum 60d | −0.001 | −0.01 | 4.0% | 0.6% | −1.6% | −0.03 | −59.2% | 3.2% | −8.7% |
| Low volatility 20d | 0.018 | 0.08 | 4.3% | 0.3% | −3.8% | −0.27 | −32.9% | 3.9% | −1.5% |
| Price-volume corr 20d (neg) | 0.015 | 0.09 | 8.4% | 3.7% | +0.7% | 0.12 | −48.9% | 4.4% | −25.8% |
| **Alpha158 + LightGBM** | 0.045 | 0.29 | 9.0% | 3.7% | +0.7% | 0.13 | −33.9% | 4.9% | −21.1% |
| CSI300 index (buy & hold) |  |  |  | 2.7% | +0.0% |  | −46.7% |  | −17.1% |
| Equal-weight CSI300 (gross) |  |  |  | 2.9% | +0.1% |  | −35.0% |  | −15.6% |

Test period 2 Jan 2017 to 9 Oct 2026 (the last year is year-to-date). \*Paper window = 1 Jan 2023 to 31 Jan 2024, where the paper reports +53.17%.
<!-- /AUTO:phase1_table -->

**What we learned**

1. **The machine-learning baseline genuinely predicts returns.** LightGBM on Qlib's
   158 handcrafted factors reaches a rank IC of about 0.045, in line with Qlib's own
   published numbers, which also confirms the data and pipeline are right.
2. **Trading costs eat that edge.** The signal lasts about a day, so the model has to
   trade constantly. Before costs it earns roughly three times the index; after
   costs it is about level with it. Refreshing all 50 stocks daily would earn about
   21% a year gross but cost about 32% a year. **A useful signal has to survive
   costs, not just predict.**
3. **The paper's test year was a losing year for everyone.** Over Jan 2023 to Jan
   2024 the CSI300 fell about 17% and every baseline lost money, which makes a +53%
   result a red flag worth testing (Phase 6).
4. **Classic single factors don't beat the index** long-only after costs.

**Checks:** index returns match the index's published levels; the backtester
matches an independent vectorised calculation (21.3% vs 21.4% gross); a random
signal shows no edge; the regression features match `numpy.polyfit` exactly.

---

## Phase 2: LLMs write factors

![Strength vs novelty](figures/phase2_strength_vs_novelty.png)

Each LLM gets the same prompt (`src/prompts.py`): the factor language, how factors
are used, and the baseline families to avoid copying. The prompt never names the
market, country, tickers or dates. Replies are parsed, validated, de-duplicated
and scored on **Jul 2014 to Dec 2020 only**.

<!-- AUTO:phase2_summary -->
- Discovery period: **1 Jul 2014 to 28 Dec 2020** (2021 onward is held out)
- Factors proposed: **251** from **4** model(s); valid and scored: **245**
- Strong: **102** · novel: **140** · shortlisted: **28**
- Reference: best single Alpha158 feature is ROC5 (rank IC 0.043, t 8.7); median feature |rank IC| 0.018
<!-- /AUTO:phase2_summary -->

**Which LLM writes the most useful factors?**

![Model comparison](figures/phase2_models.png)

<!-- AUTO:phase2_models -->
| Model | Proposed | Valid | Invalid | Duplicate | Strong | Shortlisted | Wrong sign | Best factor |
|---|---|---|---|---|---|---|---|---|
| groq: openai/gpt-oss-120b | 91 | 95% | 0% | 5% | 32 | 10 | 73% | `vol_adi_imbalance` (−0.035) |
| openrouter: nvidia/nemotron-3-ultra-550b-a55b:free | 60 | 100% | 0% | 0% | 26 | 7 | 78% | `rev_short_term` (+0.042) |
| seed: claude (chat) | 40 | 100% | 0% | 0% | 21 | 6 | 28% | `vwap_stretch_reversal` (+0.046) |
| gemini: gemini-3.8-flash | 60 | 98% | 0% | 2% | 23 | 5 | 46% | `range_expansion_momentum` (−0.040) |
<!-- /AUTO:phase2_models -->

**Shortlist** (strong: |t| ≥ 3, |rank IC| ≥ 0.015, same sign in ≥ 70% of years;
novel: correlation < 0.7 with every Alpha158 feature; unique: correlation < 0.7 with
any stronger LLM factor). "Flipped" means the factor works in the opposite
direction to the LLM's hypothesis.

<!-- AUTO:phase2_shortlist -->
| Factor | From | Expression | Rank IC | t | Day-to-day stability | Top-fifth excess / yr | Closest baseline feature |
|---|---|---|---|---|---|---|---|
| `vwap_stretch_reversal` | seed: claude (chat) | `-Decay(close / vwap - 1, 10)` | +0.046 | +10.9 | 0.82 | +19.1% | 0.49 (RSV10) |
| `rev_intraday_vwap` | openrouter: nvidia/nemotron-3-ultra-550b-a55b:free | `-EMA((close - vwap) / (high - low), 5) * CsRank(volume)` | +0.040 | +10.4 | 0.56 | +17.2% | 0.59 (VWAP0) |
| `rev_sign_range` | openrouter: nvidia/nemotron-3-ultra-550b-a55b:free | `-Sign(returns) * (close - TsMin(low, 10)) / (TsMax(high, 10) - TsMin(low, 10))` | +0.033 | +8.6 | −0.05 | +12.0% | 0.56 (RANK5) |
| `rev_short_term_overshoot` | groq: openai/gpt-oss-120b | `-Mean(Sign(Delta(close, 1)) * Abs(Delta(close, 1) - Mean(Delta(close, 1), 3)), 5)` | +0.026 | +7.6 | 0.47 | +7.8% | 0.56 (MA5) |
| `volume_return_covariance` (flipped) | gemini: gemini-3.8-flash | `EMA(Cov(returns, volume / Mean(volume, 20), 20), 10)` | −0.024 | −6.3 | 0.99 | +6.6% | 0.66 (CORD30) |
| `volume_stability` | seed: claude (chat) | `-Std(Log(volume), 40)` | +0.024 | +6.2 | 0.99 | +9.9% | 0.46 (WVMA60) |
| `corr_close_vol_sign` (flipped) | groq: openai/gpt-oss-120b | `Corr(close, volume, 20) * Sign(returns)` | −0.024 | −6.9 | −0.05 | +0.7% | 0.43 (RANK5) |
| `low_idiosyncratic_vol` | seed: claude (chat) | `-Std(CsDemean(returns), 40)` | +0.023 | +4.8 | 0.99 | +3.4% | 0.59 (STD30) |
| `intraday_close_position` (flipped) | openrouter: nvidia/nemotron-3-ultra-550b-a55b:free | `(close - low) / (high - low) * CsRank(volume)` | −0.022 | −6.0 | 0.52 | −4.8% | 0.58 (KSFT2) |
| `volume_spike_fade` | seed: claude (chat) | `-TsMax(volume, 20) / Mean(volume, 60)` | +0.022 | +6.1 | 0.95 | +5.7% | 0.46 (WVMA20) |
| `vol_cv` (flipped) | openrouter: nvidia/nemotron-3-ultra-550b-a55b:free | `Std(volume, 20) / Mean(volume, 20) * CsRank(volume)` | −0.022 | −5.9 | 0.95 | +2.9% | 0.29 (CORR20) |
| `vol_persistence` (flipped) | openrouter: nvidia/nemotron-3-ultra-550b-a55b:free | `Corr(volume, Ref(volume, 1), 60)` | −0.020 | −5.1 | 0.98 | +14.4% | 0.49 (CORR60) |
| `intraday_gap_momentum` | groq: openai/gpt-oss-120b | `(Ref(open, 1) - Ref(close, 1)) / Ref(close, 1) * TsRank(volume, 10)` | +0.020 | +5.1 | 0.00 | +7.3% | 0.43 (BETA5) |
| `rev_delta_volume_cross` (flipped) | groq: openai/gpt-oss-120b | `Delta(close, 1) * (1 - CsRank(volume))` | −0.020 | −5.7 | −0.04 | −3.7% | 0.66 (KMID) |
| `mom_long_term_corr_ret_vol` (flipped) | groq: openai/gpt-oss-120b | `Corr(returns, volume, 60) * TsRank(close, 40)` | −0.019 | −4.6 | 0.94 | +1.4% | 0.65 (RANK60) |
<!-- /AUTO:phase2_shortlist -->

**What we learned from the first batch.** The first 40 factors (`llm_runs/seed/`)
were written by Claude in the chat session where this project was designed,
answering the exact prompt above before any factor was scored. They are kept as a
labelled baseline; `--no-seed` leaves them out.

1. **One genuinely new, strong signal:** `vwap_stretch_reversal` (stocks that keep
   closing above the day's average traded price tend to fall back) beat every
   baseline feature, held in all 7 discovery years, and overlaps at most 0.49 with
   the 158 baseline features. Its t-stat of about 11 is far beyond what 40 tries
   could produce by luck.
2. **The LLM's market intuition was often wrong-signed:** 7 of 8 momentum ideas
   worked backwards, because China large caps tend to reverse rather than trend.
3. **Most strong ideas were restatements** of existing reversal features
   (correlation 0.87–0.99); without the novelty check they would look like
   discoveries.
4. **Predictive is not the same as useful long-only:** some factors only spot
   losers, which a fund that only buys can't use. Phase 3 filters on top-fifth
   performance after costs.

**Checks:** 19 automated tests (unsafe code rejected; operators match pandas; changing
future prices never changes past factor values; the LLM client handles every
provider format, rate limits, bad keys and messy replies); the top factor's IC was
recomputed independently with scipy (identical); a time-shuffled placebo gives
an IC of about zero.

---

## Phase 3: India (NSE)

![India: model vs Nifty 200](figures/india_phase1_growth.png)

The same pipeline, now on the Indian market. Everything is built from NSE's official
daily files, downloaded on your computer by `python run_india.py`.

- **Data:** NSE's daily equity bhavcopy (every listed stock, including later-delisted
  ones) and NSE's daily index closes. Both of NSE's file formats are handled (it changed
  format on 8 July 2024).
- **Splits and bonuses:** on an ex-date NSE adjusts the stock's "previous close", so the
  adjustment is measured from the data itself; no separate corporate-actions feed is
  needed. Closes from the trade-to-trade (BE) series are used for continuity, so a stock
  moving between series isn't mistaken for a split.
- **Universe:** each month, the 200 most-traded EQ stocks over the previous three months
  (point-in-time, no survivorship bias). **Benchmark:** Nifty 200.
- **Indian trading:** 0.12% to buy and 0.13% to sell (STT, stamp duty, exchange fees,
  DP charge); a stock locked at its circuit limit can't be traded at the open.

<!-- AUTO:india_data -->
- NSE data **1 Jan 2014 to 9 Oct 2026**: 3,149 trading days; universe from 1 Apr 2014
- **674** different stocks passed through the top-200 universe (median 200 members a day)
- **5236** split/bonus/rights adjustments detected from NSE's previous-close field
- Benchmark: **Nifty 200**, 3,163 → 13,063
- Locked-circuit days among members: 0.027% up, 0.039% down
<!-- /AUTO:india_data -->

<!-- AUTO:india_phase1_table -->
| Strategy | Rank IC | ICIR | Gross / yr | Net / yr | Excess vs index / yr | Info ratio | Max drawdown | Costs / yr | Paper window* |
|---|---|---|---|---|---|---|---|---|---|
| Reversal 5d | 0.019 | 0.13 | 6.0% | 4.7% | −6.3% | −0.52 | −64.5% | 1.3% | +28.3% |
| Momentum 60d | 0.000 | 0.00 | 10.2% | 9.2% | −2.6% | −0.24 | −48.4% | 0.9% | +31.0% |
| Low volatility 20d | 0.019 | 0.11 | 11.3% | 6.9% | −5.1% | −0.72 | −41.1% | 4.0% | +25.4% |
| Price-volume corr 20d (neg) | 0.017 | 0.15 | 8.8% | 7.2% | −4.4% | −0.51 | −50.6% | 1.5% | +23.1% |
| **Alpha158 + LightGBM** | 0.056 | 0.45 | 14.0% | 10.0% | −1.9% | −0.26 | −48.2% | 3.6% | +24.0% |
| Nifty 200 index (buy & hold) |  |  |  | 12.0% | +0.0% |  | −37.9% |  | +25.1% |
| Equal-weight top-200 NSE (gross) |  |  |  | 10.2% | −1.2% |  | −55.3% |  | +46.5% |

Test period 2 Jan 2017 to 9 Oct 2026 (the last year is year-to-date). \*Paper window = 1 Jan 2023 to 31 Jan 2024, the paper's China test period, shown for comparison.
<!-- /AUTO:india_phase1_table -->

**Do the LLM factors work in India too?** Each factor is scored on India's 2014–2020
data exactly as it was on China's.

![China vs India](figures/india_transfer.png)

<!-- AUTO:india_transfer -->
- **245** LLM factors scored in both markets on 2014–2020; correlation of their scores across markets **+0.68** (rank correlation +0.67)
- Strong in China (|t| ≥ 3): **131**; of these, same direction in India: **110**, strong in India too: **76**
- Strong in India: **126**; strong in both, same direction: **76**

| Factor | Rank IC China | Rank IC India |
|---|---|---|
| `vol_adi_imbalance` | −0.035 | −0.026 |
| `rev_sign_range` | +0.033 | +0.035 |
| `rev_short_term_overshoot` | +0.026 | +0.022 |
| `cross_sectional_shock_reversal` | +0.038 | +0.037 |
| `normalized_short_term_return_reversal` | +0.035 | +0.025 |
<!-- /AUTO:india_transfer -->

![India strength vs novelty](figures/india_phase2_strength_vs_novelty.png)

<!-- AUTO:india_phase2_summary -->
- Discovery period: **1 Jul 2014 to 28 Dec 2020** (2021 onward is held out)
- Factors proposed: **251** from **4** model(s); valid and scored: **245**
- Strong: **77** · novel: **142** · shortlisted: **25**
- Reference: best single Alpha158 feature is KLEN (rank IC -0.044, t -11.1); median feature |rank IC| 0.015
<!-- /AUTO:india_phase2_summary -->

<!-- AUTO:india_phase2_shortlist -->
| Factor | From | Expression | Rank IC | t | Day-to-day stability | Top-fifth excess / yr | Closest baseline feature |
|---|---|---|---|---|---|---|---|
| `low_parkinson_vol` | seed: claude (chat) | `-Mean(Log(high / low), 20)` | +0.037 | +7.3 | 0.99 | +3.6% | 0.66 (STD20) |
| `rev_sign_range` | openrouter: nvidia/nemotron-3-ultra-550b-a55b:free | `-Sign(returns) * (close - TsMin(low, 10)) / (TsMax(high, 10) - TsMin(low, 10))` | +0.035 | +14.2 | −0.02 | +10.6% | 0.59 (OPEN0) |
| `vol_surge` (flipped) | openrouter: nvidia/nemotron-3-ultra-550b-a55b:free | `volume / Mean(volume, 20) * CsRank(volume)` | −0.030 | −8.9 | 0.69 | +13.2% | 0.70 (VMA20) |
| `gap_volatility` | seed: claude (chat) | `-Std(open / Ref(close, 1) - 1, 20)` | +0.029 | +7.5 | 0.96 | +6.8% | 0.47 (STD20) |
| `lottery_avoidance` | seed: claude (chat) | `-TsMax(returns, 20)` | +0.028 | +7.5 | 0.95 | +6.2% | 0.56 (MIN20) |
| `volume_weighted_volatility_ratio` | gemini: gemini-3.8-flash | `-Std(returns * volume / Mean(volume, 20), 30)` | +0.027 | +7.7 | 0.97 | +5.2% | 0.69 (WVMA30) |
| `low_beta` | seed: claude (chat) | `-Corr(returns, returns - CsDemean(returns), 60) * Std(returns, 60)` | +0.026 | +4.3 | 0.99 | +2.2% | 0.47 (STD60) |
| `rev_delta_volume_cross` (flipped) | groq: openai/gpt-oss-120b | `Delta(close, 1) * (1 - CsRank(volume))` | −0.024 | −9.3 | −0.01 | +9.5% | 0.68 (KMID2) |
| `amihud_illiquidity` (flipped) | seed: claude (chat) | `CsRank(Mean(Abs(returns) / (volume * vwap), 20))` | −0.023 | −7.2 | 1.00 | −4.3% | 0.28 (KLEN) |
| `rev_short_term_overshoot` | groq: openai/gpt-oss-120b | `-Mean(Sign(Delta(close, 1)) * Abs(Delta(close, 1) - Mean(Delta(close, 1), 3)), 5)` | +0.022 | +9.1 | 0.47 | +10.7% | 0.58 (SUMD5) |
<!-- /AUTO:india_phase2_shortlist -->

**Checks:** 5 India tests on a synthetic NSE archive (both file formats, splits,
holidays, listings and delistings, locked circuits, blocked downloads, resume). The
whole India pipeline was also run on 11 years of synthetic **random-walk** prices: every
factor and the LightGBM model showed no edge (largest |t| 1.3, rank IC ≈ 0), which is
the right answer and shows nothing in the pipeline peeks at the future.

*Indian data is used here for research. Under SEBI's rules, public educational use needs
price data at least 30 days old, and specific stock recommendations need SEBI
registration (see [ROADMAP.md](ROADMAP.md)).*

---

## Design choices that keep results honest

- **Point-in-time universe:** a stock is eligible only on days it was actually in the
  CSI300, so there's no survivorship bias.
- **No peeking:** signal at the close of day *t*, trade at the open of *t+1*, earn
  open *t+1* → open *t+2*. The factor language only looks backwards (tested).
- **China market rules:** no buying a stock that opens limit-up, no selling one that
  opens limit-down, no trading suspended stocks.
- **Costs:** 0.05% on buys, 0.15% on sells (Qlib's China defaults).
- **Walk-forward training** for the LightGBM baseline, retrained every January.
- **Held-out period:** factor selection uses 2014–2020 only.
- **Market-blind prompts:** no market, dates or tickers, so the Phase 5 leakage test
  compares like with like.

## Data

**India:** NSE's official daily bhavcopy and index-close files, downloaded by
`src/india_download.py` (resumable; about 6,000 small files from 2014). Not committed to git.

**China:** daily A-share data in Qlib format from the community-maintained
[chenditc/investment_data](https://github.com/chenditc/investment_data).
`run_phase1.py` downloads it (about 570 MB) and it is not committed to git. It's a free
community dataset, not a vendor feed: suspension days show as missing prices, ST
stocks' 5% limit isn't modelled, and new trading days are added daily, so results
from a later download differ slightly in the latest year.

## Project layout

```
ROADMAP.md               the plan for Phases 3-9
run_phase1.py            data -> features -> baselines -> chart -> README table
run_phase2.py            tests -> LLM factors -> scoring -> charts -> README tables
run_india.py             NSE download -> India dataset -> baselines -> factor scores -> China vs India
src/
  markets.py             per-market settings (paths, costs, rules); --market china|india
  india_download.py      download NSE bhavcopies and index closes (resumable, polite)
  india_parse.py         read both NSE bhavcopy formats and index files
  build_india.py         India dataset: split/bonus adjustment, top-200 universe, benchmark
  compare_markets.py     do LLM factors work in both China and India?
  qlib_reader.py         read Qlib .bin files without installing Qlib
  build_dataset.py       CSI300 point-in-time panel, tradability flags, quality checks
  data.py                wide (date x stock) matrices and next-day returns
  features.py            Alpha158 baseline factor set
  build_features.py      compute and save the Alpha158 matrix
  backtest.py            Top-K/Dropout backtester with limits, suspensions, costs
  metrics.py             IC, ICIR, returns, Sharpe, drawdown, information ratio
  run_baselines.py       Phase 1 experiments
  dsl.py                 the factor language: safe parser, operators, evaluator
  prompts.py             LLM prompts (market-blind)
  llm.py                 client for all providers; `check` and `models` commands
  generate_factors.py    ask an LLM, validate, de-duplicate, save to the library
  score_factors.py       score on 2014-2020: IC, stability, novelty, per-model summary
  style.py               shared chart style
  plot_phase1.py         figures/phase1_growth.png
  plot_phase2.py         figures/phase2_strength_vs_novelty.png, figures/phase2_models.png
  report.py              fills the README tables from results/
tests/                   test_dsl.py (factor language), test_llm.py (LLM client),
                         test_india.py + fake_nse.py (Indian pipeline on a synthetic NSE archive)
factors/library.jsonl    every proposed factor: valid, invalid (with reason), duplicate
llm_runs/                every raw LLM reply (prompt, reply, model, tokens)
results/                 CSV/JSON/parquet outputs
figures/                 all charts
```

## Tests

```bash
python tests/test_dsl.py
python tests/test_llm.py
python tests/test_india.py
```

They need no data and no API keys, and run automatically on GitHub for every push
(`.github/workflows/tests.yml`).

## Authors

Chaitya Nanavati, with Claude (Anthropic) as AI collaborator.

## Disclaimer

Research and education only; not investment advice.

## License

MIT, see [LICENSE](LICENSE).
