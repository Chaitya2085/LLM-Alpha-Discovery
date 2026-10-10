"""Phase 1 baselines. Everything later (LLM-generated factors) must beat these.

Test period 2017-01-01 -> latest. 2014-2016 is warm-up / first training data.

  1. Benchmark index (buy & hold): CSI300 for China, Nifty 200 for India
  2. Equal-weight universe (gross, reference only)
  3. Classic single factors via Top-50/Dropout-5: 5d reversal, 60d momentum,
     low volatility, price-volume correlation
  4. Alpha158 + LightGBM, walk-forward: retrained every year on all prior
     data, previous year held out for early stopping. This is Qlib's standard
     benchmark model and the strongest 'human-designed' baseline.

The paper's test window (2023-01-01 -> 2024-01-31) is reported separately.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from backtest import topk_dropout
from data import load_market
from metrics import backtest_summary, ic_series, ic_summary, perf, yearly

import markets

ROOT = Path(__file__).resolve().parents[1]
CFG = markets.from_argv()
RES = CFG.results
TEST_START, PAPER = "2017-01-01", ("2023-01-01", "2024-01-31")
GAP = 3  # trading days dropped at train/valid/test boundaries (label spans t+1..t+2)

LGB_PARAMS = dict(  # Qlib's published Alpha158 LightGBM settings, lr raised for speed
    objective="regression", learning_rate=0.1, colsample_bytree=0.8879,
    subsample=0.8789, subsample_freq=1, lambda_l1=205.6999, lambda_l2=580.9768,
    max_depth=8, num_leaves=210, min_data_in_leaf=200, num_threads=2, verbose=-1, seed=7,
)


def cs_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile rank per date, centred to [-0.5, 0.5]."""
    return df.groupby(level="date").rank(pct=True) - 0.5


def lgb_walk_forward(X: pd.DataFrame, y: pd.Series, dates: pd.DatetimeIndex) -> tuple[pd.Series, dict]:
    Xr = cs_rank(X).astype("float32")
    yr = cs_rank(y.to_frame())["fwd"].astype("float32")
    d = X.index.get_level_values("date")
    preds, log = [], {}
    for year in range(pd.Timestamp(TEST_START).year, dates[-1].year + 1):
        t0 = pd.Timestamp(f"{year}-01-01")
        if dates[0] > pd.Timestamp(f"{year - 2}-03-01"):
            continue  # not enough history before this year to train and validate
        v0 = pd.Timestamp(f"{year - 1}-01-01")
        tdays = dates[dates < v0]
        vdays = dates[(dates >= v0) & (dates < t0)]
        tr = (d >= "2014-04-01") & (d <= tdays[-1 - GAP]) & yr.notna().values
        va = (d >= vdays[0]) & (d <= vdays[-1 - GAP]) & yr.notna().values
        te = (d >= t0) & (d < pd.Timestamp(f"{year + 1}-01-01"))
        if tr.sum() < 20000 or va.sum() < 5000 or te.sum() == 0:
            continue
        t = time.time()
        model = lgb.train(
            LGB_PARAMS, lgb.Dataset(Xr[tr], yr[tr]), num_boost_round=1000,
            valid_sets=[lgb.Dataset(Xr[va], yr[va])],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        preds.append(pd.Series(model.predict(Xr[te], num_iteration=model.best_iteration),
                               index=X.index[te]))
        log[year] = {"train_rows": int(tr.sum()), "best_iter": model.best_iteration,
                     "secs": round(time.time() - t, 1)}
        print(f"  LGB {year}: {log[year]}", flush=True)
    return pd.concat(preds), log


def main() -> None:
    RES.mkdir(parents=True, exist_ok=True)
    m = load_market()
    X = pd.read_parquet(CFG.proc / "alpha158.parquet")
    fwd_long = m.fwd_ret.stack().rename("fwd")
    y = fwd_long.reindex(X.index)

    def wide(name):
        return X[name].unstack("code").reindex(index=m.dates, columns=m.member.columns).astype("float64")

    scores = {
        "Reversal 5d": wide("ROC5"),                 # past losers ranked high
        "Momentum 60d": 1.0 / wide("ROC60"),         # past winners ranked high
        "Low volatility 20d": -wide("STD20"),
        "Price-volume corr 20d (neg)": -wide("CORR20"),
    }
    print("Training walk-forward LightGBM on Alpha158 ...", flush=True)
    lgb_pred, lgb_log = lgb_walk_forward(X, y, m.dates)
    scores["Alpha158 + LightGBM"] = lgb_pred.unstack("code").reindex(
        index=m.dates, columns=m.member.columns)
    lgb_pred.to_frame("score").to_parquet(RES / "lgb_alpha158_scores.parquet")

    rows, yearly_tbl, dailies = [], {}, {}
    for name, s in scores.items():
        s_test = s.loc[TEST_START:]
        ic = ic_series(s_test, m.fwd_ret, m.member, "pearson")
        ric = ic_series(s_test, m.fwd_ret, m.member, "spearman")
        bt = topk_dropout(s, m, topk=50, n_drop=5, start=TEST_START).daily
        pw = bt.loc[PAPER[0]:PAPER[1]]
        row = {"strategy": name, **{f"rank_{k}": v for k, v in ic_summary(ric).items()},
               "IC_mean": ic.mean(), **backtest_summary(bt),
               "paper_window_net_return": (1 + pw["net"]).prod() - 1}
        rows.append(row)
        yearly_tbl[name] = yearly(bt)
        dailies[name] = bt["net"]
        print(f"  {name}: excess {row['excess_ann_return']:.2%}  IR {row['info_ratio']:.2f}  "
              f"RankIC {row['rank_IC_mean']:.4f}", flush=True)

    bench = bt["bench"]
    ew = m.fwd_ret.where(m.member).mean(axis=1).loc[bench.index]
    for name, r in [(f"{m.bench_label} index (buy & hold)", bench),
                    (f"Equal-weight {m.universe_label} (gross)", ew)]:
        p = perf(r)
        pw = r.loc[PAPER[0]:PAPER[1]]
        rows.append({"strategy": name, "net_ann_return": p["ann_return"], "net_sharpe": p["sharpe"],
                     "net_max_dd": p["max_drawdown"], "bench_ann_return": perf(bench)["ann_return"],
                     "excess_ann_return": perf(r - bench)["ann_return"],
                     "paper_window_net_return": (1 + pw).prod() - 1})
        yearly_tbl[name] = r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1)
        dailies[name] = r

    summary = pd.DataFrame(rows).set_index("strategy")
    summary.to_csv(RES / "baseline_summary.csv")
    pd.DataFrame(yearly_tbl).to_csv(RES / "baseline_yearly_returns.csv")
    pd.DataFrame(dailies).to_parquet(RES / "baseline_daily_returns.parquet")
    (RES / "lgb_training_log.json").write_text(json.dumps(lgb_log, indent=2))
    pd.set_option("display.width", 200)
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
