"""Fill the result tables in README.md from the files in results/.

The README has blocks like

    <!-- AUTO:phase1_table -->
    ...generated table...
    <!-- /AUTO:phase1_table -->

and this script replaces what's between the markers, so the README always shows
the numbers from your latest run. Run it after run_phase1.py / run_phase2.py
(both call it automatically):

    python src/report.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
README = ROOT / "README.md"


def day(x) -> str:
    t = pd.Timestamp(x)
    return f"{t.day} {t:%b %Y}"  # portable (no %-d, which Windows lacks)


def pct(x, signed=False) -> str:
    if pd.isna(x):
        return ""
    s = f"{x:+.1%}" if signed else f"{x:.1%}"
    return s.replace("-", "−")


def num(x, fmt="{:.3f}") -> str:
    return "" if pd.isna(x) else fmt.format(x).replace("-", "−")


def phase1_table() -> str | None:
    f = RES / "baseline_summary.csv"
    if not f.exists():
        return None
    s = pd.read_csv(f, index_col=0)
    rows = ["| Strategy | Rank IC | ICIR | Gross / yr | Net / yr | Excess vs index / yr | Info ratio | "
            "Max drawdown | Costs / yr | Paper window* |", "|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in s.iterrows():
        bold = name.startswith("Alpha158")
        n = f"**{name}**" if bold else name
        rows.append(f"| {n} | {num(r.get('rank_IC_mean'))} | {num(r.get('rank_ICIR'), '{:.2f}')} | "
                    f"{pct(r.get('gross_ann_return'))} | {pct(r.get('net_ann_return'))} | "
                    f"{pct(r.get('excess_ann_return'), True)} | {num(r.get('info_ratio'), '{:.2f}')} | "
                    f"{pct(r.get('net_max_dd'))} | {pct(r.get('ann_cost_drag'))} | "
                    f"{pct(r.get('paper_window_net_return'), True)} |")
    q = json.loads((ROOT / "data" / "processed" / "data_quality.json").read_text()) \
        if (ROOT / "data" / "processed" / "data_quality.json").exists() else {}
    end = day(q["end"]) if q.get("end") else "the latest data"
    note = (f"\n\nTest period 2 Jan 2017 to {end} (the last year is year-to-date). "
            "\\*Paper window = 1 Jan 2023 to 31 Jan 2024, where the paper reports +53.17%.")
    return "\n".join(rows) + note


def phase2_summary() -> str | None:
    f = RES / "phase2_summary.json"
    if not f.exists():
        return None
    s = json.loads(f.read_text())
    ref = s["best_alpha158_single_feature_rank_ic"]
    d0, d1 = (day(x) for x in s["discovery_period"])
    return (f"- Discovery period: **{d0} to {d1}** "
            f"(2021 onward is held out)\n"
            f"- Factors proposed: **{s['factors_in_library']}** from **{s.get('models', 1)}** model(s); "
            f"valid and scored: **{s['scored']}**\n"
            f"- Strong: **{s['strong']}** · novel: **{s['novel']}** · shortlisted: **{s['shortlist']}**\n"
            f"- Reference: best single Alpha158 feature is {ref['feature']} "
            f"(rank IC {ref['rank_ic']:.3f}, t {ref['t']:.1f}); median feature |rank IC| "
            f"{ref['median_abs_rank_ic']:.3f}")


def phase2_shortlist() -> str | None:
    f = RES / "phase2_factor_scores.csv"
    if not f.exists():
        return None
    s = pd.read_csv(f)
    s = s[s["shortlist"] == True]  # noqa: E712
    if s.empty:
        return "_No factor is on the shortlist yet._"
    s = s.reindex(s["rank_ic"].abs().sort_values(ascending=False).index).head(15)
    rows = ["| Factor | From | Expression | Rank IC | t | Day-to-day stability | Top-fifth excess / yr | "
            "Closest baseline feature |", "|---|---|---|---|---|---|---|---|"]
    for _, r in s.iterrows():
        flip = " (flipped)" if r["direction"] < 0 else ""
        rows.append(f"| `{r['name']}`{flip} | {r['model']} | `{r['expression']}` | {num(r['rank_ic'], '{:+.3f}')} | "
                    f"{num(r['ic_t'], '{:+.1f}')} | {num(r['autocorr'], '{:.2f}')} | "
                    f"{pct(r['top_q_excess'], True)} | {num(r['max_corr_a158'], '{:.2f}')} ({r['closest_a158']}) |")
    return "\n".join(rows)


def phase2_models() -> str | None:
    f = RES / "phase2_model_summary.csv"
    if not f.exists():
        return None
    m = pd.read_csv(f).sort_values("shortlist", ascending=False)
    rows = ["| Model | Proposed | Valid | Invalid | Duplicate | Strong | Shortlisted | Wrong sign | Best factor |",
            "|---|---|---|---|---|---|---|---|---|"]
    for _, r in m.iterrows():
        rows.append(f"| {r['model']} | {r['proposed']} | {r['valid_pct']:.0f}% | {r['invalid_pct']:.0f}% | "
                    f"{r['duplicate_pct']:.0f}% | {r['strong']} | {r['shortlist']} | "
                    f"{num(r['wrong_sign_pct'], '{:.0f}%')} | `{r['best_factor']}` ({num(r['best_rank_ic'], '{:+.3f}')}) |")
    return "\n".join(rows)


BLOCKS = {"phase1_table": phase1_table, "phase2_summary": phase2_summary,
          "phase2_shortlist": phase2_shortlist, "phase2_models": phase2_models}


def main() -> None:
    text = README.read_text()
    for key, fn in BLOCKS.items():
        pat = re.compile(rf"(<!-- AUTO:{key} -->\n).*?(\n<!-- /AUTO:{key} -->)", re.S)
        if not pat.search(text):
            print(f"    README has no AUTO:{key} block, skipping")
            continue
        body = fn()
        if body is None:
            print(f"    {key}: results not found yet, left unchanged")
            continue
        text = pat.sub(lambda m: m.group(1) + body + m.group(2), text)
        print(f"    README: updated {key}")
    README.write_text(text)


if __name__ == "__main__":
    main()
