import { usePoll, pct, signed } from "./api.js";
import LineChart from "./LineChart.jsx";

const VERDICT = {
  "USEFUL WARNING": ["Would have warned", "good"],
  "WEAK HINT": ["A weak hint, not proven", "warn"],
  "NO WARNING VALUE": ["Did not warn reliably", "bad"],
  "NOT ENOUGH WEAK WEEKS": ["Not enough weak weeks to test", "bad"],
};
const f2 = x => (x == null ? "–" : signed(x, 2));

function Period({ label, t }) {
  const z = t.zones, d = t.weak_vs_rest_next_week;
  return (
    <tr>
      <td>{label}</td>
      {["weak", "middle", "strong"].map(k => (
        <td key={k} className="r num">{f2(z[k].next_week)}<br /><span className="small muted">{z[k].weeks} wk · {z[k].bad_weeks == null ? "–" : pct(z[k].bad_weeks, 0)} bad</span></td>
      ))}
      <td className="r num">{f2(d.diff)}<br /><span className="small muted">t {d.t == null ? "–" : d.t.toFixed(1)}</span></td>
    </tr>
  );
}

export default function Health() {
  const [data] = usePoll("health");
  const h = data?.health;
  if (!h) return (
    <section className="panel"><h2>Market health</h2>
      <p className="small muted">Not computed yet. It appears after the next team run (the Regime Watcher works it out from the NSE data).</p></section>
  );
  const [label, tone] = VERDICT[h.verdict] || [h.verdict, "warn"];
  const hist = data.history;
  return (
    <section className="chart" aria-label="Market health">
      <div className="row" style={{ justifyContent: "space-between" }}>
        <div>
          <span className="eyebrow">Market health · {h.as_of}</span>
          <h2>The NSE top 200 scores {Math.round(h.score)}/100 ({h.zone}); as a warning sign: {label.toLowerCase()}</h2>
        </div>
        <span className={`verdict-pill ${tone}`}>{label}</span>
      </div>
      <p className="small muted">How broad the market is: how many of the 200 stocks are in uptrends, rising vs falling, new highs vs lows, sectors taking part, and the index trend. Each is compared with its own history, so 50 is a normal day. Under {h.low} is weak, over {h.high} strong (fixed before testing).</p>
      <div className="comp-list">
        {Object.entries(h.components).map(([k, c]) => (
          <div className="comp" key={k}>
            <span className="small">{c.name}</span>
            <div className="comp-track"><i style={{ width: `${c.score}%` }} /><b style={{ left: "50%" }} /></div>
            <span className="small num">{Math.round(c.score)}</span>
          </div>
        ))}
      </div>
      {hist?.dates?.length > 10 && <>
        <h3>The last three years</h3>
        <LineChart dates={hist.dates} series={[{ key: "s", label: "Market health", color: "var(--s-model)", values: hist.score }]}
          fmt={x => x.toFixed(0)} height={220} label="Market health over three years" />
      </>}
      <h3>Would a weak score have warned of bad weeks?</h3>
      <p className="small muted">{h.why}. Each cell: the Nifty 200's average return the week after, and how often that week fell more than 2%.</p>
      <div className="scroll">
        <table>
          <thead><tr><th>Period</th><th className="r">Weak (under {h.low})</th><th className="r">Middle</th><th className="r">Strong (over {h.high})</th><th className="r">Weak minus the rest</th></tr></thead>
          <tbody>
            <Period label={`2014–2020 (${h.test.discovery.weeks} weeks)`} t={h.test.discovery} />
            <Period label={`2021–now (${h.test.holdout.weeks} weeks)`} t={h.test.holdout} />
          </tbody>
        </table>
      </div>
      {Object.keys(h.exam || {}).length > 0 && <>
        <h3>A simple rule on the exam weeks: hold half in cash while the score is weak</h3>
        <div className="scroll">
          <table>
            <thead><tr><th>Version</th><th className="r">Always invested</th><th className="r">Half cash when weak</th><th className="r">Worst fall (always → rule)</th><th className="r">Weak weeks</th></tr></thead>
            <tbody>{Object.entries(h.exam).map(([v, e]) => (
              <tr key={v}><td>{v}</td><td className="r num">{signed(e.always.ann_return)}</td><td className="r num">{signed(e.half_cash_when_weak.ann_return)}</td>
                <td className="r num">{pct(e.always.worst_fall, 0)} → {pct(e.half_cash_when_weak.worst_fall, 0)}</td><td className="r num">{e.weak_weeks} of {e.weeks}</td></tr>
            ))}</tbody>
          </table>
        </div>
        <span className="small muted">The rule and its 30 cut-off were fixed before looking; switching costs are counted. Research only, not a timing signal.</span>
      </>}
      <span className="small muted">{h.note} Idea adapted from tradermonty/claude-trading-skills (MIT). Source: results/india/market_health.json</span>
    </section>
  );
}
