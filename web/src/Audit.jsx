import { useState } from "react";
import { pct, signed } from "./api.js";
import LineChart from "./LineChart.jsx";

const VERDICT = {
  "PICKING EDGE": ["Picking adds an edge", "good"],
  MAYBE: ["Maybe: could still be luck", "warn"],
  "NOT PROVEN": ["Not proven yet", "warn"],
  "NO PICKING EDGE": ["No picking edge", "bad"],
};
const sure = t => (t >= 2 ? "unlikely to be luck" : t >= 1 ? "could be luck" : "can't tell from luck");

function Split({ s }) {
  // model's average return = the part riding the 200 stocks explains + what's left from picking
  const parts = [
    { label: `riding the 200 stocks (${s.beta_equal.toFixed(2)} × equal weight)`, v: s.explained_by_equal_ann, color: "var(--s-equal)" },
    { label: "left over from picking", v: s.alpha_ann, color: s.alpha_ann >= 0 ? "var(--s-model)" : "var(--stop)" },
  ];
  const total = parts.reduce((a, p) => a + Math.abs(p.v), 0) || 1;
  return (
    <div style={{ display: "grid", gap: 6 }}>
      <div className="splitbar" role="img" aria-label="How the model's return splits">
        {parts.map(p => <span key={p.label} style={{ width: `${(Math.abs(p.v) / total) * 100}%`, background: p.color }} title={`${p.label}: ${signed(p.v)}`} />)}
      </div>
      <div className="legend">
        {parts.map(p => <span key={p.label}><i style={{ background: p.color }} />{signed(p.v)} {p.label}</span>)}
      </div>
      <span className="small muted">Model {signed(s.model_ann)} a year on average after costs; equal weight {signed(s.equal_ann)}; Nifty 200 {signed(s.index_ann)}.</span>
    </div>
  );
}

function Histogram({ h }) {
  const [hov, setHov] = useState(null);
  const W = 1100, H = 200, L = 8, B = 26;
  const n = h.counts.length, max = Math.max(...h.counts), total = h.counts.reduce((a, b) => a + b, 0);
  const bw = (W - 2 * L) / n;
  const zero = h.edges.findIndex(e => e >= 0);
  return (
    <div style={{ position: "relative" }}>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="One-year replays: model minus equal weight">
        {h.counts.map((c, i) => {
          const hgt = (c / max) * (H - B - 10), x = L + i * bw;
          const pos = h.edges[i] >= 0;
          return (
            <g key={i} onPointerEnter={() => setHov(i)} onPointerLeave={() => setHov(null)}>
              <rect x={x} y={0} width={bw} height={H - B} fill="transparent" />
              <rect x={x + 1} y={H - B - hgt} width={Math.max(1, bw - 2)} height={hgt} rx="2"
                fill={pos ? "var(--s-model)" : "var(--idle)"} opacity={hov === null || hov === i ? 1 : 0.55} />
            </g>
          );
        })}
        {zero > 0 && <line x1={L + zero * bw} x2={L + zero * bw} y1={4} y2={H - B} stroke="var(--ink)" strokeDasharray="3 3" />}
        {zero > 0 && <text x={L + zero * bw + 4} y={14} fontSize="11" style={{ fill: "var(--ink)" }}>0 = same as equal weight</text>}
        {[0, Math.floor(n / 2), n].map(i => <text key={i} x={L + i * bw} y={H - 8} fontSize="11" textAnchor={i === 0 ? "start" : i === n ? "end" : "middle"}>{signed(h.edges[i], 0)}</text>)}
      </svg>
      {hov !== null && (
        <div className="tip" style={{ left: Math.min(((L + hov * bw) / W) * 100, 70) + "%", top: 0 }}>
          <div><span>{signed(h.edges[hov], 1)} to {signed(h.edges[hov + 1], 1)}</span></div>
          <div><span>Replays</span><span className="num">{pct(h.counts[hov] / total, 1)}</span></div>
        </div>
      )}
    </div>
  );
}

function Ranges({ r }) {
  const rows = [["The model", r.model, "var(--s-model)"], ["Nifty 200", r.index, "var(--s-index)"], ["All 200, equal weight", r.equal, "var(--s-equal)"]];
  const lo = Math.min(...rows.map(x => x[1]["5"])), hi = Math.max(...rows.map(x => x[1]["95"]));
  const X = v => ((v - lo) / (hi - lo)) * 100;
  return (
    <div className="ranges">
      {rows.map(([name, q, c]) => (
        <div className="range-row" key={name}>
          <span className="small">{name}</span>
          <div className="range-track" title={`5% of replays below ${signed(q["5"])}, half below ${signed(q["50"])}, 5% above ${signed(q["95"])}`}>
            <i style={{ left: `${X(q["5"])}%`, width: `${X(q["95"]) - X(q["5"])}%`, background: c, opacity: 0.25 }} />
            <i style={{ left: `${X(q["25"])}%`, width: `${X(q["75"]) - X(q["25"])}%`, background: c, opacity: 0.6 }} />
            <b style={{ left: `${X(q["50"])}%`, background: c }} />
          </div>
          <span className="small num">{signed(q["5"], 0)} … <strong>{signed(q["50"], 0)}</strong> … {signed(q["95"], 0)}</span>
        </div>
      ))}
      <span className="small muted">Each bar: the middle 90% of one-year results (light), the middle half (dark), and the typical year (line).</span>
    </div>
  );
}

export default function Audit({ audit }) {
  const variants = Object.keys(audit?.exam || {}).sort((x, y) => x.includes("LLM") - y.includes("LLM"));
  const [v, setV] = useState(variants[0]);
  if (!variants.length) return null;
  const a = audit.exam[v] || audit.exam[variants[0]];
  const d = a.decompose, e = d.equal, r = a.replay;
  const [label, tone] = VERDICT[e.verdict] || [e.verdict, "warn"];
  const live = Object.entries(audit.live || {});
  return (
    <>
      <section className="chart" aria-label="Is the edge real?">
        <div className="row" style={{ justifyContent: "space-between" }}>
          <div>
            <span className="eyebrow">Is the edge real? · exam weeks {a.from.slice(0, 4)} to {a.to.slice(0, 4)}</span>
            <h2>Beyond holding all 200 stocks equally, picking added {signed(e.alpha_ann)} a year: {label.toLowerCase()}</h2>
          </div>
          <div className="seg" role="group" aria-label="Model version">
            {variants.map(n => <button key={n} aria-pressed={n === v} onClick={() => setV(n)}>{n}</button>)}
          </div>
        </div>
        <p className="small muted">The model beat the Nifty 200 mostly by owning mid-sized companies, which holding all 200 stocks equally also does. This check takes out the part that a simple copy explains and asks whether anything is left from the picking itself.</p>
        <Split s={d.split} />
        <div className="scroll">
          <table>
            <thead><tr><th>Compared with</th><th className="r">Left over a year</th><th className="r">t</th><th>How sure</th><th>Verdict</th></tr></thead>
            <tbody>{["equal", "index", "two_factor"].map(k => {
              const x = d[k], [lab, tn] = VERDICT[x.verdict] || [x.verdict, "warn"];
              return <tr key={k}><td>{x.name}</td><td className={`r num ${x.alpha_ann > 0 ? "pos" : "neg"}`}>{signed(x.alpha_ann)}</td>
                <td className="r num">{x.alpha_t.toFixed(1)}</td><td className="small">{sure(x.alpha_t)}</td>
                <td><span className={`verdict-pill ${tn}`}>{lab}</span></td></tr>;
            })}</tbody>
          </table>
        </div>
        <span className="small muted">t of 2 or more means the leftover is unlikely to be luck. Basis: the model after costs against baselines without costs; Newey-West errors; baselines chosen before looking. {live.length ? "" : "The same check runs on the live ledger after 26 revealed weeks."}</span>
        <h3>The leftover over each trailing year</h3>
        <LineChart dates={a.rolling.dates} series={[{ key: "a", label: "Left over beyond equal weight", color: "var(--s-model)", values: a.rolling.alpha_ann }]}
          fmt={x => signed(x, 0)} height={240} label="Trailing one-year leftover return" />
        <h3>By year and market</h3>
        <div className="scroll">
          <table><thead><tr><th>Period</th>{[...a.periods.years, ...a.periods.markets].map(p => <th key={p.label} className="r">{p.label}</th>)}</tr></thead>
            <tbody>
              <tr><td>Left over</td>{[...a.periods.years, ...a.periods.markets].map(p => <td key={p.label} className={`r num ${p.alpha_ann > 0 ? "pos" : "neg"}`}>{signed(p.alpha_ann)}</td>)}</tr>
              <tr><td className="muted">Weeks</td>{[...a.periods.years, ...a.periods.markets].map(p => <td key={p.label} className="r num muted">{p.weeks}</td>)}</tr>
            </tbody></table>
        </div>
        <span className="small muted">Rising or falling market = the Nifty 200's move over the 13 weeks before.</span>
      </section>
      <section className="chart" aria-label="Range of outcomes">
        <span className="eyebrow">Range of outcomes · {r.replays.toLocaleString()} one-year replays</span>
        <h2>In {pct(r.p_beat_index, 0)} of replayed years the model beat the Nifty 200; in {pct(r.p_beat_equal, 0)} it beat holding all 200 equally</h2>
        <p className="small muted">The {r.source_weeks} real exam weeks, shuffled in {r.block_weeks}-week blocks (so bad stretches stay together) into {r.replays.toLocaleString()} possible years. A replay of the past, not a forecast: the future can do things these weeks never did.</p>
        <div className="tiles">
          <div className="stat"><span className="eyebrow">Beat the Nifty 200</span><b>{pct(r.p_beat_index, 0)}</b><span className="small muted">of replayed years</span></div>
          <div className="stat"><span className="eyebrow">Beat equal weight</span><b>{pct(r.p_beat_equal, 0)}</b><span className="small muted">of replayed years</span></div>
          <div className="stat"><span className="eyebrow">Lost money</span><b>{pct(r.p_loss, 0)}</b><span className="small muted">of replayed years</span></div>
          <div className="stat"><span className="eyebrow">Worst fall in a year</span><b>{signed(r.worst_drawdown["50"], 0)}</b><span className="small muted">typical; {signed(r.worst_drawdown["5"], 0)} in the worst 5%</span></div>
        </div>
        <Ranges r={r} />
        <h3>The model minus equal weight, over one year</h3>
        <Histogram h={r.hist_vs_equal} />
        <span className="small muted">Seed {r.seed}: run it again and you get the same answer. Source: results/india/edge_audit.json</span>
      </section>
    </>
  );
}
