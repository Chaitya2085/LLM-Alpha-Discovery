import { useMemo, useRef, useState } from "react";
import { usePoll, pct, signed, lakh } from "./api.js";

const SERIES = [
  { key: "model", label: "The model", color: "var(--s-model)", width: 2.5 },
  { key: "index", label: "Nifty 200", color: "var(--s-index)", width: 2 },
  { key: "equal", label: "All 200 stocks, equal weight", color: "var(--s-equal)", width: 2, dash: "5 4" },
];
const VERDICT = {
  "REAL EDGE": "Real edge: positive and at least 95% likely to be real after counting every design tried",
  PROMISING: "Promising, not proven: likely real on its own, but not after counting every design tried",
  "NO PROVEN EDGE": "No proven edge on years the model had never seen",
};

function GrowthChart({ g }) {
  const [hover, setHover] = useState(null);
  const svgRef = useRef(null);
  const W = 760, H = 300, L = 52, R = 160, T = 14, B = 34;
  const series = SERIES.filter(s => g[s.key]);
  const { x, y, ticks, years } = useMemo(() => {
    const all = series.flatMap(s => g[s.key]);
    const lo = Math.min(0.8, ...all), hi = Math.max(...all) * 1.04;
    const n = g.dates.length;
    const x = i => L + (i / (n - 1)) * (W - L - R);
    const y = v => T + (1 - (v - lo) / (hi - lo)) * (H - T - B);
    const step = hi - lo > 3 ? 1 : 0.5;
    const ticks = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) ticks.push(+v.toFixed(2));
    const years = [];
    g.dates.forEach((d, i) => { if (i === 0 || d.slice(0, 4) !== g.dates[i - 1].slice(0, 4)) years.push([i, d.slice(0, 4)]); });
    return { x, y, ticks, years };
  }, [g]); // eslint-disable-line react-hooks/exhaustive-deps
  const path = arr => arr.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)} ${y(v).toFixed(1)}`).join("");
  const onMove = e => {
    const r = svgRef.current.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * W;
    const i = Math.round(((px - L) / (W - L - R)) * (g.dates.length - 1));
    if (i < 0 || i >= g.dates.length) return setHover(null);
    setHover({ i, left: (x(i) / W) * r.width, top: 10 });
  };
  const last = g.dates.length - 1;
  const ends = series.map(s => ({ s, v: g[s.key][last] })).sort((a, b) => b.v - a.v);
  let prevY = -1e9;
  return (
    <div style={{ position: "relative" }}>
      <svg ref={svgRef} viewBox={`0 0 ${W} ${H}`} role="img" onPointerMove={onMove} onPointerLeave={() => setHover(null)}
        aria-label="Growth of 1 lakh rupees since the start of the final exam">
        {ticks.map(v => (
          <g key={v}>
            <line x1={L} x2={W - R} y1={y(v)} y2={y(v)} stroke="var(--line)" />
            <text x={L - 8} y={y(v) + 4} textAnchor="end" fontSize="11">₹{v.toFixed(1)}L</text>
          </g>
        ))}
        {years.map(([i, yr]) => <text key={yr} x={x(i)} y={H - 12} fontSize="11" textAnchor="start">{yr}</text>)}
        {series.map(s => <path key={s.key} d={path(g[s.key])} fill="none" stroke={s.color} strokeWidth={s.width}
          strokeDasharray={s.dash} strokeLinejoin="round" />)}
        {ends.map(({ s, v }) => {
          const ly = Math.max(y(v) + 4, prevY + 15); prevY = ly;
          return (
            <g key={s.key}>
              <circle cx={x(last)} cy={y(v)} r="4" fill={s.color} stroke="var(--card)" strokeWidth="2" />
              <text x={x(last) + 10} y={ly} fontSize="11.5" style={{ fill: "var(--ink)" }}>{lakh(v)}</text>
            </g>
          );
        })}
        {hover && (
          <g>
            <line x1={x(hover.i)} x2={x(hover.i)} y1={T} y2={H - B} stroke="var(--muted)" strokeDasharray="3 3" />
            {series.map(s => <circle key={s.key} cx={x(hover.i)} cy={y(g[s.key][hover.i])} r="4" fill={s.color} stroke="var(--card)" strokeWidth="2" />)}
          </g>
        )}
        <rect x={L} y={T} width={W - L - R} height={H - T - B} fill="transparent" />
      </svg>
      {hover && (
        <div className="tip" style={{ left: Math.min(hover.left + 12, 520), top: hover.top }}>
          <strong className="num">{g.dates[hover.i]}</strong>
          {series.map(s => <div key={s.key}><span>{s.label}</span><span className="num">{lakh(g[s.key][hover.i])}</span></div>)}
        </div>
      )}
    </div>
  );
}

function Exam({ exam, growth }) {
  const variants = Object.keys(exam.variants);
  const [v, setV] = useState(variants[0]);
  const [table, setTable] = useState(false);
  const d = exam.variants[v];
  const b = d.book;
  const g = growth?.[v];
  const eq = g?.equal ? Math.pow(g.equal.at(-1), 52 / g.dates.length) - 1 : null;
  return (
    <section className="chart" aria-label="Final exam">
      <div className="row" style={{ justifyContent: "space-between" }}>
        <div>
          <span className="eyebrow">Final exam · {exam.years?.[0]} to {exam.years?.at(-1)} · settings frozen first</span>
          <h2>₹1 lakh in the model became {lakh(g?.model.at(-1))}, against {lakh(g?.index.at(-1))} in the Nifty 200</h2>
        </div>
        <div className="seg" role="group" aria-label="Model version">
          {variants.map(n => <button key={n} aria-pressed={n === v} onClick={() => setV(n)}>{n}</button>)}
        </div>
      </div>
      <div className={`verdict ${d.verdict.verdict.split(" ")[0]}`}>
        <strong>{VERDICT[d.verdict.verdict] || d.verdict.verdict}</strong>
        <span className="small">Chance the edge is real: {pct(d.verdict.psr, 0)} on its own, {pct(d.verdict.dsr, 0)} after counting the designs tried (pass mark 95%).
          {eq != null && ` Holding all 200 stocks equally earned ${pct(eq)} a year, so compare the model with that too.`}</span>
      </div>
      <div className="tiles">
        <div className="stat"><span className="eyebrow">A year, after costs</span><b>{pct(b.ann_return)}</b><span className="small muted">Nifty 200: {pct(b.bench_ann_return)}</span></div>
        <div className="stat"><span className="eyebrow">Over the index</span><b className={b.excess_ann > 0 ? "pos" : "neg"}>{signed(b.excess_ann)}</b><span className="small muted">a year, after {pct(b.cost_ann)} of costs</span></div>
        <div className="stat"><span className="eyebrow">Weeks beating it</span><b>{pct(b.weekly_hit_rate, 0)}</b><span className="small muted">of {b.weeks} weeks</span></div>
        <div className="stat"><span className="eyebrow">Worst fall</span><b>{pct(b.max_drawdown, 0)}</b><span className="small muted">from a high</span></div>
        <div className="stat"><span className="eyebrow">Confidence gap</span><b>{d.calibration_gap == null ? "–" : `${(d.calibration_gap * 100).toFixed(1)} pts`}</b><span className="small muted">stated vs what happened</span></div>
      </div>
      <div className="legend" aria-label="Legend">
        {SERIES.map(s => <span key={s.key}><i style={{ background: s.color }} />{s.label}</span>)}
        <button className="btn" style={{ marginLeft: "auto", padding: "3px 10px" }} onClick={() => setTable(!table)}>{table ? "Show chart" : "Show as table"}</button>
      </div>
      {g && !table && <GrowthChart g={g} />}
      {g && table && (
        <div className="scroll" style={{ maxHeight: 320 }}>
          <table>
            <thead><tr><th>Week</th>{SERIES.map(s => <th key={s.key} className="r">{s.label}</th>)}</tr></thead>
            <tbody>{g.dates.map((dt, i) => (i % 4 === 0 || i === g.dates.length - 1) && (
              <tr key={dt}><td className="num">{dt}</td>{SERIES.map(s => <td key={s.key} className="r num">{g[s.key] ? lakh(g[s.key][i]) : "–"}</td>)}</tr>
            ))}</tbody>
          </table>
        </div>
      )}
      <span className="small muted">Weekly, 40 stocks, after Indian trading costs. Each year predicted by a model trained only on earlier years. Source: results/india/phase6_weekly.csv</span>
    </section>
  );
}

function Ledger({ rows, scores }) {
  const weeks = {};
  rows.forEach(r => { (weeks[r.as_of] ||= []).push(r); });
  const score = (as_of, variant) => scores.find(s => s.as_of === as_of && s.variant === variant);
  return (
    <section className="panel" aria-label="Prediction ledger">
      <h2>The live record</h2>
      <p className="small muted">Each week's predictions are fingerprinted before the week starts and stay sealed for 31 days, so nobody can change them after seeing the result. The fingerprints are public in ledger/commitments.csv.</p>
      {rows.length === 0 ? <p className="muted">No weeks recorded yet.</p> : (
        <div className="scroll">
          <table>
            <thead><tr><th>Week after</th><th>Version</th><th>Fingerprint</th><th>Result</th></tr></thead>
            <tbody>
              {Object.entries(weeks).reverse().flatMap(([as_of, rs]) => rs.map(r => {
                const s = score(as_of, r.variant);
                return (
                  <tr key={as_of + r.variant}>
                    <td className="num">{as_of}</td><td>{r.variant}</td><td className="num">{r.fingerprint}…</td>
                    <td>{s ? <span className={s.excess > 0 ? "pos" : "neg"}>{signed(s.book_return, 2)} vs {signed(s.index_return, 2)}</span>
                      : r.revealed ? "revealed, scoring soon" : `sealed until ${r.reveal_on}`}</td>
                  </tr>
                );
              }))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

export default function Track() {
  const [t, error] = usePoll("track");
  if (error) return null;
  if (!t) return <p className="muted">Loading…</p>;
  return (
    <>
      {t.exam ? <Exam exam={t.exam} growth={t.growth} /> : <p className="empty">Run the final exam first: python run_phase6.py</p>}
      <Ledger rows={t.ledger} scores={t.scores} />
    </>
  );
}
