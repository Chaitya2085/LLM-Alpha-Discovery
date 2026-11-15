import { useState } from "react";
import { api, usePoll, pct, signed } from "./api.js";
import LineChart from "./LineChart.jsx";

const rupees = x => (x == null ? "–" : `₹${Math.round(x).toLocaleString("en-IN")}`);

export function OpenForm({ code, onDone, gate }) {
  const [amount, setAmount] = useState(50000);
  const [reason, setReason] = useState("");
  const [stop, setStop] = useState(8);
  const [weeks, setWeeks] = useState(4);
  const [err, setErr] = useState(null);
  const [busy, setBusy] = useState(false);
  const submit = async e => {
    e.preventDefault(); setErr(null); setBusy(true);
    try {
      await api("paper/open", { code, amount: +amount, reason, stop_pct: stop ? +stop / 100 : null, plan_weeks: weeks ? +weeks : null });
      onDone?.();
    } catch (x) { setErr(x.message); }
    setBusy(false);
  };
  return (
    <form className="paper-form" onSubmit={submit}>
      <label>Amount (paper rupees)<input className="search" type="number" min="1000" step="1000" value={amount} onChange={e => setAmount(e.target.value)} /></label>
      <label>Stop: flag it if it falls (%)<input className="search" type="number" min="0" max="50" value={stop} onChange={e => setStop(e.target.value)} /></label>
      <label>Plan to hold (weeks)<input className="search" type="number" min="0" max="52" value={weeks} onChange={e => setWeeks(e.target.value)} /></label>
      <label className="wide">Why? What would prove you wrong?<textarea className="idea" rows={2} value={reason} onChange={e => setReason(e.target.value)}
        placeholder="e.g. ranked #3 for falling quietly on low volume; wrong if it keeps falling on heavy volume" /></label>
      <div className="row wide">
        <button className="btn primary" disabled={busy || (gate && !gate.allowed)}>Open the paper position</button>
        <span className="small muted">Pretend money only. No real orders exist in this project.</span>
      </div>
      {gate && !gate.allowed && <p className="err wide">The Risk Officer has paused new positions: {gate.reasons.join("; ")}</p>}
      {err && <p className="err wide">{err}</p>}
    </form>
  );
}

function Row({ t, onClose }) {
  const [busy, setBusy] = useState(false);
  return (
    <tr>
      <td><strong className="num">{t.code}</strong><br /><span className="small muted">{t.opened_on}{t.rank_at_open ? ` · ranked #${t.rank_at_open}` : ""}</span></td>
      <td className="r num">{rupees(t.amount)}</td>
      <td className={`r num ${t.return > 0 ? "pos" : t.return < 0 ? "neg" : ""}`}>{signed(t.return)}</td>
      <td className={`r num ${t.vs_index > 0 ? "pos" : t.vs_index < 0 ? "neg" : ""}`}>{signed(t.vs_index)}</td>
      <td className="small">{t.reason}{t.flags?.length > 0 && <div className="flags">{t.flags.map(f => <span className="flag" key={f}>⚠ {f}</span>)}</div>}</td>
      <td>{onClose ? <button className="btn" disabled={busy} onClick={async () => { setBusy(true); await onClose(t.id); setBusy(false); }}>Close</button>
        : <span className="small muted">{t.closed_on}</span>}</td>
    </tr>
  );
}

export default function Paper({ go }) {
  const [p, error, reload] = usePoll("paper");
  if (error) return <p className="err">{error}</p>;
  if (!p) return <p className="muted">Loading…</p>;
  const close = async id => { await api(`paper/close/${id}`, {}); reload(); };
  const d = p.digest, r = p.rules;
  return (
    <>
      <div className="notice"><strong>Paper journal (personal mode).</strong> Practise with pretend money, write down why before every position, and learn from the record. It stays on this computer. Nothing here is advice or a real order.</div>
      <section className="chart">
        <span className="eyebrow">Paper account</span>
        <h2>{rupees(p.value)} ({signed(p.value / p.start_balance - 1)} since the start)</h2>
        <div className="tiles">
          <div className="stat"><span className="eyebrow">Paper cash</span><b>{rupees(p.cash)}</b><span className="small muted">of {rupees(p.start_balance)} to start</span></div>
          <div className="stat"><span className="eyebrow">Open positions</span><b>{p.open.length} of {r.max_open}</b><span className="small muted">at most {pct(r.max_share, 0)} in one stock</span></div>
          <div className="stat"><span className="eyebrow">Below its high</span><b className={p.drawdown < -0.05 ? "neg" : ""}>{pct(p.drawdown, 1)}</b><span className="small muted">new positions pause at -{pct(r.drawdown_pause, 0)}</span></div>
          <div className="stat"><span className="eyebrow">New positions</span><b>{p.gate.allowed ? "allowed" : "paused"}</b><span className="small muted">{p.gate.allowed ? "the Risk Officer's checks pass" : p.gate.reasons[0]}</span></div>
        </div>
        {p.curve?.dates?.length > 5 && <LineChart dates={p.curve.dates} series={[{ key: "v", label: "Paper account", color: "var(--s-model)", values: p.curve.value }]}
          fmt={x => `₹${(x / 1e5).toFixed(1)}L`} height={200} label="Paper account value" />}
      </section>
      <section className="panel">
        <div className="row" style={{ justifyContent: "space-between" }}>
          <h2>Open positions</h2>
          <button className="btn" onClick={() => go("stocks")}>Open one from a stock page →</button>
        </div>
        {p.open.length === 0 ? <p className="small muted">None yet. Open a stock's page and use "Paper position".</p> : (
          <div className="scroll"><table>
            <thead><tr><th>Stock</th><th className="r">Amount</th><th className="r">Return</th><th className="r">vs Nifty 200</th><th>Why</th><th></th></tr></thead>
            <tbody>{p.open.map(t => <Row key={t.id} t={t} onClose={close} />)}</tbody>
          </table></div>
        )}
      </section>
      <section className="panel">
        <h2>Weekly review</h2>
        {!d.closed ? <p className="small muted">The review starts with your first closed position.</p> : <>
          <div className="tiles">
            <div className="stat"><span className="eyebrow">Closed</span><b>{d.closed}</b><span className="small muted">{pct(d.win_rate, 0)} gained</span></div>
            <div className="stat"><span className="eyebrow">Average</span><b>{signed(d.avg_return)}</b><span className="small muted">win {signed(d.avg_win)} · loss {signed(d.avg_loss)}</span></div>
            <div className="stat"><span className="eyebrow">vs Nifty 200</span><b className={d.avg_vs_index > 0 ? "pos" : "neg"}>{signed(d.avg_vs_index)}</b><span className="small muted">beat it {pct(d.beat_index_rate, 0)} of the time</span></div>
            <div className="stat"><span className="eyebrow">Best / worst</span><b>{signed(d.best, 0)} / {signed(d.worst, 0)}</b><span className="small muted">single positions</span></div>
          </div>
          {Object.keys(d.by_source).length > 0 && <table className="mini"><thead><tr><th>Source</th><th className="r">Closed</th><th className="r">Gained</th><th className="r">vs Nifty 200</th></tr></thead>
            <tbody>{Object.entries(d.by_source).map(([k, x]) => <tr key={k}><td>{k}</td><td className="r num">{x.closed}</td><td className="r num">{pct(x.win_rate, 0)}</td><td className="r num">{signed(x.avg_vs_index)}</td></tr>)}</tbody></table>}
          <div className="scroll"><table>
            <thead><tr><th>Stock</th><th className="r">Amount</th><th className="r">Return</th><th className="r">vs Nifty 200</th><th>Why</th><th>Closed</th></tr></thead>
            <tbody>{p.closed.slice().reverse().map(t => <Row key={t.id} t={t} />)}</tbody>
          </table></div>
        </>}
        <span className="small muted">The Risk Officer's rules, fixed in advance: a written reason first; at most {r.max_open} open and {pct(r.max_share, 0)} of the start in one stock; a pause when the account is {pct(r.drawdown_pause, 0)} below its high; {r.cooldown_days} days off after {r.losing_streak} losses in a row. Ideas adapted from tradermonty/claude-trading-skills (MIT).</span>
      </section>
    </>
  );
}
