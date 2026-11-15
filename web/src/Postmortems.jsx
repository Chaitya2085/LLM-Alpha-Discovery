import { signed, pct } from "./api.js";

function Card({ pm }) {
  const r = pm.result;
  return (
    <article className="panel">
      <div className="row" style={{ justifyContent: "space-between" }}>
        <div><span className="eyebrow">Week after {pm.as_of}</span><h3>{pm.variant}</h3></div>
        <div className="row">
          <span className={`verdict-pill ${r.beat_index ? "good" : "bad"}`}>{r.beat_index ? "Beat" : "Trailed"} the Nifty 200</span>
          <span className={`verdict-pill ${r.beat_equal ? "good" : "bad"}`}>{r.beat_equal ? "Beat" : "Trailed"} equal weight</span>
        </div>
      </div>
      <ul className="facts">{pm.notes.map((n, i) => <li key={i}>{n}</li>)}</ul>
      <details>
        <summary>Picks and sectors</summary>
        <div className="two" style={{ marginTop: 8 }}>
          <table className="mini"><thead><tr><th>Best picks</th><th className="r"></th></tr></thead>
            <tbody>{pm.best.map(b => <tr key={b.code}><td className="num">{b.code}</td><td className="r num pos">{signed(b.return)}</td></tr>)}</tbody></table>
          <table className="mini"><thead><tr><th>Worst picks</th><th className="r"></th></tr></thead>
            <tbody>{pm.worst.map(b => <tr key={b.code}><td className="num">{b.code}</td><td className="r num neg">{signed(b.return)}</td></tr>)}</tbody></table>
        </div>
        {[...pm.sectors.helped, ...pm.sectors.hurt].length > 0 && (
          <table className="mini"><thead><tr><th>Sector</th><th className="r">Book / all 200</th><th className="r">Weight effect</th><th className="r">Picking effect</th></tr></thead>
            <tbody>{[...pm.sectors.helped, ...pm.sectors.hurt].map(x => (
              <tr key={x.sector}><td>{x.sector}</td><td className="r num">{pct(x.book_weight, 0)} / {pct(x.universe_weight, 0)}</td>
                <td className="r num">{signed(x.allocation, 2)}</td><td className="r num">{signed(x.selection, 2)}</td></tr>
            ))}</tbody></table>
        )}
      </details>
    </article>
  );
}

export default function Postmortems({ data, ledger }) {
  const weeks = data?.weeks || [];
  const next = (ledger || []).filter(r => !r.revealed).map(r => r.reveal_on).sort()[0];
  const pats = Object.entries(data?.patterns || {}).filter(([, p]) => p.by_mood);
  return (
    <section className="chart" aria-label="Post-mortems">
      <span className="eyebrow">The Critic's post-mortems</span>
      <h2>{weeks.length ? `${weeks.length} revealed book(s), and why each went the way it did` : "Why each sealed week went the way it did"}</h2>
      <p className="small muted">When a sealed week opens (31 days later), the Critic writes one post-mortem from the numbers only: the result, whether the ranking worked across all 200, which sector bets helped or hurt, whether the model's main reasons paid off, and the market at the time. Written once, never edited. Idea adapted from tradermonty/claude-trading-skills (MIT).</p>
      {!weeks.length && <p className="muted">{next ? `The first one is written after ${next}, the first time the team runs after that date.` : "No sealed weeks yet."}</p>}
      {weeks.slice().reverse().slice(0, 12).map(pm => <Card key={pm.as_of + pm.variant} pm={pm} />)}
      {pats.length > 0 && <>
        <h3>Patterns so far</h3>
        {pats.map(([v, p]) => (
          <div className="scroll" key={v}>
            <table><thead><tr><th>{v} · {p.weeks} weeks</th><th className="r">Weeks</th><th className="r">Beat the index</th><th className="r">Beat equal weight</th></tr></thead>
              <tbody>{[...Object.entries(p.by_mood), ...Object.entries(p.by_health).map(([k, x]) => [`health: ${k}`, x])].map(([k, x]) => (
                <tr key={k}><td>{k}</td><td className="r num">{x.weeks}</td><td className="r num">{pct(x.beat_index, 0)}</td><td className="r num">{pct(x.beat_equal, 0)}</td></tr>
              ))}</tbody></table>
          </div>
        ))}
        <span className="small muted">Descriptions to learn from, not rules: nothing here changes the frozen model.</span>
      </>}
    </section>
  );
}
