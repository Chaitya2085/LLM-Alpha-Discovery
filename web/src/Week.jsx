import { usePoll } from "./api.js";

export default function Week() {
  const [w, error] = usePoll("week");
  if (error) return null;
  if (!w) return <p className="muted">Loading…</p>;
  if (!w.stocks.length) return <p className="empty">No explanations yet. Run the team's weekly jobs (War room → Run the team) after recording a week.</p>;
  return (
    <>
      <div className="notice">
        <strong>Personal mode.</strong> This page names stocks, so it only exists on your computer. It shows how a research
        model ranked stocks for the week after {w.as_of}; it is not a recommendation to buy or sell anything.
      </div>
      <div className="row" style={{ justifyContent: "space-between" }}>
        <h2>Why the model ranks these stocks highly</h2>
        <span className="small muted">{w.variant} · top {w.stocks.length} of the book</span>
      </div>
      <p className="small muted">The bar is the model's chance that the stock beats the Nifty 200 next week; the line marks 50%. Near 50% is normal: one stock in one week is close to a coin flip. The edge comes from ranking many stocks, not from certainty about one.</p>
      <div className="picks">
        {w.stocks.map(s => (
          <article className="pick" key={s.code}>
            <div>
              <div className="code">{s.code}</div>
              <div className="num small">{Math.round(s.confidence * 100)}% chance</div>
              <div className="meter" aria-hidden="true"><i style={{ width: `${s.confidence * 100}%` }} /><b /></div>
            </div>
            <div style={{ display: "grid", gap: 6 }}>
              <p>{s.summary}</p>
              <p className="small muted"><strong>What would change the view:</strong> {s.watch}</p>
              {s.risk?.length > 0 && <div className="flags">{s.risk.map(r => <span className="flag" key={r}>⚠ {r}</span>)}</div>}
              <details><summary>The facts behind this</summary><ul>{s.facts.map(f => <li key={f}>{f}</li>)}</ul></details>
            </div>
          </article>
        ))}
      </div>
    </>
  );
}
