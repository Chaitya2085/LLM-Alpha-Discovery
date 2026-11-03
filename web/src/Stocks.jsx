import { useMemo, useState } from "react";
import { api, usePoll, pct, signed } from "./api.js";
import LineChart, { Legend } from "./LineChart.jsx";

const SORTS = { rank: (a, b) => a.rank - b.rank, code: (a, b) => a.code.localeCompare(b.code), sector: (a, b) => (a.sector || "~").localeCompare(b.sector || "~") || a.rank - b.rank };

function List({ data, open }) {
  const [q, setQ] = useState("");
  const [sort, setSort] = useState("rank");
  const [onlyBook, setOnlyBook] = useState(false);
  const rows = useMemo(() => {
    const t = q.trim().toLowerCase();
    return data.stocks.filter(s => (!t || s.code.toLowerCase().includes(t) || (s.sector || "").toLowerCase().includes(t)) && (!onlyBook || s.in_book))
      .sort(SORTS[sort]);
  }, [data, q, sort, onlyBook]);
  return (
    <>
      <div className="row" style={{ justifyContent: "space-between" }}>
        <div><h2>Every stock, ranked</h2>
          <p className="small muted">All {data.count} stocks the model ranked for the week after {data.as_of} (Alpha158 version). The top 40 are the book.</p></div>
        <div className="row">
          <input className="search" value={q} onChange={e => setQ(e.target.value)} placeholder="Search a stock or sector" aria-label="Search" />
          <label className="toggle"><input type="checkbox" checked={onlyBook} onChange={e => setOnlyBook(e.target.checked)} /> book only</label>
        </div>
      </div>
      <div className="scroll">
        <table className="hover">
          <thead><tr>
            <th><button className="th" onClick={() => setSort("rank")}>Rank</button></th>
            <th><button className="th" onClick={() => setSort("code")}>Stock</button></th>
            <th><button className="th" onClick={() => setSort("sector")}>Sector</button></th>
            <th className="r">Chance of beating the index</th><th>In the book</th><th>Flags</th>
          </tr></thead>
          <tbody>
            {rows.map(s => (
              <tr key={s.code} onClick={() => open(s.code)} tabIndex={0} onKeyDown={e => e.key === "Enter" && open(s.code)}>
                <td className="num">{s.rank}</td><td className="num"><strong>{s.code}</strong></td>
                <td className="small">{s.sector || "–"}</td><td className="r num">{pct(s.p, 1)}</td>
                <td className="small">{s.in_book ? "✓ Alpha158" : ""}{s.in_book && s.in_book_llm ? " · " : ""}{s.in_book_llm ? "✓ LLM" : ""}</td>
                <td>{s.flags.length > 0 && <span className="flag">⚠ {s.flags.length}</span>}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

const PRICE = [
  { key: "stock", label: "The stock", color: "var(--s-model)", width: 2.5 },
  { key: "index", label: "Nifty 200", color: "var(--s-index)" },
];

function Detail({ code, back }) {
  const [s, error, reload] = usePoll(`stocks/${encodeURIComponent(code)}`);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);
  if (error) return <><button className="btn" onClick={back}>← All stocks</button><p className="err">{error}</p></>;
  if (!s) return <p className="muted">Loading {code}…</p>;
  const explain = async () => {
    setBusy(true); setErr(null);
    try { await api(`stocks/${encodeURIComponent(code)}/explain`, {}); reload(); } catch (e) { setErr(e.message); }
    setBusy(false);
  };
  const series = s.prices ? PRICE.map(p => ({ ...p, values: s.prices[p.key] })) : null;
  const ups = s.reasons.filter(r => r.contribution > 0), downs = s.reasons.filter(r => r.contribution <= 0);
  return (
    <>
      <button className="btn" onClick={back} style={{ justifySelf: "start" }}>← All stocks</button>
      <div className="notice"><strong>Personal mode.</strong> How a research model ranks this stock for the week after {s.as_of}. Not a recommendation to buy or sell.</div>
      <section className="chart">
        <div className="row" style={{ justifyContent: "space-between" }}>
          <div><span className="eyebrow">{s.sector || "NSE top 200"}</span><h2 className="num">{s.code}</h2></div>
          {s.flags.length > 0 && <div className="flags">{s.flags.map(f => <span className="flag" key={f}>⚠ {f}</span>)}</div>}
        </div>
        <div className="tiles">
          <div className="stat"><span className="eyebrow">Model rank</span><b>#{s.rank} of {s.of}</b><span className="small muted">{s.in_book ? "in this week's book" : "not in the book (top 40)"}</span></div>
          <div className="stat"><span className="eyebrow">Chance of beating the index</span><b>{pct(s.p, 1)}</b><span className="small muted">next week; 50% = coin flip</span></div>
          {s.llm && <div className="stat"><span className="eyebrow">With LLM factors</span><b>#{s.llm.rank}</b><span className="small muted">{pct(s.llm.p, 1)} · {s.llm.in_book ? "in its book" : "not in its book"}</span></div>}
          {s.returns && <div className="stat"><span className="eyebrow">Past month</span><b className={s.returns.month > 0 ? "pos" : "neg"}>{signed(s.returns.month)}</b><span className="small muted">week {signed(s.returns.week)} · year {signed(s.returns.year)}</span></div>}
        </div>
        {series && <><Legend series={series} /><LineChart dates={s.prices.dates} series={series} fmt={v => v.toFixed(0)} label={`${s.code} against the Nifty 200 over a year`} />
          <span className="small muted">The past year, both started at 100. Past moves are context, not a forecast.</span></>}
      </section>
      <div className="two">
        <section className="panel">
          <h3>Why the model ranks it here</h3>
          {s.reasons.length === 0 && <p className="small muted">Reasons for this stock appear after the next weekly run (python run_team.py --skip-download --weekly).</p>}
          {ups.length > 0 && <><span className="eyebrow">Raised its score</span><ul className="facts">{ups.map(r => <li key={r.feature}>{r.fact}</li>)}</ul></>}
          {downs.length > 0 && <><span className="eyebrow" style={{ color: "var(--stop)" }}>Lowered its score</span><ul className="facts">{downs.map(r => <li key={r.feature}>{r.fact}</li>)}</ul></>}
          {s.note ? <div className="answer"><strong>In plain words:</strong> {s.note.summary}<br /><span className="muted">What would change the view: {s.note.watch}</span></div>
            : s.reasons.length > 0 && <button className="btn" disabled={busy} onClick={explain}>{busy ? "The Explainer is writing…" : "Ask the Explainer to put this in plain words"}</button>}
          {err && <p className="small" style={{ color: "var(--stop)" }}>{err}</p>}
        </section>
        <section className="panel">
          <h3>Its weeks in the ledger</h3>
          {s.history.length === 0 ? <p className="small muted">No sealed weeks yet.</p> : (
            <table><thead><tr><th>Week after</th><th className="r">Rank</th><th className="r">Chance</th><th>Book</th></tr></thead>
              <tbody>{s.history.map(h => <tr key={h.as_of}><td className="num">{h.as_of}</td><td className="r num">#{h.rank}</td><td className="r num">{pct(h.p, 1)}</td><td>{h.in_book ? "✓" : ""}</td></tr>)}</tbody></table>
          )}
          <p className="small muted">Each week's ranking is sealed before the week starts; results are scored 31 days later.</p>
        </section>
      </div>
    </>
  );
}

export default function Stocks({ arg, go }) {
  const [data, error] = usePoll("stocks");
  if (arg) return <Detail code={decodeURIComponent(arg)} back={() => go("stocks")} />;
  if (error) return null;
  if (!data) return <p className="muted">Loading…</p>;
  if (!data.stocks.length) return <p className="empty">No weekly ranking yet. Run the team's weekly jobs first.</p>;
  return <List data={data} open={c => go(`stocks/${encodeURIComponent(c)}`)} />;
}
