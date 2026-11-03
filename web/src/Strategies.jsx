import { useMemo, useState } from "react";
import { usePoll, signed, pct } from "./api.js";
import { verdictLabel } from "./Lab.jsx";

const GROUPS = [
  ["all", "All"], ["works", "Work after costs"], ["costly", "Costs eat it"], ["unstable", "Unstable"],
  ["backwards", "Backwards"], ["weak", "No signal"], ["copy", "Copies"], ["untested", "Daily test only"], ["broken", "Broken"],
];
const inGroup = (s, g) => g === "all" || (g === "works" ? ["keep", "star"].includes(s.verdict)
  : g === "untested" ? !s.verdict && s.status === "valid"
    : g === "broken" ? ["broken", "invalid"].includes(s.verdict)
      : g === "copy" ? ["copy", "duplicate"].includes(s.verdict) : s.verdict === g);

export default function Strategies({ test }) {
  const [data] = usePoll("strategies");
  const [g, setG] = useState("works");
  const [q, setQ] = useState("");
  const [who, setWho] = useState("all");
  const [open, setOpen] = useState(null);
  const list = data?.strategies || [];
  const researchers = useMemo(() => [...new Set(list.map(s => s.researcher))].sort(), [list]);
  const rows = useMemo(() => {
    const t = q.trim().toLowerCase();
    return list.filter(s => inGroup(s, g) && (who === "all" || s.researcher === who)
      && (!t || [s.name, s.expression, s.hypothesis, s.category].some(x => (x || "").toLowerCase().includes(t))))
      .sort((a, b) => (b.net_excess_ann ?? -9) - (a.net_excess_ann ?? -9));
  }, [list, g, q, who]);
  if (!data) return <p className="muted">Loading…</p>;
  const count = k => list.filter(s => inGroup(s, k)).length;
  return (
    <>
      <div>
        <span className="eyebrow">Strategy library</span>
        <h2>Every idea the AI researchers wrote, and what happened to it</h2>
        <p className="small muted">{list.length} ideas from every Researcher, the research rounds and the lab. "Work after costs" means the weekly judge (2014 to 2020) found it beats holding all stocks equally after Indian trading costs. Ideas from the first round were only tested daily; test any of them weekly in the lab.</p>
      </div>
      <div className="row">
        {GROUPS.map(([k, l]) => <button key={k} className="chip" aria-pressed={g === k} onClick={() => setG(k)}>{l} <span className="muted num">{count(k)}</span></button>)}
      </div>
      <div className="row">
        <input className="search" value={q} onChange={e => setQ(e.target.value)} placeholder="Search name, formula or idea" aria-label="Search" />
        <select className="search" value={who} onChange={e => setWho(e.target.value)} aria-label="Researcher">
          <option value="all">All researchers</option>
          {researchers.map(r => <option key={r} value={r}>{r}</option>)}
        </select>
      </div>
      <div className="scroll">
        <table className="hover">
          <thead><tr><th>Idea</th><th>Researcher</th><th>Verdict</th><th className="r">After costs</th><th className="r">Weeks ahead</th></tr></thead>
          <tbody>
            {rows.slice(0, 300).map(s => {
              const [label, tone] = verdictLabel(s.verdict);
              return [
                <tr key={s.id} onClick={() => setOpen(open === s.id ? null : s.id)} tabIndex={0} onKeyDown={e => e.key === "Enter" && setOpen(open === s.id ? null : s.id)}>
                  <td><strong>{(s.name || s.id).replace(/_/g, " ")}</strong><br /><span className="small muted">{s.id} · {s.category}</span></td>
                  <td className="small">{s.researcher}</td>
                  <td><span className={`verdict-pill ${tone}`}>{s.verdict ? label : "daily test only"}</span></td>
                  <td className={`r num ${s.net_excess_ann > 0 ? "pos" : s.net_excess_ann < 0 ? "neg" : ""}`}>{s.net_excess_ann == null ? "–" : signed(s.net_excess_ann)}</td>
                  <td className="r num">{s.hit_rate == null ? "–" : pct(s.hit_rate, 0)}</td>
                </tr>,
                open === s.id && (
                  <tr key={s.id + "-d"} className="detail"><td colSpan={5}>
                    <code className="expr">{s.expression}</code>
                    {s.hypothesis && <p className="small">{s.hypothesis}</p>}
                    {s.idea && <p className="small muted">Your idea: {s.idea}</p>}
                    {s.reason && <p className="small"><strong>The Critic:</strong> {s.reason}</p>}
                    {s.error && <p className="small muted">{s.error}</p>}
                    {s.daily_rank_ic != null && <p className="small muted">Daily test: rank IC {s.daily_rank_ic.toFixed(3)}{s.shortlist ? " · on the Phase 2 shortlist" : ""}</p>}
                    {s.status === "valid" && <button className="btn" onClick={() => test(s.expression)}>Test it weekly in the lab</button>}
                  </td></tr>
                ),
              ];
            })}
          </tbody>
        </table>
      </div>
      {rows.length > 300 && <p className="small muted">Showing the first 300; search to narrow it down.</p>}
    </>
  );
}
