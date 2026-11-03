import { useEffect, useState } from "react";
import { api, pct, signed } from "./api.js";
import LineChart, { Legend } from "./LineChart.jsx";

const EXAMPLES = [
  "Stocks that fell a lot this month but on low trading volume",
  "Calm, steadily rising stocks beat jumpy ones",
  "Stocks that keep closing near the day's high",
  "Heavily traded stocks whose price hasn't moved yet",
];
const VERDICT = {
  star: ["Works strongly after costs", "good"], keep: ["Works after costs", "good"],
  costly: ["Real signal, but costs eat it", "warn"], unstable: ["Works in some years, fails in others", "warn"],
  backwards: ["Works in reverse", "warn"], weak: ["No reliable signal", "bad"], copy: ["A copy of a known idea", "bad"],
  duplicate: ["Already in the library", "bad"], broken: ["Couldn't be computed", "bad"],
};
export const verdictLabel = v => (VERDICT[v] || [v || "not tested", "bad"]);

function Result({ r, i, job, onSaved, flip }) {
  const [saved, setSaved] = useState(r.saved_as);
  const [err, setErr] = useState(null);
  const [label, tone] = verdictLabel(r.verdict);
  const series = r.curve ? [
    { key: "f", label: "This idea's top fifth, after costs", color: "var(--s-model)", values: r.curve.factor, width: 2.5 },
    { key: "e", label: "All stocks equally", color: "var(--s-equal)", values: r.curve.equal, dash: "5 4" },
  ] : null;
  const save = async () => {
    setErr(null);
    try { const s = await api("lab-save", { job, index: i }); setSaved(s.id); onSaved?.(); } catch (e) { setErr(e.message); }
  };
  return (
    <article className="chart">
      <div className="row" style={{ justifyContent: "space-between" }}>
        <div><span className="eyebrow">{r.category}</span><h3>{(r.name || "formula").replace(/_/g, " ")}</h3></div>
        <span className={`verdict-pill ${tone}`}>{label}</span>
      </div>
      <code className="expr">{r.expression || r.raw_expression}</code>
      {r.hypothesis && <p className="small muted">{r.hypothesis}</p>}
      <p><strong>The Critic:</strong> {r.reason || r.error}</p>
      {r.judge_status === "scored" && (
        <div className="tiles">
          <div className="stat"><span className="eyebrow">After costs</span><b className={r.net_excess_ann > 0 ? "pos" : "neg"}>{signed(r.net_excess_ann)}</b><span className="small muted">a year vs all stocks equally</span></div>
          <div className="stat"><span className="eyebrow">Before costs</span><b>{signed(r.gross_excess_ann)}</b><span className="small muted">costs {pct(r.cost_ann)} a year</span></div>
          <div className="stat"><span className="eyebrow">Weeks ahead</span><b>{pct(r.hit_rate, 0)}</b><span className="small muted">of {r.weeks} weeks</span></div>
          <div className="stat"><span className="eyebrow">Signal strength</span><b>t {r.weekly_ic_t?.toFixed(1)}</b><span className="small muted">2 or more = reliable</span></div>
          <div className="stat"><span className="eyebrow">Same direction</span><b>{pct(r.years_same_sign, 0)}</b><span className="small muted">of years</span></div>
        </div>
      )}
      {series && <><Legend series={series} /><LineChart dates={r.curve.dates} series={series} fmt={v => `₹${v.toFixed(2)}L`} height={230} label="Growth of 1 lakh, 2014 to 2020" />
        <span className="small muted">₹1 lakh in the idea's top fifth each week, held 5 days, after Indian costs, 2014 to 2020 only.</span></>}
      {r.closest_known && r.max_corr_known > 0.3 && <p className="small muted">Most similar known feature: {r.closest_known} (similarity {r.max_corr_known.toFixed(2)})</p>}
      <div className="row">
        {r.verdict === "backwards" && <button className="btn" onClick={() => flip(`-(${r.expression})`)}>Test the opposite idea</button>}
        {saved ? <span className="small">Saved to the library as {saved}.</span>
          : r.status === "valid" && <button className="btn" onClick={save}>Save to the strategy library</button>}
        {err && <span className="small" style={{ color: "var(--stop)" }}>{err}</span>}
      </div>
    </article>
  );
}

export default function Lab({ preset }) {
  const [mode, setMode] = useState(preset ? "formula" : "idea");
  const [idea, setIdea] = useState("");
  const [expr, setExpr] = useState(preset || "");
  const [job, setJob] = useState(null);
  const [err, setErr] = useState(null);
  const [grammar, setGrammar] = useState(null);
  useEffect(() => { if (preset) { setMode("formula"); setExpr(preset); } }, [preset]);
  useEffect(() => {
    if (!job || job.status === "done" || job.status === "failed") return undefined;
    const t = setInterval(() => api(`lab/${job.id}`).then(setJob).catch(e => setErr(e.message)), 1500);
    return () => clearInterval(t);
  }, [job]);
  const run = async (e, override) => {
    e?.preventDefault(); setErr(null);
    try {
      const body = override ? { expression: override } : mode === "idea" ? { idea } : { expression: expr };
      const { id } = await api("lab", body);
      setJob({ id, status: "queued", step: "Starting", results: [] });
    } catch (x) { setErr(x.message); }
  };
  const busy = job && (job.status === "queued" || job.status === "running");
  return (
    <>
      <section className="panel">
        <span className="eyebrow">Research lab</span>
        <h2>Test an idea the honest way</h2>
        <p className="small muted">Describe a stock-picking idea. A Researcher turns it into formulas, then the weekly judge tests each one on 2014 to 2020 only (top fifth, held a week, after Indian costs, against holding all stocks equally) and the Critic gives a verdict. The exam years (2021 on) are never used here.</p>
        <div className="view-switch" role="group" aria-label="How to describe it" style={{ justifySelf: "start" }}>
          <button aria-pressed={mode === "idea"} onClick={() => setMode("idea")}>In plain words</button>
          <button aria-pressed={mode === "formula"} onClick={() => setMode("formula")}>I'll write the formula</button>
        </div>
        <form onSubmit={run} style={{ display: "grid", gap: 8 }}>
          {mode === "idea" ? <>
            <textarea className="idea" value={idea} onChange={e => setIdea(e.target.value)} rows={3} maxLength={600}
              placeholder="e.g. stocks that fell a lot this month but on low trading volume" aria-label="Your idea" />
            <div className="row">{EXAMPLES.map(x => <button type="button" className="chip" key={x} onClick={() => setIdea(x)}>{x}</button>)}</div>
          </> : <>
            <input className="idea mono" value={expr} onChange={e => setExpr(e.target.value)} placeholder="e.g. -Pct(close, 20) / (Mean(volume, 20) / Mean(volume, 60))" aria-label="Formula" />
            <button type="button" className="btn" style={{ justifySelf: "start" }} onClick={() => grammar ? setGrammar(null) : api("grammar").then(g => setGrammar(g.grammar))}>{grammar ? "Hide" : "Show"} the formula language</button>
            {grammar && <pre className="grammar">{grammar}</pre>}
          </>}
          <div className="row">
            <button className="btn primary" disabled={busy || (mode === "idea" ? !idea.trim() : !expr.trim())}>{busy ? "Testing…" : "Test it"}</button>
            {busy && <span className="small muted" role="status">{job.step}…</span>}
          </div>
        </form>
        {(err || job?.error) && <p className="err">{err || job.error}</p>}
      </section>
      {job?.status === "done" && <>
        <h2>{job.results.filter(r => ["keep", "star"].includes(r.verdict)).length} of {job.results.length} work after costs</h2>
        {job.results.map((r, i) => <Result key={i} r={r} i={i} job={job.id}
          flip={x => { setMode("formula"); setExpr(x); scrollTo(0, 0); run(null, x); }} />)}
      </>}
    </>
  );
}
