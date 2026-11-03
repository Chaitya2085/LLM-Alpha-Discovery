import { useEffect, useRef, useState } from "react";
import { usePoll, ago, clock } from "../api.js";

const PLACES = ["overview", "Head office", "Market data", "Model room", "Research lab", "Meeting room",
  "Risk & compliance", "Communications", "Reception"];
const LABEL = { overview: "Whole office", Reception: "Lounge" };

function webglOK() {
  try { const c = document.createElement("canvas"); return !!(c.getContext("webgl2") || c.getContext("webgl")); }
  catch (e) { return false; }
}

export default function Office({ overview }) {
  const box = useRef(null);
  const office = useRef(null);
  const [ready, setReady] = useState(false);
  const [failed, setFailed] = useState(!webglOK());
  const [selected, setSelected] = useState(null);
  const [track] = usePoll("track");
  const [events] = usePoll("events?last=80", selected ? 4000 : 0);

  useEffect(() => {
    if (failed) return undefined;
    let alive = true;
    import("./Office3D.js").then(({ default: Office3D }) => {
      if (!alive || !box.current) return;
      try {
        const dark = matchMedia("(prefers-color-scheme: dark)").matches;
        const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
        office.current = new Office3D(box.current, { onSelect: setSelected, dark, reducedMotion: reduced });
        setReady(true);
      } catch (e) { setFailed(true); }
    }).catch(() => setFailed(true));
    return () => { alive = false; office.current?.dispose(); office.current = null; };
  }, [failed]);

  useEffect(() => {
    if (!ready || !office.current || !overview) return;
    const v = track?.exam?.variants?.["Alpha158 only"];
    const exam = v ? { model: v.book.ann_return, index: v.book.bench_ann_return, verdict: v.verdict.verdict } : null;
    office.current.update({ agents: overview.agents, overview, exam });
  }, [ready, overview, track]);

  if (failed) return <p className="small muted">This browser can't show the 3D office, so here is the list view.</p>;
  const agent = overview?.agents?.find(a => a.name === selected);
  const mine = (events || []).filter(e => e.agent === selected).slice(0, 6);
  return (
    <section className="office" aria-label="3D office">
      <div className="office-bar" role="group" aria-label="Go to a room">
        {PLACES.map(p => (
          <button key={p} className="chip" onClick={() => { setSelected(null); office.current?.flyTo(p); }}>{LABEL[p] || p}</button>
        ))}
      </div>
      <div className="office-stage" ref={box}>
        {!ready && <div className="office-loading">Opening the office…</div>}
        <div className="office-hint small">Drag to turn · scroll or pinch to zoom · click a room or a person</div>
      </div>
      {agent && (
        <aside className="office-panel" aria-label={agent.name}>
          <div className="row" style={{ justifyContent: "space-between" }}>
            <div><span className="eyebrow">{agent.desk}</span><h3>{agent.name}</h3></div>
            <button className="btn" onClick={() => setSelected(null)} aria-label="Close">✕</button>
          </div>
          <p className="small muted">{agent.job}</p>
          <p><span className={`st ${agent.status}`}>{agent.status}</span> {agent.message || "No action yet."}</p>
          {agent.time && <span className="small muted">{ago(agent.time)}</span>}
          <div className="row">
            <button className="btn" onClick={() => office.current?.flyTo(agent.name)}>Go to desk</button>
          </div>
          {mine.length > 0 && <>
            <span className="eyebrow" style={{ marginTop: 6 }}>Recent</span>
            <ul className="office-recent">{mine.map((e, i) => <li key={i}><span className="num muted">{clock(e.time)}</span> {e.message}</li>)}</ul>
          </>}
        </aside>
      )}
    </section>
  );
}
