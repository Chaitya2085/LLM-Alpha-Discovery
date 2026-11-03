import { useState } from "react";
import { api, usePoll, ago, clock } from "./api.js";

const DESKS = ["Head office", "Market data", "Model room", "Risk & compliance", "Communications", "Research lab"];
const STATUS_LABEL = { ok: "ok", warning: "warning", stop: "stopped", idle: "idle" };
const DOT = { ok: "var(--ok)", warning: "var(--warn)", stop: "var(--stop)", idle: "var(--idle)" };
const initials = n => n.split(" ").map(w => w[0]).join("").slice(0, 2).toUpperCase();

function Strip({ o, reload }) {
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState(null);
  const start = async (download, weekly) => {
    setBusy(true); setMsg(null);
    try { await api("run", { download, weekly }); setMsg("Team started. Watch the desks update."); reload(); }
    catch (e) { setMsg(e.message); }
    setBusy(false);
  };
  const lr = o?.last_run || {};
  return (
    <section className="strip" aria-label="Status">
      <div className="stat"><span className="eyebrow">Data through</span><b>{o?.as_of || "–"}</b><span className="small muted">NSE top 200</span></div>
      <div className="stat"><span className="eyebrow">Market mood</span><b>{o?.regime || "–"}</b><span className="small muted">from the Nifty 200</span></div>
      <div className="stat"><span className="eyebrow">Last run</span><b>{o?.running ? "running…" : (STATUS_LABEL[lr.status] || "never")}</b>
        <span className="small muted">{o?.running ? "the team is working" : ago(lr.finished)}</span></div>
      <div className="stat"><span className="eyebrow">Ledger</span><b>{o?.ledger_weeks ?? "–"} week{o?.ledger_weeks === 1 ? "" : "s"}</b><span className="small muted">predictions sealed</span></div>
      <div className="runbox">
        <button className="btn primary" disabled={busy || o?.running} onClick={() => start(false, false)}
          title="Run the team on the data already on this computer">Run the team</button>
        <button className="btn" disabled={busy || o?.running} onClick={() => start(true, false)}
          title="Download the latest NSE files first (after the 3:30 pm IST close)">Download &amp; run</button>
        {msg && <span className="small muted" role="status">{msg}</span>}
      </div>
    </section>
  );
}

function Floor({ agents, running }) {
  return (
    <section className="floor" aria-label="The team">
      <div className="row" style={{ justifyContent: "space-between" }}>
        <h2>The floor</h2>
        <span className="small muted">Each desk shows the agent's latest action</span>
      </div>
      <div className="desks">
        {DESKS.map(d => (
          <div className="desk" key={d}>
            <span className="desk-name">{d}</span>
            {agents.filter(a => a.desk === d).map(a => (
              <div key={a.name} className={`agent ${running ? "working" : ""}`} title={a.job}>
                <span className="avatar" style={{ "--dot": DOT[a.status] }} aria-hidden="true">{initials(a.name)}</span>
                <div>
                  <div className="agent-name">{a.name}<span className={`st ${a.status}`}>{STATUS_LABEL[a.status]}</span></div>
                  <p className="agent-msg">{a.message || a.job}</p>
                  {a.time && <span className="small muted num">{ago(a.time)}</span>}
                </div>
              </div>
            ))}
          </div>
        ))}
      </div>
    </section>
  );
}

function Approvals({ reload }) {
  const [cards, , load] = usePoll("approvals", 5000);
  const [err, setErr] = useState(null);
  const pending = (cards || []).filter(c => c.status === "pending");
  const recent = (cards || []).filter(c => c.status !== "pending").slice(0, 3);
  const answer = async (id, approve) => {
    setErr(null);
    try { await api(`approvals/${id}`, { approve }); load(); reload(); } catch (e) { setErr(e.message); }
  };
  return (
    <div className="panel">
      <h3>The Chief Strategist asks</h3>
      {pending.length === 0 && <p className="small muted">Nothing waiting for you. The next research round is asked for once a month.</p>}
      {pending.map(c => (
        <div className="card-ask" key={c.id}>
          <strong>{c.question}</strong>
          <span className="small muted">{c.agent} · {c.detail} · asked {ago(c.asked_at)}</span>
          <div className="row">
            <button className="btn primary" onClick={() => answer(c.id, true)}>Approve</button>
            <button className="btn danger" onClick={() => answer(c.id, false)}>Not now</button>
          </div>
          <span className="small muted">An approved job runs the next time the team runs.</span>
        </div>
      ))}
      {recent.map(c => (
        <p className="small muted" key={c.id}>{c.status === "approved" ? "Approved" : "Declined"}: {c.question}{c.outcome ? ` (${c.outcome})` : ""}</p>
      ))}
      {err && <p className="small" style={{ color: "var(--stop)" }}>{err}</p>}
    </div>
  );
}

function Ask() {
  const [q, setQ] = useState("");
  const [a, setA] = useState(null);
  const [busy, setBusy] = useState(false);
  const send = async e => {
    e.preventDefault();
    if (!q.trim()) return;
    setBusy(true);
    try { setA((await api("ask", { question: q, private: true })).answer); } catch (err) { setA(err.message); }
    setBusy(false);
  };
  return (
    <div className="panel">
      <h3>Ask the Chief Strategist</h3>
      <form className="ask-form" onSubmit={send}>
        <input value={q} onChange={e => setQ(e.target.value)} placeholder="e.g. why is the status a warning?" aria-label="Your question" />
        <button className="btn primary" disabled={busy}>{busy ? "…" : "Ask"}</button>
      </form>
      {a && <div className="answer" role="status">{a}</div>}
      <span className="small muted">Answers come only from the team's log and journal.</span>
    </div>
  );
}

function Feed() {
  const [priv, setPriv] = useState(false);
  const [n, setN] = useState(20);
  const [events] = usePoll(`events?last=${n}&private=${priv}`, 4000);
  return (
    <section className="panel" aria-label="Activity">
      <div className="row" style={{ justifyContent: "space-between" }}>
        <h2>What your agents just did</h2>
        <label className="toggle"><input type="checkbox" checked={priv} onChange={e => setPriv(e.target.checked)} /> show stock names</label>
      </div>
      {!events?.length && <p className="muted">No activity yet. Press "Run the team".</p>}
      <ul className="feed">
        {(events || []).map((e, i) => (
          <li key={i} className={`${e.status} ${e.private ? "private" : ""}`}>
            <span className="t">{clock(e.time)}</span><span className="who">{e.agent}</span><span className="msg">{e.message}</span>
          </li>
        ))}
      </ul>
      {events?.length >= n && <button className="btn" onClick={() => setN(n + 40)}>Show earlier activity</button>}
    </section>
  );
}

export default function WarRoom({ overview, reload }) {
  return (
    <>
      <Strip o={overview} reload={reload} />
      <div className="room">
        <Floor agents={overview?.agents || []} running={overview?.running} />
        <div className="side"><Approvals reload={reload} /><Ask /></div>
      </div>
      <Feed />
    </>
  );
}
