import { createRoot } from "react-dom/client";
import { useEffect, useState } from "react";
import "./styles.css";
import { usePoll } from "./api.js";
import WarRoom from "./WarRoom.jsx";
import Track from "./Track.jsx";
import Week from "./Week.jsx";
import Stocks from "./Stocks.jsx";
import Lab from "./Lab.jsx";
import Strategies from "./Strategies.jsx";

const PAGES = [
  { id: "room", label: "War room" },
  { id: "week", label: "This week" },
  { id: "stocks", label: "Stocks" },
  { id: "lab", label: "Research lab" },
  { id: "strategies", label: "Strategies" },
  { id: "track", label: "Track record" },
];
const parse = () => { const [p, ...rest] = location.hash.slice(1).split("/"); return [p || "room", rest.join("/") || null]; };

function App() {
  const [[page, arg], setRoute] = useState(parse);
  const [preset, setPreset] = useState(null);
  const [overview, error, reload] = usePoll("overview", 4000);
  useEffect(() => {
    const on = () => { setRoute(parse()); scrollTo(0, 0); };
    addEventListener("hashchange", on);
    return () => removeEventListener("hashchange", on);
  }, []);
  const go = h => { location.hash = h; };
  return (
    <>
      <header className="top">
        <div className="top-in">
          <span className="brand">LLM Alpha Discovery</span>
          <nav className="tabs" aria-label="Pages">
            {PAGES.map(p => (
              <button key={p.id} className="tab" aria-current={page === p.id ? "page" : undefined} onClick={() => go(p.id)}>
                {p.label}
                {p.id === "room" && overview?.pending ? <span className="badge-count" title="decisions waiting">{overview.pending}</span> : null}
              </button>
            ))}
          </nav>
        </div>
      </header>
      <main>
        {error && <div className="err">Can't reach the backend ({error}). Is <code>python run_app.py</code> still running?</div>}
        {page === "room" && <WarRoom overview={overview} reload={reload} />}
        {page === "week" && <Week go={go} />}
        {page === "stocks" && <Stocks arg={arg} go={go} />}
        {page === "lab" && <Lab preset={preset} />}
        {page === "strategies" && <Strategies test={e => { setPreset(e); go("lab"); }} />}
        {page === "track" && <Track />}
      </main>
      <footer>Personal mode: this app runs only on your computer. {overview?.disclaimer}</footer>
    </>
  );
}

createRoot(document.getElementById("root")).render(<App />);
