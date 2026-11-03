import { createRoot } from "react-dom/client";
import { useEffect, useState } from "react";
import "./styles.css";
import { usePoll } from "./api.js";
import WarRoom from "./WarRoom.jsx";
import Track from "./Track.jsx";
import Week from "./Week.jsx";

const PAGES = [
  { id: "room", label: "War room" },
  { id: "track", label: "Track record" },
  { id: "week", label: "This week" },
];

function App() {
  const [page, setPage] = useState(() => location.hash.slice(1) || "room");
  const [overview, error, reload] = usePoll("overview", 4000);
  useEffect(() => {
    const on = () => setPage(location.hash.slice(1) || "room");
    addEventListener("hashchange", on);
    return () => removeEventListener("hashchange", on);
  }, []);
  const go = id => { location.hash = id; setPage(id); };
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
        {page === "track" && <Track />}
        {page === "week" && <Week />}
      </main>
      <footer>Personal mode: this app runs only on your computer. {overview?.disclaimer}</footer>
    </>
  );
}

createRoot(document.getElementById("root")).render(<App />);
