// Small helpers for talking to the Python backend (app/server.py).
import { useEffect, useRef, useState } from "react";

export async function api(path, body) {
  const res = await fetch("/api/" + path, body === undefined ? {} : {
    method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
  return data;
}

// Fetch a path now and every `every` ms (0 = once). Returns [data, error, reload].
export function usePoll(path, every = 0) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const tick = useRef(0);
  const load = () => {
    const mine = ++tick.current;
    api(path).then(d => { if (mine === tick.current) { setData(d); setError(null); } })
      .catch(e => { if (mine === tick.current) setError(e.message); });
  };
  useEffect(() => {
    load();
    if (!every) return undefined;
    const id = setInterval(load, every);
    return () => clearInterval(id);
  }, [path, every]); // eslint-disable-line react-hooks/exhaustive-deps
  return [data, error, load];
}

export const pct = (x, d = 1) => (x == null ? "–" : `${(x * 100).toFixed(d)}%`);
export const signed = (x, d = 1) => (x == null ? "–" : `${x > 0 ? "+" : ""}${(x * 100).toFixed(d)}%`);
export const lakh = x => (x == null ? "–" : `₹${x.toFixed(2)}L`);

export function ago(iso) {
  if (!iso) return "never";
  const s = (Date.now() - Date.parse(iso)) / 1000;
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < 86400) return `${Math.round(s / 3600)} h ago`;
  return `${Math.round(s / 86400)} d ago`;
}

export function clock(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  const same = d.toDateString() === new Date().toDateString();
  const t = d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
  return same ? t : `${d.toLocaleDateString(undefined, { day: "numeric", month: "short" })} ${t}`;
}
