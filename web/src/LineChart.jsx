import { useMemo, useRef, useState } from "react";

// A small line chart with a crosshair tooltip. series: [{ key, label, color, values, dash?, width? }]
export default function LineChart({ dates, series, fmt = v => v.toFixed(2), height = 260, label = "Chart" }) {
  const [hover, setHover] = useState(null);
  const ref = useRef(null);
  const W = 760, H = height, L = 56, R = 120, T = 12, B = 30;
  const { x, y, ticks, marks } = useMemo(() => {
    const all = series.flatMap(s => s.values).filter(v => v != null);
    let lo = Math.min(...all), hi = Math.max(...all);
    const pad = (hi - lo) * 0.06 || 1; lo -= pad; hi += pad;
    const n = dates.length;
    const x = i => L + (i / Math.max(1, n - 1)) * (W - L - R);
    const y = v => T + (1 - (v - lo) / (hi - lo)) * (H - T - B);
    const raw = (hi - lo) / 4, mag = Math.pow(10, Math.floor(Math.log10(raw))), step = [1, 2, 2.5, 5, 10].map(k => k * mag).find(k => k >= raw);
    const ticks = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) ticks.push(+v.toFixed(6));
    const marks = [];
    const byYear = (Date.parse(dates[n - 1]) - Date.parse(dates[0])) / 864e5 > 730;   // over 2 years: label years
    dates.forEach((d, i) => {
      const key = byYear ? d.slice(0, 4) : d.slice(0, 7);
      if (i === 0 || key !== (byYear ? dates[i - 1].slice(0, 4) : dates[i - 1].slice(0, 7))) marks.push([i, byYear ? key : new Date(d).toLocaleDateString(undefined, { month: "short" })]);
    });
    return { x, y, ticks, marks: byYear ? marks : marks.filter((_, k) => k % 2 === 0) };
  }, [dates, series, H]); // eslint-disable-line react-hooks/exhaustive-deps
  const path = vals => vals.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)} ${y(v).toFixed(1)}`).join("");
  const move = e => {
    const r = ref.current.getBoundingClientRect();
    const i = Math.round((((e.clientX - r.left) / r.width) * W - L) / (W - L - R) * (dates.length - 1));
    if (i < 0 || i >= dates.length) return setHover(null);
    setHover({ i, left: (x(i) / W) * r.width });
  };
  const last = dates.length - 1;
  let prev = -1e9;
  const ends = series.map(s => ({ s, v: s.values[last] })).sort((a, b) => b.v - a.v);
  return (
    <div style={{ position: "relative" }}>
      <svg ref={ref} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={label} onPointerMove={move} onPointerLeave={() => setHover(null)}>
        {ticks.map(v => <g key={v}><line x1={L} x2={W - R} y1={y(v)} y2={y(v)} stroke="var(--line)" /><text x={L - 8} y={y(v) + 4} textAnchor="end" fontSize="11">{fmt(v)}</text></g>)}
        {marks.map(([i, t]) => <text key={i} x={x(i)} y={H - 10} fontSize="11">{t}</text>)}
        {series.map(s => <path key={s.key} d={path(s.values)} fill="none" stroke={s.color} strokeWidth={s.width || 2} strokeDasharray={s.dash} strokeLinejoin="round" />)}
        {ends.map(({ s, v }) => {
          const ly = Math.max(y(v) + 4, prev + 15); prev = ly;
          return <g key={s.key}><circle cx={x(last)} cy={y(v)} r="4" fill={s.color} stroke="var(--card)" strokeWidth="2" />
            <text x={x(last) + 10} y={ly} fontSize="11.5" style={{ fill: "var(--ink)" }}>{fmt(v)}</text></g>;
        })}
        {hover && <g><line x1={x(hover.i)} x2={x(hover.i)} y1={T} y2={H - B} stroke="var(--muted)" strokeDasharray="3 3" />
          {series.map(s => <circle key={s.key} cx={x(hover.i)} cy={y(s.values[hover.i])} r="4" fill={s.color} stroke="var(--card)" strokeWidth="2" />)}</g>}
        <rect x={L} y={T} width={W - L - R} height={H - T - B} fill="transparent" />
      </svg>
      {hover && (
        <div className="tip" style={{ left: Math.min(hover.left + 12, 480), top: 6 }}>
          <strong className="num">{dates[hover.i]}</strong>
          {series.map(s => <div key={s.key}><span>{s.label}</span><span className="num">{fmt(s.values[hover.i])}</span></div>)}
        </div>
      )}
    </div>
  );
}

export function Legend({ series }) {
  return <div className="legend">{series.map(s => <span key={s.key}><i style={{ background: s.color }} />{s.label}</span>)}</div>;
}
