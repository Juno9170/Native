// Semicircular gauge for the overall score (0..1).
export default function Gauge({ score }) {
  const pct = Math.round(score * 100);
  const color = score >= 0.8 ? 'var(--olive)' : score >= 0.5 ? 'var(--rust)' : 'var(--saddle)';
  return (
    <div className="gauge" role="img" aria-label={`Overall score ${pct} percent`}>
      <svg viewBox="0 0 200 112" className="gauge-svg">
        <path
          d="M 14 100 A 86 86 0 0 1 186 100"
          fill="none"
          stroke="var(--sand)"
          strokeWidth="16"
          strokeLinecap="round"
        />
        <path
          d="M 14 100 A 86 86 0 0 1 186 100"
          fill="none"
          stroke={color}
          strokeWidth="16"
          strokeLinecap="round"
          pathLength="100"
          strokeDasharray={`${pct} 100`}
          className="gauge-arc"
        />
      </svg>
      <div className="gauge-number" style={{ color }}>
        {pct}
        <span className="gauge-pct">%</span>
      </div>
    </div>
  );
}
