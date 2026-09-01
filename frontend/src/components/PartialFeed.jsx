export default function PartialFeed({ partials }) {
  if (partials.length === 0) return null;
  return (
    <details className="partials">
      <summary>Live transcription ({partials.length})</summary>
      <ul className="partial-list">
        {partials.map((p, i) => (
          <li key={i} className="partial-line">
            <span className="partial-sec">
              {typeof p.processedSec === 'number' ? `${p.processedSec.toFixed(1)}s` : ''}
            </span>
            <span className="ipa" lang="und-fonipa">
              {p.ipa}
            </span>
          </li>
        ))}
      </ul>
    </details>
  );
}
