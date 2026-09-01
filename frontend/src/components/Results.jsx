import Gauge from './Gauge.jsx';

function band(w) {
  if (!w.ipa) return 'na'; // nothing aligned — word not attempted
  if (w.score >= 0.8) return 'good';
  if (w.score >= 0.5) return 'mid';
  return 'poor';
}

export default function Results({ result, onReset }) {
  const words = result.words || [];
  // Word-segmented "you said" from the alignment; unsaid words show as gaps.
  const saidIpa = words.length
    ? words.map((w) => w.ipa || '·').join(' ')
    : result.ipa || '';
  return (
    <section className="card results">
      <h2 className="results-title">Your results</h2>

      <Gauge score={result.score ?? 0} />

      <div className="ipa-compare">
        <div className="ipa-block">
          <h3>Expected</h3>
          <p className="ipa" lang="und-fonipa">
            {result.expectedIpa || '—'}
          </p>
        </div>
        <div className="ipa-block">
          <h3>You said</h3>
          <p className="ipa" lang="und-fonipa">
            {saidIpa || '—'}
          </p>
        </div>
      </div>

      {words.length > 0 && (
        <div className="word-section">
          <h3>Word by word</h3>
          <ul className="word-list">
            {words.map((w, i) => (
              <li
                key={`${w.word}-${i}`}
                className={`word-chip ${band(w)}`}
                title={`${w.word}: expected /${w.expected || '—'}/, heard /${w.ipa || '—'}/ — ${w.ipa ? Math.round((w.score ?? 0) * 100) + '%' : 'not said'}`}
              >
                <span className="word-text">{w.word}</span>
                <span className="word-expected">/{w.expected || '—'}/</span>
                <span className="word-score">{w.ipa ? `${Math.round((w.score ?? 0) * 100)}%` : 'not said'}</span>
              </li>
            ))}
          </ul>
          <p className="legend">
            <span className="legend-key good" /> good
            <span className="legend-key mid" /> needs work
            <span className="legend-key poor" /> poor
            <span className="legend-key na" /> not said
          </p>
        </div>
      )}

      <button type="button" className="btn" onClick={onReset}>
        Practice again
      </button>
    </section>
  );
}
