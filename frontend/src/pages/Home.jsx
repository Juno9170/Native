import { useMemo, useState } from 'react';
import { usePronunciationSession } from '../hooks/usePronunciationSession.js';
import Results from '../components/Results.jsx';
import PartialFeed from '../components/PartialFeed.jsx';
import WordStrip from '../components/WordStrip.jsx';

const MAX_WORDS = 200;

function countWords(text) {
  return text.trim().split(/\s+/).filter(Boolean).length;
}

function fmtElapsed(sec) {
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `${m}:${String(s).padStart(2, '0')}`;
}

export default function Home() {
  const [text, setText] = useState('');
  const [warned, setWarned] = useState(false);
  const { status, partials, wordIndex, result, error, elapsed, start, stop, reset } =
    usePronunciationSession();

  const words = useMemo(() => countWords(text), [text]);
  const wordList = useMemo(() => text.trim().split(/\s+/).filter(Boolean), [text]);
  const canRecord = words >= 1 && words <= MAX_WORDS;

  const onTextChange = (e) => {
    const value = e.target.value;
    if (countWords(value) > MAX_WORDS) {
      setWarned(true);
      return; // hard stop at 200 words
    }
    setWarned(false);
    setText(value);
  };

  const busy = status === 'connecting' || status === 'recording' || status === 'scoring';

  if (status === 'done' && result) {
    return (
      <div className="page">
        <Results result={result} onReset={reset} />
      </div>
    );
  }

  return (
    <div className="page">
      <section className="hero">
        <h1>Hear yourself the way others do.</h1>
        <p>
          Type a short passage in English, read it aloud, and get a phoneme-level score for every
          word.
        </p>
      </section>

      <section className="card">
        <label className="field-label" htmlFor="passage">
          Your passage
          <span className={`word-counter ${warned ? 'over' : ''}`}>
            {words}/{MAX_WORDS} words
          </span>
        </label>
        {busy ? (
          // wordIndex is fractional progress (12.5 = halfway through word
          // 12): the strip highlights the word being read and slides
          // continuously; already-read words trail behind in stone. Until
          // the first word is underlined the reader isn't live yet — show
          // the warm-up state instead.
          <WordStrip
            words={wordList}
            currentIndex={wordIndex}
            loading={status === 'connecting' || wordIndex < 0}
          />
        ) : (
          <>
            <textarea
              id="passage"
              className="passage"
              rows="6"
              placeholder="Type or paste up to 200 words of English here…"
              value={text}
              onChange={onTextChange}
            />
            <p className="hint">
              English only, please — {MAX_WORDS} words max.
              {warned && <strong> That is over the {MAX_WORDS}-word limit.</strong>}
            </p>
          </>
        )}

        {error && (
          <div className="error-box" role="alert">
            <p>{error}</p>
            <button type="button" className="btn btn-quiet" onClick={reset}>
              Start over
            </button>
          </div>
        )}

        <div className="controls">
          {status === 'recording' ? (
            <>
              <button type="button" className="btn btn-record recording" onClick={stop}>
                <span className="pulse-dot" aria-hidden="true" />
                Stop
              </button>
              <span className="elapsed" aria-live="polite">
                {fmtElapsed(elapsed)}
              </span>
            </>
          ) : status === 'scoring' ? (
            <p className="scoring" aria-live="polite">
              <span className="spinner" aria-hidden="true" />
              Scoring your pronunciation…
            </p>
          ) : (
            <button
              type="button"
              className="btn btn-record"
              disabled={!canRecord || status === 'connecting'}
              onClick={() => start(text)}
            >
              {status === 'connecting' ? 'Connecting…' : '● Record'}
            </button>
          )}
        </div>

        <PartialFeed partials={partials} />
      </section>
    </div>
  );
}
