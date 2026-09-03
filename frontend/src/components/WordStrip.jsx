import { useEffect, useRef } from 'react';

// Scrolling reader: the passage as a horizontal strip that slides along as
// the backend reports reading progress. currentIndex is FRACTIONAL: 12.5
// means halfway through word 12's phonemes, so the strip slides continuously
// (teleprompter-style) instead of stepping whole words. The word being read
// (floor(currentIndex)) is highlighted; already-read words trail behind.
// Neutral by design — no score colors while recording.
// The track is positioned with a CSS-transitioned transform, NOT
// scrollIntoView: position updates arrive several times a second while
// recording, and interrupting a native smooth-scroll that often makes the
// browser restart the animation endlessly — the strip looks frozen, then
// teleports. A composited transform transition simply retargets mid-flight.
export default function WordStrip({ words, currentIndex, loading }) {
  const stripRef = useRef(null);
  const trackRef = useRef(null);

  useEffect(() => {
    const strip = stripRef.current;
    const track = trackRef.current;
    if (!strip || !track || loading || words.length === 0) return;
    const p = Math.min(Math.max(currentIndex, 0), words.length - 1);
    const i0 = Math.floor(p);
    const i1 = Math.min(i0 + 1, words.length - 1);
    const frac = p - i0;
    const center = (el) => el.offsetLeft + el.offsetWidth / 2;
    const e0 = track.children[i0];
    const e1 = track.children[i1];
    if (!e0 || !e1) return;
    const wordCenter = center(e0) + (center(e1) - center(e0)) * frac;
    track.style.transform = `translateX(${strip.clientWidth / 2 - wordCenter}px)`;
  }, [currentIndex, words.length, loading]);

  const current = Math.floor(currentIndex);
  return (
    <div className="word-strip" ref={stripRef} aria-label="Reading progress">
      {loading ? (
        <p className="strip-loading" aria-live="polite">
          <span className="loading-dot" aria-hidden="true" />
          <span className="loading-dot" aria-hidden="true" />
          <span className="loading-dot" aria-hidden="true" />
          <span>warming up your words…</span>
        </p>
      ) : (
        <div className="word-strip-track" ref={trackRef}>
          {words.map((w, i) => (
            <span
              key={i}
              className={`strip-word ${i < current ? 'spoken' : ''} ${i === current ? 'current' : ''}`}
            >
              {w}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
