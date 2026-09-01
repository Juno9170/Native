import { useEffect, useRef } from 'react';

// Scrolling reader: the passage as a horizontal strip that slides along as
// the backend reports reading progress (wordIndex = last word matched in the
// speech so far). Neutral by design — no score colors while recording.
export default function WordStrip({ words, currentIndex }) {
  const trackRef = useRef(null);

  useEffect(() => {
    const track = trackRef.current;
    if (!track || words.length === 0) return;
    const i = Math.min(Math.max(currentIndex, 0), words.length - 1);
    const el = track.children[i];
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' });
    }
  }, [currentIndex, words.length]);

  return (
    <div className="word-strip" aria-label="Reading progress">
      <div className="word-strip-track" ref={trackRef}>
        {words.map((w, i) => (
          <span
            key={i}
            className={`strip-word ${i < currentIndex ? 'spoken' : ''} ${i === currentIndex ? 'current' : ''}`}
          >
            {w}
          </span>
        ))}
      </div>
    </div>
  );
}
