import { useEffect, useRef } from 'react';

// Scrolling reader: the passage as a horizontal strip that slides along as
// the backend reports reading progress (wordIndex = last word matched in the
// speech so far). Neutral by design — no score colors while recording.
// The track is positioned with a CSS-transitioned transform, NOT
// scrollIntoView: position updates arrive every ~0.3 s while recording, and
// interrupting a native smooth-scroll that often makes the browser restart
// the animation endlessly — the strip looks frozen, then teleports. A
// composited transform transition simply retargets mid-flight.
export default function WordStrip({ words, currentIndex }) {
  const stripRef = useRef(null);
  const trackRef = useRef(null);

  useEffect(() => {
    const strip = stripRef.current;
    const track = trackRef.current;
    if (!strip || !track || words.length === 0) return;
    const i = Math.min(Math.max(currentIndex, 0), words.length - 1);
    const el = track.children[i];
    if (!el) return;
    const target = strip.clientWidth / 2 - (el.offsetLeft + el.offsetWidth / 2);
    track.style.transform = `translateX(${target}px)`;
  }, [currentIndex, words.length]);

  return (
    <div className="word-strip" ref={stripRef} aria-label="Reading progress">
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
