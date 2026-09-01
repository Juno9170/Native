import { useEffect, useRef } from 'react';

export default function About() {
  const heroRef = useRef(null);

  // Cheap parallax: drift the hero text slightly against scroll.
  useEffect(() => {
    const onScroll = () => {
      if (heroRef.current) {
        heroRef.current.style.transform = `translateY(${window.scrollY * 0.15}px)`;
      }
    };
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  return (
    <div className="page about">
      <section className="about-hero">
        <div ref={heroRef}>
          <h1>About Native</h1>
          <p className="lede">
            Native is a pronunciation trainer that listens like a patient teacher.
          </p>
        </div>
      </section>

      <section className="card about-card">
        <h2>How it works</h2>
        <ol className="steps">
          <li>
            <strong>Write.</strong> Type up to 200 words of English — a paragraph you want to
            master, a presentation intro, anything.
          </li>
          <li>
            <strong>Read.</strong> Press record and read the passage aloud. Your microphone audio is
            streamed as raw PCM over a WebSocket to the scoring backend — nothing is stored.
          </li>
          <li>
            <strong>Review.</strong> A wav2vec2 phoneme model transcribes what you actually said
            into IPA. That transcription is compared, phoneme by phoneme, against the expected
            General American pronunciation of your text.
          </li>
        </ol>
        <p>
          Each word gets a score from a feature-based edit distance between expected and actual
          phonemes, so you can see exactly which sounds need practice — not just a vague "good job."
        </p>
      </section>
    </div>
  );
}
