import { useCallback, useEffect, useRef, useState } from 'react';

// Wire format: PCM signed 16-bit little-endian, 16000 Hz, mono.
const TARGET_RATE = 16000;

function resample(input, srcRate, dstRate) {
  if (srcRate === dstRate) return input;
  const ratio = srcRate / dstRate;
  const outLen = Math.max(1, Math.floor(input.length / ratio));
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const pos = i * ratio;
    const idx = Math.floor(pos);
    const frac = pos - idx;
    const a = input[idx];
    const b = idx + 1 < input.length ? input[idx + 1] : a;
    out[i] = a + (b - a) * frac;
  }
  return out;
}

function floatToPcm16(input) {
  const buf = new ArrayBuffer(input.length * 2);
  const view = new DataView(buf);
  for (let i = 0; i < input.length; i++) {
    const s = Math.max(-1, Math.min(1, input[i]));
    view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return buf;
}

function wsUrl() {
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
  return `${proto}://${window.location.host}/ws`;
}

// Statuses: idle -> connecting (mic + ws + start sent) -> recording (got ready)
//           -> scoring (stop sent, awaiting final) -> done | error
export function usePronunciationSession() {
  const [status, setStatus] = useState('idle');
  const [partials, setPartials] = useState([]);
  const [wordIndex, setWordIndex] = useState(-1); // reader strip position
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [elapsed, setElapsed] = useState(0);

  const wsRef = useRef(null);
  const ctxRef = useRef(null);
  const streamRef = useRef(null);
  const nodeRef = useRef(null);
  const gainRef = useRef(null);
  const readyRef = useRef(false);
  const stoppedRef = useRef(false);
  const pendingRef = useRef([]); // pcm chunks captured before "ready"
  const timerRef = useRef(null);
  const startTimeRef = useRef(0);

  const stopTimer = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  // The clock starts when the reader goes live (first word tracked), not on
  // "ready" — warm-up time isn't the user's reading time.
  const startTimer = useCallback(() => {
    if (timerRef.current) return;
    startTimeRef.current = Date.now();
    timerRef.current = setInterval(() => {
      setElapsed(Math.floor((Date.now() - startTimeRef.current) / 1000));
    }, 250);
  }, []);

  const closeAudio = useCallback(() => {
    if (nodeRef.current) {
      nodeRef.current.port.onmessage = null;
      nodeRef.current.disconnect();
      nodeRef.current = null;
    }
    if (gainRef.current) {
      gainRef.current.disconnect();
      gainRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (ctxRef.current) {
      ctxRef.current.close().catch(() => {});
      ctxRef.current = null;
    }
  }, []);

  const closeWs = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.onmessage = null;
      wsRef.current.onerror = null;
      wsRef.current.onclose = null;
      if (wsRef.current.readyState <= 1) wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  const cleanupAll = useCallback(() => {
    stopTimer();
    closeAudio();
    closeWs();
    readyRef.current = false;
    stoppedRef.current = false;
    pendingRef.current = [];
  }, [stopTimer, closeAudio, closeWs]);

  useEffect(() => cleanupAll, [cleanupAll]);

  const fail = useCallback(
    (message) => {
      cleanupAll();
      setError(message);
      setStatus('error');
    },
    [cleanupAll],
  );

  const handleMessage = useCallback(
    (ev) => {
      let msg;
      try {
        msg = JSON.parse(ev.data);
      } catch {
        return;
      }
      switch (msg.type) {
        case 'ready': {
          readyRef.current = true;
          const ws = wsRef.current;
          if (ws && ws.readyState === WebSocket.OPEN) {
            for (const chunk of pendingRef.current) ws.send(chunk);
          }
          pendingRef.current = [];
          setStatus('recording');
          break;
        }
        case 'partial':
          setPartials((p) => [...p, msg]);
          if (typeof msg.wordIndex === 'number' && msg.wordIndex >= 0) {
            // Chunk alignments are ground truth and may recalibrate BACKWARD
            // to fix a run-ahead reader.
            startTimer();
            setWordIndex(msg.wordIndex);
          }
          break;
        case 'progress':
          // Peek alignments are hints: never move the strip backward.
          if (typeof msg.wordIndex === 'number' && msg.wordIndex >= 0) {
            startTimer();
            setWordIndex((cur) => Math.max(cur, msg.wordIndex));
          }
          break;
        case 'finishing':
          // Server detected the last word and is wrapping up: stop the mic
          // and show the scoring state; "final" arrives shortly.
          stopTimer();
          stoppedRef.current = true;
          closeAudio();
          setStatus('scoring');
          break;
        case 'final':
          stopTimer();
          closeAudio();
          closeWs();
          setResult(msg);
          setStatus('done');
          break;
        case 'error':
          fail(msg.message || 'The server reported an error.');
          break;
        default:
          break;
      }
    },
    [stopTimer, startTimer, closeAudio, closeWs, fail],
  );

  const start = useCallback(
    async (text) => {
      setError(null);
      setResult(null);
      setPartials([]);
      setWordIndex(-1);
      setElapsed(0);
      setStatus('connecting');
      readyRef.current = false;
      stoppedRef.current = false;
      pendingRef.current = [];

      let stream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
        });
      } catch {
        setStatus('error');
        setError(
          'Microphone access was denied. Allow microphone permission in your browser, then try again.',
        );
        return;
      }
      streamRef.current = stream;

      const ws = new WebSocket(wsUrl());
      wsRef.current = ws;
      ws.onopen = () => {
        ws.send(JSON.stringify({ type: 'start', text }));
      };
      ws.onmessage = handleMessage;
      ws.onerror = () => {
        fail('Could not reach the scoring server. Is the backend running?');
      };
      ws.onclose = () => {
        // Unexpected close mid-session (final arrives as a message, then we close deliberately).
        setStatus((s) => {
          if (s === 'connecting' || s === 'recording' || s === 'scoring') {
            cleanupAll();
            setError('The connection to the server was lost. Please try again.');
            return 'error';
          }
          return s;
        });
      };

      try {
        const ctx = new AudioContext();
        ctxRef.current = ctx;
        await ctx.audioWorklet.addModule('/pcm-worklet.js');
        const source = ctx.createMediaStreamSource(stream);
        const node = new AudioWorkletNode(ctx, 'pcm-forwarder');
        nodeRef.current = node;
        node.port.onmessage = (ev) => {
          if (stoppedRef.current) return;
          const pcm = floatToPcm16(resample(ev.data, ctx.sampleRate, TARGET_RATE));
          const sock = wsRef.current;
          if (readyRef.current && sock && sock.readyState === WebSocket.OPEN) {
            sock.send(pcm);
          } else {
            pendingRef.current.push(pcm);
          }
        };
        // Keep the node pulling audio without playing the mic back.
        const gain = ctx.createGain();
        gain.gain.value = 0;
        gainRef.current = gain;
        source.connect(node);
        node.connect(gain);
        gain.connect(ctx.destination);
      } catch {
        fail('Audio capture is not available in this browser (AudioWorklet required).');
      }
    },
    [handleMessage, fail, cleanupAll],
  );

  const stop = useCallback(() => {
    if (status === 'connecting') {
      // Never got "ready" — treat as cancel.
      cleanupAll();
      setStatus('idle');
      return;
    }
    if (status !== 'recording') return;
    stopTimer();
    stoppedRef.current = true;
    closeAudio();
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'stop' }));
      setStatus('scoring');
    } else {
      fail('The connection to the server was lost. Please try again.');
    }
  }, [status, stopTimer, closeAudio, cleanupAll, fail]);

  const reset = useCallback(() => {
    cleanupAll();
    setPartials([]);
    setWordIndex(-1);
    setResult(null);
    setError(null);
    setElapsed(0);
    setStatus('idle');
  }, [cleanupAll]);

  return { status, partials, wordIndex, result, error, elapsed, start, stop, reset };
}
