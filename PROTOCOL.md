# Native — internal wire contracts

Three services: `frontend` (React), `backend` (Go), `inference` (Python/FastAPI).
All audio on the wire is **PCM signed 16-bit little-endian, 16000 Hz, mono** ("pcm16").

## Client ↔ backend (WebSocket, `/ws`)

Client → server:

- Text frame JSON `{"type":"start","text":"<the passage, ≤200 words>"}` — opens a recording session.
- Binary frames — raw pcm16 chunks, sent continuously while recording.
- Text frame JSON `{"type":"stop"}` — recording finished; server begins final scoring.

Server → client (all text frames, JSON):

- `{"type":"ready"}` — ack of `start`, safe to stream audio.
- `{"type":"partial","ipa":"h ə l oʊ","processedSec":12.5,"wordIndex":4}` — interim transcription of a completed ~2.5 s chunk (appended chunk, not cumulative). `wordIndex` (optional) comes from the cumulative-transcript alignment, the ground truth: it MAY decrease to recalibrate a run-ahead reader — apply it absolutely.
- `{"type":"progress","wordIndex":6}` — reader-position update from a "peek": a trailing ~2.5 s window transcribed every ~0.33 s of new audio (denoise off, own worker, stale peeks dropped). Peeks are not part of the scored transcript; the frame carries no `ipa`. Hints only: apply as a lower bound (never move backward on progress frames).
- `{"type":"finishing"}` — the server detected the last word of the passage and is ending the session automatically (≈0.7 s grace after detection). Client should stop the mic and show the scoring state; `final` follows. The client may still send `stop` first if the user finishes early or stops mid-passage.
- `{"type":"final","ipa":"...","expectedIpa":"...","score":0.87,"words":[{"word":"hello","expected":"həloʊ","ipa":"həlo","score":0.9}]}` — full-utterance result. `score` is 0..1 (1 = perfect match). `words[].ipa` is the aligned actual segment (may be empty).
- `{"type":"error","message":"..."}` — fatal for the session; client should reset UI.

## Backend ↔ inference (HTTP, base from env `INFERENCE_URL`, default `http://localhost:9000`)

- `GET /health` → `{"status":"ok","device":"cuda"|"cpu","model":"facebook/wav2vec2-lv-60-espeak-cv-ft"}`
- `POST /transcribe` — body: raw pcm16 bytes (`Content-Type: application/octet-stream`) → `{"ipa":"..."}`. IPA uses the espeak alphabet produced by the wav2vec2 model; words separated by spaces, phonemes may be space-separated. Optional query `?denoise=0|1` overrides the service's DENOISE default (reader peeks pass `denoise=0` — position-finding doesn't need spectral gating and it dominates peek latency).
- `POST /score` — JSON `{"text":"the passage","actualIpa":"..."}` → `{"expectedIpa":"...","score":0.87,"words":[{"word":"hello","expected":"həloʊ","ipa":"həlo","score":0.9}]}`. Target dialect: General American (espeak-ng `en-us`). Distance: panphon feature edit distance, normalized to 0..1.
- `POST /align` — JSON `{"text":"the passage","actualIpa":"...","mode":"prefix"|"window","fromWord":0,"maxWord":null}` → `{"wordIndex":4}`; `-1` when nothing trustworthy matched. `prefix` (default): actualIpa is the cumulative transcript, fitting alignment; `maxWord` (when ≥0) hard-caps the answer — chunks arrive every ~2.5 s, so the position can physically advance only so far between them (backend passes last chunk position +8). `window`: actualIpa is a short trailing window (~2.5 s), aligned locally against the band of words `[fromWord, fromWord+8)` with a per-token drift penalty from the band start; returns the end of the best-matching region, `-1` if the match is too poor (silence/noise). `maxWord` hard-caps the answer — regions ending past it aren't candidates; readers realistically skip ≤2 words between updates, so farther matches are duplicate-word coincidences. Cheap (expected phonemes are cached per text); called per chunk and per peek.

## Processing strategy (backend)

- While recording: every ≥2.5 s of new pcm16 → `POST /transcribe` → emit `partial`.
- On `stop`: transcribe the remaining tail, then re-transcribe the **full** audio once and `POST /score` with the original text + full IPA → emit `final`.
