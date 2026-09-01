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
- `{"type":"partial","ipa":"h ə l oʊ","processedSec":12.5,"wordIndex":4}` — interim transcription of audio so far (appended chunk, not cumulative). `wordIndex` (optional) is the 0-based index of the last word matched in the speech so far, for the scrolling reader; omitted when alignment is unavailable or nothing has matched.
- `{"type":"final","ipa":"...","expectedIpa":"...","score":0.87,"words":[{"word":"hello","expected":"həloʊ","ipa":"həlo","score":0.9}]}` — full-utterance result. `score` is 0..1 (1 = perfect match). `words[].ipa` is the aligned actual segment (may be empty).
- `{"type":"error","message":"..."}` — fatal for the session; client should reset UI.

## Backend ↔ inference (HTTP, base from env `INFERENCE_URL`, default `http://localhost:9000`)

- `GET /health` → `{"status":"ok","device":"cuda"|"cpu","model":"facebook/wav2vec2-lv-60-espeak-cv-ft"}`
- `POST /transcribe` — body: raw pcm16 bytes (`Content-Type: application/octet-stream`) → `{"ipa":"..."}`. IPA uses the espeak alphabet produced by the wav2vec2 model; words separated by spaces, phonemes may be space-separated.
- `POST /score` — JSON `{"text":"the passage","actualIpa":"..."}` → `{"expectedIpa":"...","score":0.87,"words":[{"word":"hello","expected":"həloʊ","ipa":"həlo","score":0.9}]}`. Target dialect: General American (espeak-ng `en-us`). Distance: panphon feature edit distance, normalized to 0..1.
- `POST /align` — JSON `{"text":"the passage","actualIpa":"<partial IPA so far>"}` → `{"wordIndex":4}`; `-1` when nothing has matched yet. Cheap (expected phonemes are cached per text); called once per partial chunk.

## Processing strategy (backend)

- While recording: every ≥2.5 s of new pcm16 → `POST /transcribe` → emit `partial`.
- On `stop`: transcribe the remaining tail, then re-transcribe the **full** audio once and `POST /score` with the original text + full IPA → emit `final`.
