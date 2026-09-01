# AGENTS.md

## Project

Native — pronunciation trainer. Type ≤200 English words, record yourself
reading them, get IPA transcription + panphon-based pronunciation scores.

## Layout

- `frontend/` — React 18 + Vite + react-router v6, plain CSS. Verify changes
  with `npm run build`.
- `backend/` — Go 1.22 (module `native/backend`), gorilla/websocket, stdlib
  HTTP client to inference. Keep deps minimal.
- `inference/` — Python 3.11 FastAPI. torch/transformers +
  `facebook/wav2vec2-lv-60-espeak-cv-ft`, phonemizer (espeak-ng `en-us`),
  panphon, noisereduce.
- `PROTOCOL.md` — the wire contracts between all three services. **Any change
  to messages, endpoints, or payload shapes must update PROTOCOL.md and all
  affected services in the same change.**
- `docker-compose.yml` — frontend:3000, backend:8080 (internal), inference:9000
  (internal). GPU reservation for inference.

## Conventions

- Audio on every wire is pcm16 s16le 16 kHz mono.
- Target dialect: General American (espeak-ng `en-us`). Normalize stress marks
  out of both expected and actual IPA before scoring — see
  `inference/app/scoring.py`.
- Design language: "Handmade Warmth" — palette #F5F0E1 #D4C4A8 #C67B5C #B5651D
  #8B4513 #6B7B3C #9C8B7A, organic asymmetric border-radius, grain overlay,
  ease-out transitions. Score colors: good #6B7B3C, mid #B5651D, poor #8B4513.
