# Native

A pronunciation trainer. Type up to 200 words (English), press record, read the
passage aloud, and get an IPA transcription of your speech plus a per-word
pronunciation score against a General American reference.

## Architecture

Three services — see `PROTOCOL.md` for the wire contracts.

- **frontend** (React + Vite + react-router, nginx) — text input, mic capture
  via AudioWorklet (pcm16, 16 kHz), WebSocket client, results view.
- **backend** (Go) — WebSocket endpoint, session management, streams audio
  chunks to the inference service and relays partial/final results.
- **inference** (Python + FastAPI) — denoise (noisereduce) → Wav2Vec2 phoneme
  model (`facebook/wav2vec2-lv-60-espeak-cv-ft`) → IPA; scoring via phonemizer
  (espeak-ng, `en-us`) + panphon feature edit distance.

## Run

Dev environment (Docker stack + Vite dev server on http://localhost:5173):

```bash
./dev.sh        # or double-click dev.bat on Windows
./dev.sh stop   # shut everything down
```

Requires Docker with the NVIDIA container toolkit for GPU (falls back to CPU
without the `deploy` section). For the production build instead:

```bash
docker compose up --build
```

Open http://localhost:3000. First start downloads the wav2vec2 model
(~1.2 GB) into the `hf-cache` volume — `GET` the backend's `/readyz` until it
reports ok.

## Roadmap

- [ ] Accent type & strength classifier over transcribed phoneme sequences
      (custom lightweight model; training data: CommonVoice accent labels).
- [ ] More target dialects beyond General American.
