"""Speech -> IPA transcription via facebook/wav2vec2-lv-60-espeak-cv-ft.

The model emits IPA in the espeak alphabet. Its tokenizer uses "|" as the
word delimiter, which we normalize to a plain space, so the output is
space-separated phonemes with space-separated words.
"""

import os

import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

MODEL_NAME = "facebook/wav2vec2-lv-60-espeak-cv-ft"
SAMPLE_RATE = 16000

# Passages are up to ~90 s, which is a lot of activations for the large
# wav2vec2 encoder (esp. on CPU), so audio is decoded in fixed-length
# chunks. No overlap is used: CTC frames are 20 ms, so boundary cuts are
# rare, and overlapping windows would risk duplicated phonemes.
CHUNK_SECONDS = 20

# Shorter than 0.25 s carries no usable speech; treat as silence.
MIN_SAMPLES = SAMPLE_RATE // 4

# Non-stationary denoising (spectral gating) before inference. Toggle with
# DENOISE=0/1, default on.
DENOISE = os.environ.get("DENOISE", "1") == "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor: Wav2Vec2Processor | None = None
model: Wav2Vec2ForCTC | None = None


def load_model() -> None:
    """Load processor + model once at startup and move to the device."""
    global processor, model
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()


@torch.no_grad()
def _decode_chunk(audio: np.ndarray) -> str:
    inputs = processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    logits = model(inputs.input_values.to(device)).logits
    pred_ids = logits.argmax(dim=-1)
    text = processor.batch_decode(pred_ids)[0]
    return text.replace("|", " ").strip()


def transcribe_pcm16(data: bytes) -> str:
    """Raw pcm16 (s16le, 16 kHz, mono) bytes -> espeak-alphabet IPA string."""
    samples = np.frombuffer(data, dtype="<i2")
    if samples.size < MIN_SAMPLES:
        return ""
    audio = samples.astype(np.float32) / 32768.0
    if DENOISE:
        import noisereduce as nr

        audio = nr.reduce_noise(y=audio, sr=SAMPLE_RATE, stationary=False)
    chunk = CHUNK_SECONDS * SAMPLE_RATE
    parts = [
        _decode_chunk(audio[start : start + chunk])
        for start in range(0, len(audio), chunk)
    ]
    return " ".join(p for p in parts if p)
