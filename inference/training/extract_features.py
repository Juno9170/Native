"""Extract accent-classifier features from the Speech Accent Archive.

Every SAA speaker reads the SAME elicitation paragraph, so the expected IPA
(espeak, en-us) is identical across samples and every deviation in the
wav2vec2 transcription is accent signal. For each sample this script:

  1. decodes the parquet audio bytes to pcm16 16 kHz mono via ffmpeg
  2. transcribes with the app's own wav2vec2 pipeline (denoise OFF — the
     archive recordings are clean, and spectral gating dominates runtime)
  3. aligns actual vs. expected phonemes with the app's alignment
     (scoring._align, full deletion cost)
  4. emits a deviation-token sequence: M:<p> match, S:<e>><a> substitution,
     D:<p> deletion, I:<p> insertion — the classifier's input
  5. records mean_dev (mean panphon cost over matched pairs) and the
     insertion/deletion fractions — the raw material for accent STRENGTH
     calibration

Output: /training/features.jsonl — one line per usable sample:
  {"id", "accent", "mean_dev", "frac_ins", "frac_del", "tokens": [...]}

Run inside the inference container:
  docker compose exec -T inference python /training/extract_features.py
Idempotent: samples already present in features.jsonl are skipped.
"""

import glob
import json
import os
import subprocess
import sys

import pandas as pd

sys.path.insert(0, "/app")
from app import scoring, transcribe  # noqa: E402

DATA_DIR = "/training/data"
OUT_PATH = "/training/features.jsonl"

# The SAA elicitation paragraph (accent.gmu.edu) — 69 words, same for all.
PARAGRAPH = (
    "Please call Stella. Ask her to bring these things with her from the "
    "store: Six spoons of fresh snow peas, five thick slabs of blue cheese, "
    "and maybe a snack for her brother Bob. We also need a small plastic "
    "snake and a big toy frog for the kids. She can scoop these things into "
    "three red bags, and we will go meet her Wednesday at the train station."
)

# Labels that are not accents.
DROP_ACCENTS = {"synthesized"}

# A usable sample must transcribe to at least this fraction of the expected
# token count (broken/truncated recordings produce far less).
MIN_LEN_FRACTION = 0.5


def decode_pcm16(audio_bytes: bytes) -> bytes:
    """Arbitrary audio bytes -> raw s16le 16 kHz mono via ffmpeg."""
    proc = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", "pipe:0",
            "-f", "s16le", "-ac", "1", "-ar", "16000", "pipe:1",
        ],
        input=audio_bytes,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode()[:200])
    return proc.stdout


def deviation_tokens(expected, actual, alts):
    """Alignment -> deviation token sequence + deviation/gap stats."""
    pairs = scoring._align(expected, actual, alts, del_cost=1.0)
    tokens, costs = [], []
    n_ins = n_del = 0
    for e, a in pairs:
        if e is not None and a is not None:
            c = scoring._positional_cost(e, expected, alts, actual[a])
            costs.append(c)
            if c == 0.0:
                tokens.append(f"M:{expected[e]}")
            else:
                tokens.append(f"S:{expected[e]}>{actual[a]}")
        elif e is not None:
            n_del += 1
            tokens.append(f"D:{expected[e]}")
        else:
            n_ins += 1
            tokens.append(f"I:{actual[a]}")
    n = max(len(pairs), 1)
    mean_dev = sum(costs) / len(costs) if costs else 1.0
    return tokens, mean_dev, n_ins / n, n_del / n


def main() -> None:
    transcribe.load_model()
    _, _, expected_tokens, alts = scoring._expected_for(PARAGRAPH)
    print(f"expected: {len(expected_tokens)} phonemes", flush=True)

    done = set()
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH) as fh:
            done = {json.loads(line)["id"] for line in fh}
    print(f"already extracted: {len(done)}", flush=True)

    frames = [
        pd.read_parquet(f)
        for f in sorted(glob.glob(f"{DATA_DIR}/original-*.parquet"))
        + sorted(glob.glob(f"{DATA_DIR}/other/original-*.parquet"))
    ]
    df = pd.concat(frames, ignore_index=True)
    df = df[~df["accent"].isin(DROP_ACCENTS)]
    print(f"samples: {len(df)}", flush=True)

    out = open(OUT_PATH, "a")
    n_ok = n_skip = 0
    for i, row in enumerate(df.itertuples(), 1):
        if row.id in done:
            continue
        try:
            pcm = decode_pcm16(row.audio["bytes"])
            ipa = transcribe.transcribe_pcm16(pcm, denoise=False)
            actual = scoring.normalize_ipa(ipa).split()
            if len(actual) < MIN_LEN_FRACTION * len(expected_tokens):
                n_skip += 1
                print(f"[{i}] {row.id} SKIP len={len(actual)}", flush=True)
                continue
            tokens, mean_dev, frac_ins, frac_del = deviation_tokens(
                expected_tokens, actual, alts
            )
            out.write(
                json.dumps(
                    {
                        "id": row.id,
                        "accent": row.accent,
                        "mean_dev": round(mean_dev, 4),
                        "frac_ins": round(frac_ins, 4),
                        "frac_del": round(frac_del, 4),
                        "tokens": tokens,
                    }
                )
                + "\n"
            )
            out.flush()
            n_ok += 1
        except Exception as exc:  # corrupt audio etc. — keep going
            n_skip += 1
            print(f"[{i}] {row.id} ERROR {exc!r}", flush=True)
        if i % 25 == 0:
            print(f"[{i}/{len(df)}] ok={n_ok} skip={n_skip}", flush=True)
    out.close()
    print(f"DONE ok={n_ok} skip={n_skip}", flush=True)


if __name__ == "__main__":
    main()
