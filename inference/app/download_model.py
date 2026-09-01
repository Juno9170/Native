"""Pre-download the wav2vec2 model into the HF cache.

Run at container start (not build time) so the image stays small and the
model is fetched once per container volume/cache.
"""

from huggingface_hub import snapshot_download

from app.transcribe import MODEL_NAME

if __name__ == "__main__":
    snapshot_download(MODEL_NAME)
