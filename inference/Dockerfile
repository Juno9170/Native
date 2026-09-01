FROM python:3.11-slim

# espeak-ng: phonemizer backend for /score; ffmpeg: audio tooling utility.
RUN apt-get update \
    && apt-get install -y --no-install-recommends espeak-ng ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 9000

# Download the model at container start (NOT build time), then serve.
CMD ["sh", "-c", "python -m app.download_model && uvicorn app.main:app --host 0.0.0.0 --port 9000"]
