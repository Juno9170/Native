#!/usr/bin/env bash
# Boot the full Native dev environment: Docker stack (backend + inference)
# plus the Vite dev server. Run from Git Bash:  ./dev.sh
set -euo pipefail
cd "$(dirname "$0")"

say() { printf '\033[1;36m[dev]\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m[dev] ERROR:\033[0m %s\n' "$*" >&2; }

wait_for_docker() {
  for _ in $(seq 1 24); do
    docker info >/dev/null 2>&1 && return 0
    sleep 5
  done
  return 1
}

# --- 1. Docker daemon -------------------------------------------------------
if ! docker info >/dev/null 2>&1; then
  say "Docker daemon not reachable; trying to start Docker Desktop..."
  launched=0
  for p in \
    "/c/Program Files/Docker/Docker/Docker Desktop.exe" \
    "$LOCALAPPDATA/Programs/DockerDesktop/Docker Desktop.exe"; do
    if [ -f "$p" ]; then
      "$p" >/dev/null 2>&1 &
      disown
      launched=1
      break
    fi
  done
  if [ "$launched" -eq 0 ]; then
    err "Docker Desktop not found at the usual install paths."
    err "Install/start it manually, then re-run ./dev.sh"
    exit 1
  fi
  say "Waiting for Docker daemon (up to 2 min)..."
  if ! wait_for_docker; then
    err "Docker daemon did not come up in time."
    err "Open Docker Desktop, wait for it to finish starting, then re-run ./dev.sh"
    exit 1
  fi
fi
say "Docker daemon is up."

# --- 2. Backend + inference stack -------------------------------------------
say "Starting backend + inference containers..."
docker compose up -d backend inference

say "Waiting for the inference model to load on GPU..."
ready=0
for _ in $(seq 1 48); do
  if docker compose exec -T inference python -c \
      "import urllib.request;urllib.request.urlopen('http://localhost:9000/health',timeout=2)" \
      >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 5
done
if [ "$ready" -eq 0 ]; then
  err "Inference service did not become healthy in time."
  err "Check logs with: docker compose logs inference"
  exit 1
fi
say "Inference is ready."

if ! curl -s -o /dev/null http://localhost:8080/; then
  err "Backend is not responding on :8080. Check: docker compose logs backend"
  exit 1
fi
say "Backend is ready on :8080."

# --- 3. Vite dev server (foreground; Ctrl+C stops it) -----------------------
if ! command -v npm >/dev/null 2>&1; then
  err "npm not found. Install Node.js, then re-run ./dev.sh"
  exit 1
fi
say "Starting Vite dev server on http://localhost:5173 ..."
cd frontend
exec npm run dev
