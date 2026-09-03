#!/usr/bin/env bash
# Native dev environment.
#
#   ./dev.sh        start everything (Docker stack + Vite dev server)
#   ./dev.sh stop   stop the dev server and the containers
#
# Windows: double-click dev.bat, or run `dev` from cmd/PowerShell.
set -euo pipefail
cd "$(dirname "$0")"

cyan=$'\033[1;36m'; green=$'\033[1;32m'; red=$'\033[1;31m'
dim=$'\033[2m'; reset=$'\033[0m'
say() { printf '%s[dev]%s %s\n' "$cyan" "$reset" "$*"; }
ok()  { printf '%s[ ok ]%s %s\n' "$green" "$reset" "$*"; }
err() { printf '%s[fail]%s %s\n' "$red" "$reset" "$*" >&2; }

# --- helpers ----------------------------------------------------------------

wait_for_docker() {
  for _ in $(seq 1 24); do
    docker info >/dev/null 2>&1 && return 0
    sleep 5
  done
  return 1
}

# Print "pid name" of the process listening on port $1, or nothing.
port_owner() {
  powershell -NoProfile -Command \
    "Get-NetTCPConnection -LocalPort $1 -State Listen -ErrorAction SilentlyContinue |
     Select-Object -First 1 | ForEach-Object {
       \$p = Get-Process -Id \$_.OwningProcess -ErrorAction SilentlyContinue
       \"\$(\$_.OwningProcess) \$(\$p.ProcessName)\" }" 2>/dev/null | tr -d '\r' || true
}

# Free port $1 if a node/npm process is squatting on it; fail on anything else.
free_port() {
  local owner pid name
  owner=$(port_owner "$1")
  [ -z "$owner" ] && return 0
  pid=${owner%% *}; name=${owner#* }
  case "$name" in
    node|npm)
      say "Port $1 held by a leftover $name process (pid $pid); killing it."
      powershell -NoProfile -Command "Stop-Process -Id $pid -Force" >/dev/null 2>&1 || true
      sleep 1
      ;;
    *)
      err "Port $1 is in use by '$name' (pid $pid). Free it and re-run."
      exit 1
      ;;
  esac
}

wait_for_url() { # $1 url, $2 attempts, $3 sleep-secs
  for _ in $(seq 1 "$2"); do
    curl -s -o /dev/null "$1" && return 0
    sleep "$3"
  done
  return 1
}

inference_ready() {
  docker compose exec -T inference python -c \
    "import urllib.request;urllib.request.urlopen('http://localhost:9000/health',timeout=2)" \
    >/dev/null 2>&1
}

# --- stop -------------------------------------------------------------------

if [ "${1:-}" = "stop" ]; then
  owner=$(port_owner 5173) || true
  if [ -n "${owner:-}" ]; then
    say "Stopping dev server (pid ${owner%% *})..."
    powershell -NoProfile -Command "Stop-Process -Id ${owner%% *} -Force" >/dev/null 2>&1 || true
  fi
  say "Stopping containers..."
  docker compose stop backend inference >/dev/null 2>&1 || true
  ok "Stopped."
  exit 0
fi

# --- 1. Docker daemon ---------------------------------------------------------

if docker info >/dev/null 2>&1; then
  ok "Docker daemon is up."
else
  say "Docker daemon not reachable; starting Docker Desktop..."
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
  printf '%s' "$dim"
  if ! wait_for_docker; then
    printf '%s' "$reset"
    err "Docker daemon did not come up within 2 minutes."
    err "Open Docker Desktop, let it finish starting, then re-run ./dev.sh"
    exit 1
  fi
  printf '%s' "$reset"
  ok "Docker daemon is up."
fi

# --- 2. Containers ------------------------------------------------------------

say "Starting backend + inference containers..."
docker compose up -d backend inference >/dev/null

if inference_ready; then
  ok "Inference ready (model already loaded)."
else
  say "Loading wav2vec2 model on GPU ${dim}(first boot after restart takes ~1 min)${reset}"
  ready=0
  for _ in $(seq 1 48); do
    inference_ready && { ready=1; break; }
    sleep 5
  done
  [ "$ready" -eq 1 ] || { err "Inference never became healthy — see: docker compose logs inference"; exit 1; }
  ok "Inference ready."
fi

wait_for_url http://localhost:8080/ 6 5 \
  || { err "Backend not responding on :8080 — see: docker compose logs backend"; exit 1; }
ok "Backend ready on :8080."

# --- 3. Vite dev server -------------------------------------------------------

command -v npm >/dev/null 2>&1 || { err "npm not found. Install Node.js first."; exit 1; }
free_port 5173

cd frontend
npm run dev &
VITE_PID=$!
# Ctrl+C / kill: take the whole npm→node→vite tree down, no orphans.
cleanup() { taskkill //PID "$VITE_PID" //T //F >/dev/null 2>&1 || true; }
trap cleanup INT TERM EXIT

wait_for_url http://localhost:5173/ 12 1 \
  || { err "Vite did not start — check the output above."; exit 1; }
ok "Vite ready."

printf '\n%s┌──────────────────────────────────────────┐%s\n' "$green" "$reset"
printf '%s│%s   Native dev environment is up           %s│%s\n' "$green" "$reset" "$green" "$reset"
printf '%s│%s                                          %s│%s\n' "$green" "$reset" "$green" "$reset"
printf '%s│%s   app        http://localhost:5173       %s│%s\n' "$green" "$reset" "$green" "$reset"
printf '%s│%s   backend    http://localhost:8080       %s│%s\n' "$green" "$reset" "$green" "$reset"
printf '%s│%s                                          %s│%s\n' "$green" "$reset" "$green" "$reset"
printf '%s│%s   stop all:  ./dev.sh stop               %s│%s\n' "$green" "$reset" "$green" "$reset"
printf '%s└──────────────────────────────────────────┘%s\n\n' "$green" "$reset"

cmd //c start "" http://localhost:5173 >/dev/null 2>&1 || true

wait "$VITE_PID"
