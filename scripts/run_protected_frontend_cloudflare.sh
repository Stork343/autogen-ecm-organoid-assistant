#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if [ -x "$ROOT_DIR/bin/cloudflared" ]; then
  CLOUDFLARED_BIN="$ROOT_DIR/bin/cloudflared"
elif command -v cloudflared >/dev/null 2>&1; then
  CLOUDFLARED_BIN="$(command -v cloudflared)"
else
  echo "cloudflared is not installed. Install Cloudflare Tunnel first." >&2
  exit 1
fi

HOST="127.0.0.1"
PORT="${FRONTEND_PUBLIC_PORT:-8525}"
HEALTH_URL="http://${HOST}:${PORT}/_stcore/health"

listener_pid() {
  lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null | head -n 1
}

frontend_healthy() {
  curl -fsS "$HEALTH_URL" >/dev/null 2>&1
}

cleanup_stale_listener() {
  local pid
  pid="$(listener_pid || true)"
  if [ -n "$pid" ] && ! frontend_healthy; then
    echo "Removing stale listener on port ${PORT} (PID ${pid})"
    kill "$pid" >/dev/null 2>&1 || true
    sleep 1
  fi
}

cleanup() {
  if [ -n "${APP_PID:-}" ] && kill -0 "$APP_PID" >/dev/null 2>&1; then
    kill "$APP_PID" || true
  fi
}
trap cleanup EXIT

set -a
if [ -f ".env" ]; then
  # shellcheck disable=SC1091
  source ".env"
fi
set +a

if [ "${FRONTEND_REQUIRE_LOGIN:-}" != "true" ] && [ "${FRONTEND_REQUIRE_LOGIN:-}" != "1" ]; then
  echo "FRONTEND_REQUIRE_LOGIN is not enabled. Refusing to expose an unprotected frontend." >&2
  exit 1
fi

if [ -z "${FRONTEND_PASSWORD:-}" ] && [ -z "${FRONTEND_PASSWORD_SHA256:-}" ]; then
  echo "Set FRONTEND_PASSWORD or FRONTEND_PASSWORD_SHA256 in .env before exposing the frontend." >&2
  exit 1
fi

echo "Starting protected frontend locally at http://${HOST}:${PORT}"
cleanup_stale_listener

if frontend_healthy; then
  echo "A healthy frontend is already running on http://${HOST}:${PORT}"
  APP_PID=""
else
# shellcheck disable=SC1091
source ".venv/bin/activate"
streamlit run src/ecm_organoid_agent/frontend.py --server.address "$HOST" --server.port "$PORT" >/tmp/ecm_frontend_public.log 2>&1 &
APP_PID=$!
sleep 3
fi

echo "Opening Cloudflare Tunnel to http://${HOST}:${PORT}"
"$CLOUDFLARED_BIN" tunnel --url "http://${HOST}:${PORT}"
