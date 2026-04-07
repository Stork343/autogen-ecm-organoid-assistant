#!/bin/zsh
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WORKSPACE_DIR="${ECM_ORGANOID_WORKSPACE:-$HOME/ECM-Organoid-Research-Desk}"
PORT="${ECM_ORGANOID_BROWSER_PORT:-8516}"
LOG_DIR="$WORKSPACE_DIR/logs"
LOG_FILE="$LOG_DIR/browser_launcher.log"
PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"
STREAMLIT_BIN="$PROJECT_DIR/.venv/bin/streamlit"
FRONTEND_PATH="$PROJECT_DIR/src/ecm_organoid_agent/frontend.py"
HEALTH_URL="http://127.0.0.1:${PORT}/_stcore/health"

mkdir -p "$LOG_DIR"

listener_pid() {
  lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null | head -n 1
}

frontend_healthy() {
  /usr/bin/curl -fsS "$HEALTH_URL" >/dev/null 2>&1
}

cleanup_stale_listener() {
  local pid
  pid="$(listener_pid || true)"
  if [[ -n "$pid" ]] && ! frontend_healthy; then
    echo "Removing stale listener on port ${PORT} (PID ${pid})" >>"$LOG_FILE"
    kill "$pid" >/dev/null 2>&1 || true
    sleep 1
  fi
}

if [[ ! -x "$PYTHON_BIN" || ! -x "$STREAMLIT_BIN" ]]; then
  /usr/bin/osascript -e 'display alert "ECM Organoid Browser Launcher" message "找不到项目虚拟环境。请先在项目目录中准备好 .venv。" as critical'
  exit 1
fi

ECM_ORGANOID_PROJECT_DIR="$WORKSPACE_DIR" "$PYTHON_BIN" - <<'PY' >/dev/null
from pathlib import Path
import os
from ecm_organoid_agent.workspace import ensure_workspace

ensure_workspace(Path(os.environ["ECM_ORGANOID_PROJECT_DIR"]))
PY

cleanup_stale_listener

if ! frontend_healthy; then
  nohup env ECM_ORGANOID_PROJECT_DIR="$WORKSPACE_DIR" \
    "$STREAMLIT_BIN" run "$FRONTEND_PATH" \
    --server.headless true \
    --server.port "$PORT" \
    --browser.gatherUsageStats false \
    >>"$LOG_FILE" 2>&1 &
fi

for _ in {1..120}; do
  if frontend_healthy; then
    /usr/bin/open "http://127.0.0.1:${PORT}/"
    exit 0
  fi
  sleep 0.5
done

/usr/bin/osascript -e "display alert \"ECM Organoid Browser Launcher\" message \"本地服务未能在预期时间内启动。请检查 $LOG_FILE\" as critical"
exit 1
