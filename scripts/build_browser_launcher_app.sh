#!/bin/zsh
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
APP_PATH="$PROJECT_DIR/dist/ECM Organoid Browser Launcher.app"
SCRIPT_PATH="$PROJECT_DIR/scripts/launch_browser_server.sh"

/usr/bin/osacompile \
  -o "$APP_PATH" \
  -e "on run" \
  -e "do shell script quoted form of \"$SCRIPT_PATH\"" \
  -e "end run"

echo "Built: $APP_PATH"
