#!/usr/bin/env bash
# Update the running CKP Gateway to the current origin/main (issue #78).
# Mirrors the Cyclone-Dashboard update flow: pull, reinstall deps, restart
# the LaunchAgent, verify /health and print /revision.
set -euo pipefail

LABEL="com.cyclone.ckp-gateway"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$HOME/.config/cyclone/ckp-gateway.env"
GUI_DOMAIN="gui/$(id -u)"

cd "$REPO_DIR"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "!! working tree is dirty; refusing to update" >&2
  git status --short >&2
  exit 1
fi
PREV="$(git rev-parse HEAD)"
git pull --ff-only
echo "==> $PREV -> $(git rev-parse HEAD)"

if [[ -x "$REPO_DIR/.venv/bin/pip" ]]; then
  "$REPO_DIR/.venv/bin/pip" install --quiet -e "$REPO_DIR"
else
  uv pip install --quiet --python "$REPO_DIR/.venv/bin/python" -e "$REPO_DIR"
fi

launchctl kickstart -k "$GUI_DOMAIN/$LABEL"

PORT="$(sed -n 's/^CKP_SERVER_PORT=//p' "$ENV_FILE" | tail -1)"
PORT="${PORT:-8092}"
echo "==> waiting for /health on 127.0.0.1:$PORT"
for _ in $(seq 1 30); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    curl -s "http://127.0.0.1:$PORT/revision"
    echo
    echo "==> update complete ($GUI_DOMAIN/$LABEL)"
    exit 0
  fi
  sleep 1
done
echo "!! gateway unhealthy after update; roll back with:" >&2
echo "   git reset --hard $PREV && .venv/bin/pip install -q -e . && launchctl kickstart -k $GUI_DOMAIN/$LABEL" >&2
exit 1
