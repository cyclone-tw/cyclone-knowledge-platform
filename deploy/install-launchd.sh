#!/usr/bin/env bash
# Install the CKP Gateway as a macOS LaunchAgent (issue #78).
#
# Pattern follows Cyclone-Dashboard deploy/install-launchd.sh: login-session
# LaunchAgent, RunAtLoad + KeepAlive, host-specific values in an env file
# outside the repo. No absolute host paths here (AGENTS.md §7).
#
# Usage:  [PORT=8092] [LOG_DIR=...] ./deploy/install-launchd.sh
set -euo pipefail

LABEL="com.cyclone.ckp-gateway"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-8092}"
LOG_DIR="${LOG_DIR:-$HOME/Library/Logs/ckp-gateway}"
ENV_FILE="$HOME/.config/cyclone/ckp-gateway.env"
PLIST_PATH="$HOME/Library/LaunchAgents/$LABEL.plist"
TEMPLATE="$REPO_DIR/deploy/$LABEL.plist.template"

# Both paths are injected into the plist XML via sed; refuse characters that
# would corrupt the XML, the sed replacement, or the embedded shell command.
for p in "$REPO_DIR" "$LOG_DIR"; do
  [[ "$p" =~ ^[A-Za-z0-9/._-]+$ ]] || { echo "!! unsafe characters in path: $p" >&2; exit 1; }
done

mkdir -p "$LOG_DIR" "$HOME/Library/LaunchAgents" "$(dirname "$ENV_FILE")"

if [[ ! -x "$REPO_DIR/.venv/bin/python" ]]; then
  echo "==> creating venv"
  # Homebrew python's ensurepip is broken on some hosts; fall back to uv.
  if ! python3 -m venv "$REPO_DIR/.venv" 2>/dev/null; then
    rm -rf "$REPO_DIR/.venv"
    command -v uv >/dev/null 2>&1 || { echo "!! python3 -m venv failed and uv is not installed" >&2; exit 1; }
    uv venv "$REPO_DIR/.venv"
  fi
fi
echo "==> installing ckp (editable, runtime deps only)"
if [[ -x "$REPO_DIR/.venv/bin/pip" ]]; then
  "$REPO_DIR/.venv/bin/pip" install --quiet -e "$REPO_DIR"
else
  uv pip install --quiet --python "$REPO_DIR/.venv/bin/python" -e "$REPO_DIR"
fi

if [[ ! -f "$ENV_FILE" ]]; then
  echo "==> writing default env file: $ENV_FILE"
  cat > "$ENV_FILE" <<EOF
# CKP Gateway runtime config (sourced by the LaunchAgent with \`set -a\`).
# Only valid CKP_<SECTION>_<KEY> names are allowed here: the config loader
# rejects unknown CKP_* keys with a hard error and the service will not start.
CKP_BUNDLE_ROOT=$HOME/Cyclone-System/cyclone-wiki
CKP_SERVER_HOST=127.0.0.1
CKP_SERVER_PORT=$PORT
CKP_PROFILE_EXPECTED_VERSION=cyclone-profile-v1
EOF
  chmod 600 "$ENV_FILE"
else
  echo "==> keeping existing env file: $ENV_FILE"
fi

echo "==> rendering $PLIST_PATH"
sed -e "s|__REPO_DIR__|$REPO_DIR|g" \
    -e "s|__LOG_DIR__|$LOG_DIR|g" \
    "$TEMPLATE" > "$PLIST_PATH"

GUI_DOMAIN="gui/$(id -u)"
launchctl bootout "$GUI_DOMAIN" "$PLIST_PATH" 2>/dev/null || true
launchctl bootstrap "$GUI_DOMAIN" "$PLIST_PATH"
launchctl kickstart -k "$GUI_DOMAIN/$LABEL"

# The env file owns the real port; resolve it exactly the way the LaunchAgent
# does (set -a + source), so export/quoted forms all parse correctly.
HEALTH_PORT="$(set -a; . "$ENV_FILE" >/dev/null 2>&1; set +a; printf '%s' "${CKP_SERVER_PORT:-$PORT}")"
echo "==> waiting for /health on 127.0.0.1:$HEALTH_PORT"
for _ in $(seq 1 30); do
  if curl -sf --connect-timeout 2 --max-time 5 "http://127.0.0.1:$HEALTH_PORT/health" >/dev/null 2>&1; then
    curl -s --connect-timeout 2 --max-time 5 "http://127.0.0.1:$HEALTH_PORT/health"
    echo
    echo "==> installed: $GUI_DOMAIN/$LABEL (logs in $LOG_DIR)"
    exit 0
  fi
  sleep 1
done
echo "!! gateway never became healthy; last stderr lines:" >&2
tail -20 "$LOG_DIR/gateway.err.log" >&2 || true
exit 1
