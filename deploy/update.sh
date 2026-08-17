#!/usr/bin/env bash
# Update the running CKP Gateway to the current origin/main (issue #78).
# Pull, then delegate dependency install, plist re-render, service reload and
# the health probe to install-launchd.sh so template changes take effect.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$REPO_DIR"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "!! working tree is dirty; refusing to update" >&2
  git status --short >&2
  exit 1
fi
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$BRANCH" != "main" ]]; then
  echo "!! checkout is on '$BRANCH', not main; switch to main first" >&2
  exit 1
fi
PREV="$(git rev-parse HEAD)"
git pull --ff-only origin main
echo "==> $PREV -> $(git rev-parse HEAD)"

if ./deploy/install-launchd.sh; then
  PORT="$(set -a; . "$HOME/.config/cyclone/ckp-gateway.env" >/dev/null 2>&1; set +a; printf '%s' "${CKP_SERVER_PORT:-8092}")"
  curl -s --connect-timeout 2 --max-time 5 "http://127.0.0.1:$PORT/revision"
  echo
  echo "==> update complete"
else
  echo "!! gateway unhealthy after update; roll back with:" >&2
  echo "   git reset --hard $PREV && ./deploy/install-launchd.sh" >&2
  exit 1
fi
