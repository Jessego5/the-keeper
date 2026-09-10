#!/usr/bin/env bash
# run.sh: one command to run the Keeper locally.
#   ./run.sh            start on :8790  (UI at /, dashboard at /dashboard)
# Creates the venv and installs deps on first run. Needs backend/.env with
# OPENAI_API_KEY (copy backend/.env.example). Runs fully offline on the stub if
# no key is present: just less interesting.
set -euo pipefail
cd "$(dirname "$0")"
PORT="${PORT:-8790}"

if [ ! -d .venv ]; then
  echo "· creating venv + installing deps (first run)…"
  python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r requirements.txt
fi

[ -f backend/.env ] || echo "· note: backend/.env missing, running on the stub. Add OPENAI_API_KEY for the real Keeper."

# macOS: build the Keeper's own notification app so proactive banners carry its
# icon (best-effort: never blocks startup; falls back to plain terminal-notifier
# or osascript). Skips if already built or if terminal-notifier isn't installed.
if [ "$(uname)" = "Darwin" ] && [ ! -d backend/notify/Keeper.app ]; then
  if command -v terminal-notifier >/dev/null 2>&1; then
    echo "· building the Keeper's notification app…"
    ./scripts/build_keeper_notifier.sh >/dev/null 2>&1 || echo "  (skipped, banners will use the fallback icon)"
  else
    echo "· tip: 'brew install terminal-notifier' for native banners with the Keeper's icon."
  fi
fi

echo "· the keeper → http://localhost:${PORT}   (dashboard: http://localhost:${PORT}/dashboard)"
exec .venv/bin/python -m uvicorn server:app --app-dir backend --port "${PORT}"
