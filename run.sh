#!/usr/bin/env bash
# run.sh — one command to run the Keeper locally.
#   ./run.sh            start on :8790  (UI at /, dashboard at /dashboard)
# Creates the venv and installs deps on first run. Needs backend/.env with
# OPENAI_API_KEY (copy backend/.env.example). Runs fully offline on the stub if
# no key is present — just less interesting.
set -euo pipefail
cd "$(dirname "$0")"
PORT="${PORT:-8790}"

if [ ! -d .venv ]; then
  echo "· creating venv + installing deps (first run)…"
  python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r requirements.txt
fi

[ -f backend/.env ] || echo "· note: backend/.env missing — running on the stub. Add OPENAI_API_KEY for the real Keeper."

echo "· the keeper → http://localhost:${PORT}   (dashboard: http://localhost:${PORT}/dashboard)"
exec .venv/bin/python -m uvicorn server:app --app-dir backend --port "${PORT}"
