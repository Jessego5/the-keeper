#!/usr/bin/env bash
# dev.sh: drive the Keeper for hands-on testing.
#
#   ./dev.sh start     start the server in the background
#   ./dev.sh stop      stop it
#   ./dev.sh restart   stop + start
#   ./dev.sh open      open the UI in your browser
#   ./dev.sh state     show energy / base_score / P(speak) / facts kept
#   ./dev.sh say "..." send a chat message, print the reply
#   ./dev.sh fast      compress the battery so it reaches out in ~seconds
#   ./dev.sh normal    return to real pacing
#   ./dev.sh reset     wipe memory + restart (fresh Keeper)
#   ./dev.sh logs      tail the server log (loop ticks, errors)
#
# The reach-out demo:  ./dev.sh say "..."  then  ./dev.sh fast  then watch the page.

set -euo pipefail
cd "$(dirname "$0")"

PORT=8737
PY=.venv/bin/python
LOG=.keeper.log
URL="http://localhost:$PORT"

_up() { curl -s -o /dev/null --max-time 2 "$URL/" 2>/dev/null; }

start() {
  if _up; then echo "already running at $URL"; return; fi
  $PY -m uvicorn server:app --app-dir backend --port "$PORT" --log-level warning \
    > "$LOG" 2>&1 &
  echo -n "starting"; for _ in 1 2 3 4 5 6 7 8; do sleep 1; _up && break; echo -n .; done
  echo; _up && echo "up at $URL" || { echo "failed, see $LOG"; tail -5 "$LOG"; }
}

stop() { pkill -f "uvicorn server:app" 2>/dev/null && echo "stopped" || echo "not running"; }

case "${1:-}" in
  start)   start ;;
  stop)    stop ;;
  restart) stop; sleep 1; start ;;
  open)    open "$URL" ;;
  state)   curl -s "$URL/state" | $PY -m json.tool ;;
  say)     curl -s -X POST "$URL/chat" -H 'content-type: application/json' \
             -d "{\"message\": $($PY -c 'import json,sys; print(json.dumps(sys.argv[1]))' "${2:-}")}" \
             | $PY -c 'import sys,json; d=json.load(sys.stdin); print("keeper:", d["reply"]); print("   [", d.get("water_state"), "| tools:", d.get("used_tools"), "| score:", d.get("score"), "]")' ;;
  fast)    curl -s -X POST "$URL/config" -H 'content-type: application/json' \
             -d '{"speed":800,"cooldown_min":20}' | $PY -m json.tool
           echo "battery compressed, watch the page; reset with ./dev.sh normal" ;;
  normal)  curl -s -X POST "$URL/config" -H 'content-type: application/json' \
             -d '{"speed":1,"cooldown_min":600}' | $PY -m json.tool ;;
  reset)   stop; sleep 1; rm -f memory_store/facts.jsonl; echo "memory wiped"; start ;;
  logs)    tail -f "$LOG" ;;
  *)       grep '^#' "$0" | sed 's/^# \{0,1\}//' ;;
esac
