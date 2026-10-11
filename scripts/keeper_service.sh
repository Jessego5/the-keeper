#!/usr/bin/env bash
# keeper_service.sh: keep the Keeper running, as a macOS LaunchAgent.
#
#   scripts/keeper_service.sh install     start now, at every login, and after a crash
#   scripts/keeper_service.sh uninstall   stop it and remove the agent
#   scripts/keeper_service.sh start|stop  load or unload it, keeping the agent file
#   scripts/keeper_service.sh restart     restart the running process
#   scripts/keeper_service.sh status      is it up, which pid, how it last exited
#   scripts/keeper_service.sh logs        tail its log
#   scripts/keeper_service.sh plist       print the agent file it would install
#
# An agent that works while you are away is only that if something keeps it
# running. launchd is the supervisor macOS already has: it starts the Keeper at
# login, restarts it if it dies, and is the one place that decides whether it
# runs, which is what the Night of Thirty-Nine Banners lacked. Three hand-started
# copies ran at once that night. Now a second copy cannot start at all (see
# backend/instance.py), and under launchd one that is refused simply waits and
# takes over when the other stops.
#
# A LaunchAgent, not a LaunchDaemon, on purpose: the Keeper reads YOUR idle time
# and screen lock and posts banners in YOUR session, none of which exist for a
# daemon running as root before anyone logs in. On an always-on Mac mini, turn on
# automatic login for the account it should keep.
#
# Note: ./dev.sh stop kills the process, and launchd will start it again. Stop the
# service with this script instead.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LABEL="local.the-keeper"
AGENT="$HOME/Library/LaunchAgents/$LABEL.plist"
LOGDIR="$HOME/Library/Logs/TheKeeper"
LOG="$LOGDIR/keeper.log"
PORT="${PORT:-8790}"
DOMAIN="gui/$(id -u)"

plist() {
  # PATH is spelled out because launchd starts with almost none, and the Keeper
  # shells out to uvx and npx (MCP servers) and terminal-notifier (banners).
  cat <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array><string>/bin/bash</string><string>$ROOT/run.sh</string></array>
  <key>WorkingDirectory</key><string>$ROOT</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    <key>PORT</key><string>$PORT</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <!-- A Keeper refused because another holds the memory exits at once. Thirty
       seconds between tries keeps that from becoming a spin. -->
  <key>ThrottleInterval</key><integer>30</integer>
  <key>StandardOutPath</key><string>$LOG</string>
  <key>StandardErrorPath</key><string>$LOG</string>
</dict>
</plist>
EOF
}

_loaded() { launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1; }

_require_mac() {
  [ "$(uname)" = "Darwin" ] || { echo "launchd is macOS only; on Linux use docker compose (restart: unless-stopped)."; exit 1; }
}

start() {
  _require_mac
  [ -f "$AGENT" ] || { echo "not installed; run: $0 install"; exit 1; }
  if _loaded; then echo "already running under launchd"; return; fi
  launchctl bootstrap "$DOMAIN" "$AGENT"
  echo "started. the keeper -> http://localhost:$PORT"
}

stop() {
  _require_mac
  if _loaded; then launchctl bootout "$DOMAIN/$LABEL"; echo "stopped"; else echo "not running under launchd"; fi
}

install() {
  _require_mac
  if pgrep -f "uvicorn server:app" >/dev/null && ! _loaded; then
    echo "· a Keeper is already running by hand (pid $(pgrep -f 'uvicorn server:app' | head -1))."
    echo "  The agent will wait until it stops, then take over. ./dev.sh stop ends it."
  fi
  mkdir -p "$LOGDIR" "$(dirname "$AGENT")"
  plist > "$AGENT"
  plutil -lint "$AGENT" >/dev/null
  _loaded && launchctl bootout "$DOMAIN/$LABEL" || true
  launchctl bootstrap "$DOMAIN" "$AGENT"
  echo "installed: $AGENT"
  echo "· starts at login, restarts after a crash. log: $LOG"
  echo "· the keeper -> http://localhost:$PORT"
}

uninstall() {
  _require_mac
  _loaded && launchctl bootout "$DOMAIN/$LABEL" || true
  rm -f "$AGENT"
  echo "removed. memory_store/ is untouched."
}

status() {
  _require_mac
  if ! _loaded; then echo "not running under launchd"; [ -f "$AGENT" ] && echo "(installed: $0 start)" || true; return; fi
  launchctl print "$DOMAIN/$LABEL" | grep -E "^\s*(state|pid|last exit code|runs) =" || true
  if curl -s -o /dev/null --max-time 2 "http://localhost:$PORT/info"; then
    echo "answering on http://localhost:$PORT"
  else
    echo "not answering on :$PORT yet (starting, or waiting for another Keeper to stop; see $0 logs)"
  fi
}

case "${1:-}" in
  install)   install ;;
  uninstall) uninstall ;;
  start)     start ;;
  stop)      stop ;;
  restart)   _require_mac; launchctl kickstart -k "$DOMAIN/$LABEL" && echo "restarted" ;;
  status)    status ;;
  logs)      tail -f "$LOG" ;;
  plist)     plist ;;
  *)         grep '^#' "$0" | sed -n '2,10p' | sed 's/^# \{0,1\}//' ;;
esac
