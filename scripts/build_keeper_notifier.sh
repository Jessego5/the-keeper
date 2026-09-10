#!/usr/bin/env bash
# Build the Keeper's notification apps: rebranded copies of terminal-notifier so
# the Keeper's own icon and name show as the PRIMARY badge on native notifications
# (not terminal-notifier's). Reproducible: regenerate anytime from the PNGs.
#
# A banner's primary badge is the icon inside the .app bundle, fixed at build time,
# so there is no argument that changes it per notification. Wearing the register
# therefore means one bundle per register:
#
#   notify/Keeper.app            the default pose, and the fallback whenever a
#                                state bundle is missing
#   notify/frozen/The Keeper.app  the cold
#   notify/tidal/The Keeper.app   the water moving
#   notify/turning/The Keeper.app the turn, the ice going out
#
# Each state bundle gets its OWN id, so macOS keeps their badges apart, but they
# are all named "The Keeper" on disk and live in a directory named for the state.
# That is deliberate: macOS labels its notification-permission prompt and the
# Notification Centre entry from the bundle's filename, not CFBundleDisplayName,
# so a bundle literally called Keeper-turning.app introduces itself to the person
# as "Keeper-turning". One companion, several faces: the name must not leak.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STATIC="$ROOT/backend/static"
NOTIFY="$ROOT/backend/notify"
SRC="$(ls -d "$(brew --prefix)"/Cellar/terminal-notifier/*/terminal-notifier.app 2>/dev/null | head -1)"

[ -d "$SRC" ] || { echo "terminal-notifier not installed (brew install terminal-notifier)"; exit 1; }

# name:source png:bundle id suffix, the default first, so a partial build still
# leaves a working Keeper.app behind.
BUILDS=(
  "Keeper.app:keeper.png:notifier"
  "frozen/The Keeper.app:keeper_frozen.png:frozen"
  "tidal/The Keeper.app:keeper_tidal.png:tidal"
  "turning/The Keeper.app:keeper_turning.png:turning"
)

built=0
for entry in "${BUILDS[@]}"; do
  APP="${entry%%:*}"; rest="${entry#*:}"
  PNG_NAME="${rest%%:*}"; SUFFIX="${rest#*:}"
  PNG="$STATIC/$PNG_NAME"
  OUT="$NOTIFY/$APP"

  # A missing pose is not fatal: notifier.py falls back to Keeper.app for any
  # register whose bundle was never built.
  [ -f "$PNG" ] || { echo "· skip $APP (no $PNG_NAME)"; continue; }

  # 1. png -> iconset -> icns
  ICONSET="$(mktemp -d)/keeper.iconset"; mkdir -p "$ICONSET"
  for s in 16 32 64 128 256 512; do
    sips -z $s $s             "$PNG" --out "$ICONSET/icon_${s}x${s}.png"    >/dev/null
    sips -z $((s*2)) $((s*2)) "$PNG" --out "$ICONSET/icon_${s}x${s}@2x.png" >/dev/null
  done
  ICNS="$(mktemp -d)/keeper.icns"; iconutil -c icns "$ICONSET" -o "$ICNS"

  # 2. fresh copy of the app
  rm -rf "$OUT"; mkdir -p "$(dirname "$OUT")"; cp -R "$SRC" "$OUT"

  # 3. swap icon + rebrand Info.plist
  cp "$ICNS" "$OUT/Contents/Resources/keeper.icns"
  rm -f "$OUT/Contents/Resources/Terminal.icns" 2>/dev/null || true
  PLIST="$OUT/Contents/Info.plist"
  /usr/libexec/PlistBuddy -c "Set :CFBundleIconFile keeper"                "$PLIST"
  /usr/libexec/PlistBuddy -c "Set :CFBundleName The Keeper"                 "$PLIST" 2>/dev/null || true
  /usr/libexec/PlistBuddy -c "Set :CFBundleDisplayName The Keeper"          "$PLIST" 2>/dev/null || \
    /usr/libexec/PlistBuddy -c "Add :CFBundleDisplayName string The Keeper" "$PLIST"
  /usr/libexec/PlistBuddy -c "Set :CFBundleIdentifier com.keeper.$SUFFIX"   "$PLIST"

  # 4. re-sign ad-hoc (icon/plist changed) + register with Launch Services
  codesign --force --deep --sign - "$OUT" >/dev/null 2>&1 || true
  /System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister -f "$OUT" 2>/dev/null || true
  touch "$OUT"
  echo "· built $APP  (com.keeper.$SUFFIX)"
  built=$((built + 1))
done

# 5. flush the caches that serve notification badge icons by bundle id: without
# this, macOS keeps showing the icon/name from a previous build. usernoted and
# NotificationCenter just restart (harmless); we deliberately do NOT killall Dock.
killall usernoted 2>/dev/null || true
killall NotificationCenter 2>/dev/null || true

echo "built $built app(s) in $NOTIFY (icon + notification caches flushed)"
