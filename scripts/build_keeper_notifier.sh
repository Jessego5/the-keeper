#!/usr/bin/env bash
# Build backend/notify/Keeper.app — a rebranded copy of terminal-notifier so the
# Keeper's own icon and name show as the PRIMARY badge on native notifications
# (not terminal-notifier's). Reproducible: regenerate anytime from keeper.png.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PNG="$ROOT/backend/static/keeper.png"
OUT="$ROOT/backend/notify/Keeper.app"
SRC="$(ls -d "$(brew --prefix)"/Cellar/terminal-notifier/*/terminal-notifier.app 2>/dev/null | head -1)"

[ -f "$PNG" ] || { echo "missing $PNG"; exit 1; }
[ -d "$SRC" ] || { echo "terminal-notifier not installed (brew install terminal-notifier)"; exit 1; }

# 1. png -> iconset -> icns
ICONSET="$(mktemp -d)/keeper.iconset"; mkdir -p "$ICONSET"
for s in 16 32 64 128 256 512; do
  sips -z $s $s     "$PNG" --out "$ICONSET/icon_${s}x${s}.png"      >/dev/null
  sips -z $((s*2)) $((s*2)) "$PNG" --out "$ICONSET/icon_${s}x${s}@2x.png" >/dev/null
done
ICNS="$(mktemp -d)/keeper.icns"; iconutil -c icns "$ICONSET" -o "$ICNS"

# 2. fresh copy of the app
rm -rf "$OUT"; mkdir -p "$(dirname "$OUT")"; cp -R "$SRC" "$OUT"

# 3. swap icon + rebrand Info.plist
cp "$ICNS" "$OUT/Contents/Resources/keeper.icns"
rm -f "$OUT/Contents/Resources/Terminal.icns" 2>/dev/null || true
PLIST="$OUT/Contents/Info.plist"
/usr/libexec/PlistBuddy -c "Set :CFBundleIconFile keeper"                       "$PLIST"
/usr/libexec/PlistBuddy -c "Set :CFBundleName The Keeper"                        "$PLIST" 2>/dev/null || true
/usr/libexec/PlistBuddy -c "Set :CFBundleDisplayName The Keeper"                 "$PLIST" 2>/dev/null || \
  /usr/libexec/PlistBuddy -c "Add :CFBundleDisplayName string The Keeper"        "$PLIST"
/usr/libexec/PlistBuddy -c "Set :CFBundleIdentifier com.keeper.notifier"         "$PLIST"

# 4. re-sign ad-hoc (icon/plist changed) + register with Launch Services
codesign --force --deep --sign - "$OUT" >/dev/null 2>&1 || true
/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister -f "$OUT" 2>/dev/null || true
touch "$OUT"

# 5. flush the caches that serve notification badge icons by bundle id — without
# this, macOS keeps showing the icon/name from a previous build. usernoted and
# NotificationCenter just restart (harmless); we deliberately do NOT killall Dock.
killall usernoted 2>/dev/null || true
killall NotificationCenter 2>/dev/null || true

echo "built: $OUT (icon + notification caches flushed)"
echo "binary: $OUT/Contents/MacOS/terminal-notifier"
