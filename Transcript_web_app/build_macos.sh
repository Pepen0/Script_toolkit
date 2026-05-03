#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt pyinstaller pillow

python generate_icon.py

rm -rf assets/icon.iconset
mkdir -p assets/icon.iconset

sips -z 16 16     assets/icon_1024.png --out assets/icon.iconset/icon_16x16.png >/dev/null
sips -z 32 32     assets/icon_1024.png --out assets/icon.iconset/icon_16x16@2x.png >/dev/null
sips -z 32 32     assets/icon_1024.png --out assets/icon.iconset/icon_32x32.png >/dev/null
sips -z 64 64     assets/icon_1024.png --out assets/icon.iconset/icon_32x32@2x.png >/dev/null
sips -z 128 128   assets/icon_1024.png --out assets/icon.iconset/icon_128x128.png >/dev/null
sips -z 256 256   assets/icon_1024.png --out assets/icon.iconset/icon_128x128@2x.png >/dev/null
sips -z 256 256   assets/icon_1024.png --out assets/icon.iconset/icon_256x256.png >/dev/null
sips -z 512 512   assets/icon_1024.png --out assets/icon.iconset/icon_256x256@2x.png >/dev/null
sips -z 512 512   assets/icon_1024.png --out assets/icon.iconset/icon_512x512.png >/dev/null
cp assets/icon_1024.png assets/icon.iconset/icon_512x512@2x.png

iconutil -c icns assets/icon.iconset -o assets/transcript_extractor.icns

pyinstaller \
  --noconfirm \
  --windowed \
  --name "Transcript Extractor" \
  --icon "assets/transcript_extractor.icns" \
  --hidden-import app \
  --add-data "templates:templates" \
  desktop_launcher.py

echo "Build complete: dist/Transcript Extractor.app"
