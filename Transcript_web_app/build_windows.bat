@echo off
setlocal

cd /d %~dp0

python -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt pyinstaller pillow

python generate_icon.py
python -c "from PIL import Image; img=Image.open('assets/icon_1024.png'); img.save('assets/transcript_extractor.ico', sizes=[(16,16),(24,24),(32,32),(48,48),(64,64),(128,128),(256,256)])"

pyinstaller ^
  --noconfirm ^
  --windowed ^
  --name "Transcript Extractor" ^
  --icon "assets/transcript_extractor.ico" ^
  --hidden-import app ^
  --add-data "templates;templates" ^
  desktop_launcher.py

echo Build complete: dist\Transcript Extractor\Transcript Extractor.exe
endlocal
