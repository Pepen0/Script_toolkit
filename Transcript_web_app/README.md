# Transcript Extractor (Desktop-Friendly)

This project can be packaged into a double-click desktop app so non-technical users can run transcription locally on their own hardware.

## End User Experience (No Coding)

1. User downloads the app (`Transcript Extractor.app` on macOS or `.exe` on Windows).
2. User double-clicks it.
3. Browser opens automatically.
4. User drops a video and clicks **Transcribe and download ZIP**.

Default behavior:
- Output: `SRT` selected
- Subtitle chunk mode: `adaptive`
- Subtitle chunk size: `3`

## Build macOS App

```bash
cd /Users/penoelo/Desktop/Script_toolkit/Transcript_web_app
./build_macos.sh
```

Result:
- `dist/Transcript Extractor.app`

## Build Windows App

On Windows CMD:

```bat
cd \path\to\Transcript_web_app
build_windows.bat
```

Result:
- `dist\Transcript Extractor\Transcript Extractor.exe`

## Local Dev Run (without packaging)

```bash
cd /Users/penoelo/Desktop/Script_toolkit/Transcript_web_app
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`.

## Notes

- First run may be slower because Whisper model files are downloaded.
- All transcription runs locally on the user machine running the app.
