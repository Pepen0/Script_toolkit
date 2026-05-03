#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import zipfile
from datetime import timedelta
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from faster_whisper import WhisperModel

app = FastAPI(title="Local Transcript Extractor")


def _resource_dir() -> Path:
    # PyInstaller extracts bundled files to _MEIPASS at runtime.
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent


templates = Jinja2Templates(directory=str(_resource_dir() / "templates"))


def format_timestamp(seconds: float, srt: bool = True) -> str:
    if seconds is None:
        return "00:00:00,000" if srt else "00:00:00.000"
    ms = int(round(seconds * 1000))
    td = timedelta(milliseconds=ms)
    hours, rem = divmod(td.seconds, 3600)
    minutes, secs = divmod(rem, 60)
    hours += td.days * 24
    millis = ms % 1000
    if srt:
        return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"
    return f"{hours:02}:{minutes:02}:{secs:02}.{millis:03}"


def write_txt(out_txt: Path, segments: List[Tuple[float, float, str]]) -> None:
    with out_txt.open("w", encoding="utf-8") as f:
        for _, _, text in segments:
            line = text.strip()
            if line:
                f.write(line + "\n")


def write_srt(out_srt: Path, segments: List[Tuple[float, float, str]]) -> None:
    with out_srt.open("w", encoding="utf-8") as f:
        for idx, (start, end, text) in enumerate(segments, start=1):
            f.write(f"{idx}\n")
            f.write(f"{format_timestamp(start, srt=True)} --> {format_timestamp(end, srt=True)}\n")
            f.write(text.strip() + "\n\n")


def write_vtt(out_vtt: Path, segments: List[Tuple[float, float, str]]) -> None:
    with out_vtt.open("w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for (start, end, text) in segments:
            f.write(f"{format_timestamp(start, srt=False)} --> {format_timestamp(end, srt=False)}\n")
            f.write(text.strip() + "\n\n")


def write_json(out_json: Path, language: str, duration: Optional[float], segments: List[Tuple[float, float, str]]) -> None:
    payload = {
        "language": language,
        "duration": duration,
        "segments": [
            {"start": start, "end": end, "text": text.strip()}
            for (start, end, text) in segments
        ],
    }
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_chunked_word_segments(words: List[Tuple[float, float, str]], chunk_size: int) -> List[Tuple[float, float, str]]:
    chunked_segments: List[Tuple[float, float, str]] = []
    if chunk_size <= 0:
        return chunked_segments
    for i in range(0, len(words), chunk_size):
        chunk = words[i : i + chunk_size]
        if not chunk:
            continue
        start = chunk[0][0]
        end = chunk[-1][1]
        text = "".join(w[2] for w in chunk).strip()
        if text:
            chunked_segments.append((start, end, text))
    return chunked_segments


def _word_visual_weight(text: str) -> float:
    core = text.strip(" \t\n\r.,!?;:\"'()[]{}")
    n = len(core)
    if n <= 4:
        return 1.0
    if n <= 7:
        return 1.5
    return 3.0


def build_adaptive_chunked_word_segments(words: List[Tuple[float, float, str]], chunk_budget: int) -> List[Tuple[float, float, str]]:
    chunked_segments: List[Tuple[float, float, str]] = []
    if chunk_budget <= 0:
        return chunked_segments
    i = 0
    while i < len(words):
        start = i
        used = 0.0
        while i < len(words):
            next_weight = _word_visual_weight(words[i][2])
            if i == start:
                used += next_weight
                i += 1
                continue
            if used + next_weight > chunk_budget:
                break
            used += next_weight
            i += 1
        chunk = words[start:i]
        text = "".join(w[2] for w in chunk).strip()
        if text:
            chunked_segments.append((chunk[0][0], chunk[-1][1], text))
    return chunked_segments


@lru_cache(maxsize=8)
def get_model(model_name: str, device: str, compute_type: str) -> WhisperModel:
    return WhisperModel(model_name, device=device, compute_type=compute_type)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form("large-v3"),
    device: str = Form("auto"),
    compute_type: str = Form("int8"),
    language: str = Form(""),
    vad: bool = Form(False),
    beam_size: int = Form(5),
    initial_prompt: str = Form(""),
    no_condition_on_previous: bool = Form(False),
    temperature: float = Form(0.0),
    word_timestamps: bool = Form(False),
    subtitle_chunk_size: int = Form(3),
    subtitle_chunk_mode: str = Form("adaptive"),
    out_txt: bool = Form(False),
    out_srt: bool = Form(True),
    out_vtt: bool = Form(False),
    out_json: bool = Form(False),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")
    if subtitle_chunk_mode not in {"fixed", "adaptive"}:
        raise HTTPException(status_code=400, detail="Invalid subtitle chunk mode")
    if subtitle_chunk_size < 0:
        raise HTTPException(status_code=400, detail="subtitle_chunk_size must be >= 0")

    ext = Path(file.filename).suffix.lower()
    if ext not in {".mp4", ".mkv", ".mov", ".m4v", ".avi", ".mpg", ".mpeg"}:
        raise HTTPException(status_code=400, detail="Unsupported video format")

    selected_count = sum([out_txt, out_srt, out_vtt, out_json])
    if selected_count == 0:
        raise HTTPException(status_code=400, detail="Select at least one output format")

    with tempfile.TemporaryDirectory(prefix="transcribe_web_") as tmpdir:
        tmp = Path(tmpdir)
        in_path = tmp / f"input{ext}"
        with in_path.open("wb") as out_f:
            shutil.copyfileobj(file.file, out_f)

        needs_word_timestamps = word_timestamps or subtitle_chunk_size > 0
        model_obj = get_model(model, device, compute_type)
        segments_iter, info = model_obj.transcribe(
            str(in_path),
            language=language or None,
            vad_filter=vad,
            beam_size=beam_size,
            best_of=beam_size,
            initial_prompt=initial_prompt or None,
            condition_on_previous_text=not no_condition_on_previous,
            temperature=temperature,
            word_timestamps=needs_word_timestamps,
        )

        collected: List[Tuple[float, float, str]] = []
        collected_words: List[Tuple[float, float, str]] = []
        for seg in segments_iter:
            collected.append((seg.start, seg.end, seg.text))
            if needs_word_timestamps and seg.words:
                for word in seg.words:
                    if word.start is None or word.end is None:
                        continue
                    collected_words.append((word.start, word.end, word.word))

        if subtitle_chunk_size > 0:
            if subtitle_chunk_mode == "adaptive":
                subtitle_segments = build_adaptive_chunked_word_segments(collected_words, subtitle_chunk_size)
            else:
                subtitle_segments = build_chunked_word_segments(collected_words, subtitle_chunk_size)
        else:
            subtitle_segments = collected

        base_name = Path(file.filename).stem
        out_dir = tmp / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)

        if out_txt:
            write_txt(out_dir / f"{base_name}.txt", collected)
        if out_srt:
            write_srt(out_dir / f"{base_name}.srt", subtitle_segments)
        if out_vtt:
            write_vtt(out_dir / f"{base_name}.vtt", subtitle_segments)
        if out_json:
            write_json(out_dir / f"{base_name}.json", info.language, info.duration, collected)

        zip_path = tmp / f"{base_name}_artifacts.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in out_dir.iterdir():
                zf.write(p, arcname=p.name)

        final_zip = Path(tempfile.gettempdir()) / f"{base_name}_artifacts_{os.getpid()}.zip"
        shutil.copy2(zip_path, final_zip)

    return FileResponse(
        path=final_zip,
        filename=final_zip.name,
        media_type="application/zip",
        background=None,
    )
