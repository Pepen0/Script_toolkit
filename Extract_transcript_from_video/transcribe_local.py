#!/usr/bin/env python3
"""
Offline MP4 → transcript extractor (TXT + SRT), no OpenAI API required.

Dependencies (CPU-only is fine; GPU optional):
    pip install faster-whisper tqdm

Notes:
- The first run will auto-download the chosen Whisper model locally.
- For best accuracy use "large-v3"; for lighter use try "medium" or "small".
- Ensure ffmpeg is installed on your system for robust media decoding.

Usage examples:
    python transcribe_local.py --input-root "." --model large-v3
    python transcribe_local.py --input-root "./Lecture 2" --model medium --workers 2
"""

import argparse
import concurrent.futures
import json
import os
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from faster_whisper import WhisperModel
from tqdm import tqdm


VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".m4v", ".avi", ".mpg", ".mpeg"}


def find_media_files(root: Path) -> List[Path]:
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )


def format_timestamp(seconds: float, srt: bool = True) -> str:
    if seconds is None:
        return "00:00:00,000" if srt else "00:00:00.000"
    ms = int(round(seconds * 1000))
    td = timedelta(milliseconds=ms)
    # timedelta's string lacks milliseconds padding sometimes; handle manually
    hours, rem = divmod(td.seconds, 3600)
    minutes, secs = divmod(rem, 60)
    hours += td.days * 24
    millis = ms % 1000
    if srt:
        return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"
    return f"{hours:02}:{minutes:02}:{secs:02}.{millis:03}"


def write_txt(out_txt: Path, segments: List[Tuple[float, float, str]]) -> None:
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with out_txt.open("w", encoding="utf-8") as f:
        for _, _, text in segments:
            line = text.strip()
            if line:
                f.write(line + "\n")


def write_srt(out_srt: Path, segments: List[Tuple[float, float, str]]) -> None:
    out_srt.parent.mkdir(parents=True, exist_ok=True)
    with out_srt.open("w", encoding="utf-8") as f:
        for idx, (start, end, text) in enumerate(segments, start=1):
            f.write(f"{idx}\n")
            f.write(f"{format_timestamp(start, srt=True)} --> {format_timestamp(end, srt=True)}\n")
            f.write(text.strip() + "\n\n")


def write_vtt(out_vtt: Path, segments: List[Tuple[float, float, str]]) -> None:
    out_vtt.parent.mkdir(parents=True, exist_ok=True)
    with out_vtt.open("w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for (start, end, text) in segments:
            f.write(f"{format_timestamp(start, srt=False)} --> {format_timestamp(end, srt=False)}\n")
            f.write(text.strip() + "\n\n")


def write_json(out_json: Path, language: str, duration: Optional[float], segments: List[Tuple[float, float, str]]) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
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


def transcribe_one(
    in_path: Path,
    out_root: Path,
    model_name: str,
    device: str,
    compute_type: str,
    language: Optional[str],
    vad_filter: bool,
    beam_size: int,
    initial_prompt: Optional[str],
    condition_on_previous_text: bool,
    temperature: float,
    word_timestamps: bool,
) -> Tuple[Path, Optional[str], float]:
    """
    Returns: (source_path, detected_language, elapsed_seconds)
    """
    model = WhisperModel(model_name, device=device, compute_type=compute_type)

    out_dir = out_root / in_path.parent.relative_to(args.input_root)
    out_stem = out_dir / in_path.stem

    out_txt = out_stem.with_suffix(".txt")
    out_srt = out_stem.with_suffix(".srt")
    out_vtt = out_stem.with_suffix(".vtt")
    out_json = out_stem.with_suffix(".json")

    # Skip if already present to avoid rework
    if out_txt.exists() and out_srt.exists():
        return in_path, None, 0.0

    start_time = time.time()

    segments_iter, info = model.transcribe(
        str(in_path),
        language=language,                 # None => auto-detect
        vad_filter=vad_filter,
        beam_size=beam_size,
        best_of=beam_size,        initial_prompt=initial_prompt,
        condition_on_previous_text=condition_on_previous_text,
        temperature=temperature,
        word_timestamps=word_timestamps,
        # chunk_length: default is fine; change if you have OOMs
    )

    detected_language = info.language
    collected: List[Tuple[float, float, str]] = []
    for seg in segments_iter:
        collected.append((seg.start, seg.end, seg.text))

    write_txt(out_txt, collected)
    write_srt(out_srt, collected)
    write_vtt(out_vtt, collected)
    write_json(out_json, detected_language, info.duration, collected)

    elapsed = time.time() - start_time
    return in_path, detected_language, elapsed


def chunked(it: Iterable[Path], n: int) -> Iterable[List[Path]]:
    chunk: List[Path] = []
    for x in it:
        chunk.append(x)
        if len(chunk) == n:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline transcription using faster-whisper.")
    p.add_argument("--input-root", type=Path, required=True, help="Root folder to scan for media.")
    p.add_argument("--out-root", type=Path, default=Path("transcripts"), help="Where to write outputs.")
    p.add_argument("--model", type=str, default="large-v3", help="Model size/name (e.g., tiny, base, small, medium, large-v3).")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Inference device.")
    p.add_argument("--compute-type", type=str, default="int8",
                   help="E.g., float32, float16, int8_float16, int8, int8_float32 (GPU/CPU dependent).")
    p.add_argument("--language", type=str, default=None, help="Force language code like 'en'; omit for auto-detect.")
    p.add_argument("--vad", action="store_true", help="Enable VAD filter for cleaner segments.")
    p.add_argument("--beam-size", type=int, default=5, help="Beam size for decoding.")
    p.add_argument("--initial-prompt", type=str, default=None, help="Optional domain prompt to bias transcription.")
    p.add_argument("--no-condition-on-previous", action="store_true", help="Do not condition on previous text.")
    p.add_argument("--temperature", type=float, default=0.0, help="Decoding temperature.")
    p.add_argument("--word-timestamps", action="store_true", help="Emit word timestamps (slower, larger JSON).")
    p.add_argument("--workers", type=int, default=1, help="Parallel workers (each loads a model; set >1 only if you have RAM/VRAM).")
    return p.parse_args()


def _worker(args_tuple):
    (in_path, args_dict) = args_tuple
    return transcribe_one(
        in_path=in_path,
        out_root=args_dict["out_root"],
        model_name=args_dict["model"],
        device=args_dict["device"],
        compute_type=args_dict["compute_type"],
        language=args_dict["language"],
        vad_filter=args_dict["vad"],
        beam_size=args_dict["beam_size"],        initial_prompt=args_dict["initial_prompt"],
        condition_on_previous_text=not args_dict["no_condition_on_previous"],
        temperature=args_dict["temperature"],
        word_timestamps=args_dict["word_timestamps"],
    )


def main() -> int:
    global args
    args = parse_args()

    media_files = find_media_files(args.input_root)
    if not media_files:
        print("No media files found.", file=sys.stderr)
        return 1

    args.out_root.mkdir(parents=True, exist_ok=True)

    tasks = [(p, vars(args)) for p in media_files]

    if args.workers <= 1:
        for p in tqdm(media_files, desc="Transcribing", unit="file"):
            _worker((p, vars(args)))
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
            for _ in tqdm(ex.map(_worker, tasks), total=len(tasks), desc="Transcribing", unit="file"):
                pass

    print(f"Done. Outputs written under: {args.out_root.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
