#!/usr/bin/env python3
"""
Fast offline live system-audio transcription using faster-whisper,
with interactive choices for model and language.

Install dependencies (once):
    pip install soundcard numpy torch faster-whisper

Also make sure FFmpeg is installed on your system:
    - Windows: install from https://ffmpeg.org/ and add to PATH
    - macOS:   brew install ffmpeg
    - Linux:   use your package manager (e.g. apt, dnf, pacman)

Usage:
    python offline_live_transcript.py
    - Choose model + language in the terminal.
    - Press Ctrl+C to stop; you'll then be asked (on stderr) if you want to save.
"""

import sys
import time
import threading
import queue
from datetime import datetime

import numpy as np
import soundcard as sc
from faster_whisper import WhisperModel


# ---------------- Configuration ---------------- #

SAMPLE_RATE = 16_000         # Hz
CHUNK_SECONDS = 2            # smaller chunk for lower latency

# Defaults (can be overridden by interactive choices)
MODEL_NAME = "small"         # "tiny", "base", "small", "medium", "large-v3"
DEVICE = "auto"              # "auto", "cpu", or "cuda"
COMPUTE_TYPE = "int8"        # e.g. "int8", "int8_float16", "float16", "float32"
LANGUAGE = None              # None = auto-detect; or "en", "fr", "es", "ar"

# Target loudness for normalization (RMS in linear scale, 0–1)
TARGET_RMS = 0.1
MAX_GAIN_DB = 25.0           # don’t boost more than this per chunk

# ------------------------------------------------ #


def choose_model_name() -> str:
    """
    Ask the user which Whisper model to use.
    """
    options = {
        "1": "tiny",
        "2": "base",
        "3": "small",
        "4": "medium",
        "5": "large-v3",
    }
    default_key = "3"  # "small"

    print("\nSelect Whisper model:", file=sys.stderr)
    print("  [1] tiny     (fastest, lowest quality)", file=sys.stderr)
    print("  [2] base     (fast, low/medium quality)", file=sys.stderr)
    print("  [3] small    (default: good balance)", file=sys.stderr)
    print("  [4] medium   (slower, better quality)", file=sys.stderr)
    print("  [5] large-v3 (slowest, best quality)", file=sys.stderr)
    print(f"Enter choice [default {default_key}]: ", end="", flush=True, file=sys.stderr)

    try:
        choice = input().strip()
    except EOFError:
        choice = ""

    model_name = options.get(choice or default_key, options[default_key])
    print(f"Using model: {model_name}", file=sys.stderr)
    return model_name


def choose_language() -> str | None:
    """
    Ask the user which language to use (or auto-detect).
    Returns language code or None.
    """
    options = {
        "1": None,    # Auto-detect
        "2": "en",    # English
        "3": "fr",    # French
        "4": "es",    # Spanish
        "5": "ar",    # Arabic
    }
    labels = {
        None: "Auto-detect",
        "en": "English",
        "fr": "French",
        "es": "Spanish",
        "ar": "Arabic",
    }
    default_key = "1"  # Auto-detect

    print("\nSelect language:", file=sys.stderr)
    print("  [1] Auto-detect", file=sys.stderr)
    print("  [2] English", file=sys.stderr)
    print("  [3] French", file=sys.stderr)
    print("  [4] Spanish", file=sys.stderr)
    print("  [5] Arabic", file=sys.stderr)
    print(f"Enter choice [default {default_key}]: ", end="", flush=True, file=sys.stderr)

    try:
        choice = input().strip()
    except EOFError:
        choice = ""

    lang_code = options.get(choice or default_key, options[default_key])
    print(f"Language: {labels[lang_code]}", file=sys.stderr)
    return lang_code


def get_loopback_microphone():
    """
    Try to get a loopback microphone (records speaker output).
    Fallback to default microphone if loopback is not available.
    """
    mics = sc.all_microphones(include_loopback=True)
    loopbacks = [m for m in mics if getattr(m, "isloopback", False)]
    if loopbacks:
        print(f"Using loopback device: {loopbacks[0].name}", file=sys.stderr)
        return loopbacks[0]
    default_mic = sc.default_microphone()
    print(f"No loopback device found, using default mic: {default_mic.name}", file=sys.stderr)
    return default_mic


def capture_loop(stop_event: threading.Event,
                 audio_queue: queue.Queue,
                 mic):
    """
    Continuously capture system audio into chunks and push them to a queue.
    """
    frames_per_chunk = SAMPLE_RATE * CHUNK_SECONDS
    print(
        f"Starting audio capture at {SAMPLE_RATE} Hz, {CHUNK_SECONDS}s chunks...",
        file=sys.stderr,
    )

    with mic.recorder(samplerate=SAMPLE_RATE) as rec:
        while not stop_event.is_set():
            try:
                data = rec.record(numframes=frames_per_chunk)
                # data: numpy float32, shape (frames, channels)
                audio_queue.put(data)
            except Exception as e:
                print(f"Capture error: {e}", file=sys.stderr)
                time.sleep(0.5)


def mix_down_to_mono(audio_np: np.ndarray) -> np.ndarray:
    """
    Convert multi-channel audio to mono by averaging across channels.
    faster-whisper expects a 1D float32 array at 16kHz.
    """
    if audio_np.ndim == 1:
        return audio_np.astype(np.float32)
    mono = np.mean(audio_np, axis=1)
    return mono.astype(np.float32)


def normalize_audio(audio: np.ndarray,
                    target_rms: float = TARGET_RMS,
                    max_gain_db: float = MAX_GAIN_DB) -> np.ndarray:
    """
    Simple RMS normalization with a cap on maximum gain.

    - audio: 1D float32 in [-1, 1]
    - target_rms: desired RMS level (e.g. 0.1)
    - max_gain_db: limit boost to avoid huge amplification of noise
    """
    eps = 1e-8
    audio = audio.astype(np.float32)

    rms = np.sqrt(np.mean(audio * audio) + eps)
    if rms < eps:
        return audio  # almost silent

    target_gain = target_rms / rms
    max_gain = 10.0 ** (max_gain_db / 20.0)
    gain = min(target_gain, max_gain)

    audio = audio * gain
    audio = np.clip(audio, -1.0, 1.0)

    return audio.astype(np.float32)


def enhance_audio(audio_np: np.ndarray) -> np.ndarray:
    """
    Mono + normalization step before feeding into the model.
    """
    mono = mix_down_to_mono(audio_np)
    normalized = normalize_audio(mono)
    return normalized


def transcription_loop(stop_event: threading.Event,
                       audio_queue: queue.Queue,
                       transcript_buffer: list[str],
                       print_lock: threading.Lock,
                       model: WhisperModel,
                       language: str | None):
    """
    Read audio chunks from the queue, run faster-whisper locally,
    append/print transcription results.

    Only the recognized text is printed to stdout (one line per chunk).
    """
    print("Starting transcription loop (faster-whisper, offline)...", file=sys.stderr)

    while not stop_event.is_set() or not audio_queue.empty():
        try:
            data = audio_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        try:
            processed_audio = enhance_audio(data)

            segments, info = model.transcribe(
                processed_audio,
                language=language,           # None => auto-detect
                vad_filter=True,             # skip silence
                beam_size=1,                 # greedy / small beam for speed
                best_of=1,
                condition_on_previous_text=False,
                temperature=0.0,
                word_timestamps=False,
            )

            texts = []
            for seg in segments:
                seg_text = seg.text.strip()
                if seg_text:
                    texts.append(seg_text)

            text = " ".join(texts).strip()
        except Exception as e:
            text = f"[Transcription error: {e}]"

        if text:
            # Print ONLY the detected text to stdout
            with print_lock:
                print(text, flush=True)
            transcript_buffer.append(text)


def save_transcript(transcript_buffer: list[str]):
    """
    Ask user whether to save transcript and, if yes, save to a file.
    Prompts are printed to stderr so stdout stays pure transcript.
    """
    if not transcript_buffer:
        print("No transcript to save.", file=sys.stderr)
        return

    print("\nSave transcript to file? [y/N]: ", end="", flush=True, file=sys.stderr)
    try:
        choice = input().strip().lower()
    except EOFError:
        choice = ""

    if choice not in ("y", "yes"):
        print("Transcript not saved.", file=sys.stderr)
        return

    default_name = "transcript_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".txt"
    print(f"Enter filename (press Enter for default: {default_name}): ",
          end="", flush=True, file=sys.stderr)
    try:
        user_input = input().strip()
    except EOFError:
        user_input = ""

    filename = user_input or default_name
    full_text = "\n".join(transcript_buffer)

    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(full_text)
        print(f"Transcript saved to: {filename}", file=sys.stderr)
    except Exception as e:
        print(f"Failed to save transcript: {e}", file=sys.stderr)


def main():
    global MODEL_NAME, LANGUAGE

    # Ask user for model + language
    MODEL_NAME = choose_model_name()
    LANGUAGE = choose_language()

    print(
        f"\nLoading faster-whisper model '{MODEL_NAME}' "
        f"(device={DEVICE}, compute_type={COMPUTE_TYPE})...",
        file=sys.stderr,
    )
    try:
        model = WhisperModel(
            MODEL_NAME,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
        )
    except Exception as e:
        print(f"Failed to load faster-whisper model: {e}", file=sys.stderr)
        print("Make sure you ran: pip install faster-whisper", file=sys.stderr)
        sys.exit(1)

    mic = get_loopback_microphone()

    stop_event = threading.Event()
    audio_queue: queue.Queue = queue.Queue()
    transcript_buffer: list[str] = []
    print_lock = threading.Lock()

    capture_thread = threading.Thread(
        target=capture_loop,
        args=(stop_event, audio_queue, mic),
        daemon=True,
    )
    transcribe_thread = threading.Thread(
        target=transcription_loop,
        args=(stop_event, audio_queue, transcript_buffer, print_lock, model, LANGUAGE),
        daemon=True,
    )

    print("Press Ctrl+C to stop.\n", file=sys.stderr)
    capture_thread.start()
    transcribe_thread.start()

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping...", file=sys.stderr)
        stop_event.set()
        capture_thread.join(timeout=2.0)
        transcribe_thread.join(timeout=10.0)

    save_transcript(transcript_buffer)
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
