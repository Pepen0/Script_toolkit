import whisper
import re
import os
import shutil
import subprocess
import tempfile
import wave

import numpy as np

# =========================
# SETTINGS
# =========================
MIN_WORDS_PER_LINE = 3
MAX_WORDS_PER_LINE = 4
MAX_PAUSE_SECONDS = 0.8

# =========================
# LOAD MODEL
# =========================
model = whisper.load_model("base")


def load_waveform(path, target_rate=16000):
    with wave.open(path, "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        frames = wav_file.readframes(wav_file.getnframes())

    if sample_width != 2:
        raise RuntimeError(f"Unsupported WAV sample width: {sample_width}")

    audio = np.frombuffer(frames, np.int16).astype(np.float32) / 32768.0

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    if sample_rate != target_rate:
        new_length = int(round(len(audio) * target_rate / sample_rate))
        old_positions = np.arange(len(audio), dtype=np.float32)
        new_positions = np.linspace(0, len(audio) - 1, new_length, dtype=np.float32)
        audio = np.interp(new_positions, old_positions, audio).astype(np.float32)

    return audio


def transcribe_audio(audio_path):
    try:
        return model.transcribe(audio_path, word_timestamps=True)
    except RuntimeError as exc:
        if "Failed to load audio" not in str(exc):
            raise

        mpg123 = shutil.which("mpg123")
        if not mpg123:
            raise

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            subprocess.run(
                [mpg123, "-q", "-w", temp_path, audio_path],
                check=True,
                capture_output=True,
                text=True,
            )
            audio = load_waveform(temp_path)
            return model.transcribe(audio, word_timestamps=True)
        except subprocess.CalledProcessError as decode_error:
            raise RuntimeError(
                f"Failed to decode audio with mpg123: {decode_error.stderr}"
            ) from decode_error
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

# =========================
# CLEAN TEXT (basic LLM-lite)
# =========================
def clean_text(text):
    text = text.strip()
    text = re.sub(r"\b(uh|um|like|you know)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    text = text.capitalize()
    return text

# =========================
# SPLIT INTO CHUNKS
# =========================
def chunk_words(words):
    def split_on_pauses(items):
        groups = []
        current = []

        for word in items:
            if current and word["start"] - current[-1]["end"] > MAX_PAUSE_SECONDS:
                groups.append(current)
                current = []
            current.append(word)

        if current:
            groups.append(current)

        return groups

    def balanced_sizes(count):
        if count <= MAX_WORDS_PER_LINE:
            return [count]

        if count == 5:
            return [3, 2]

        sizes = []
        remaining = count

        while remaining > 0:
            if remaining <= MAX_WORDS_PER_LINE:
                sizes.append(remaining)
                break

            if remaining == 5:
                sizes.extend([3, 2])
                break

            if remaining % MAX_WORDS_PER_LINE == 1:
                size = MIN_WORDS_PER_LINE
            else:
                size = MAX_WORDS_PER_LINE

            sizes.append(size)
            remaining -= size

        return sizes

    chunks = []

    for group in split_on_pauses(words):
        start = 0
        for size in balanced_sizes(len(group)):
            chunks.append(group[start:start + size])
            start += size

    return chunks

# =========================
# SRT TIME FORMAT
# =========================
def srt_time(t):
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    ms = int((t - int(t)) * 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

# =========================
# MAIN FUNCTION
# =========================
def generate_srt(audio_path, output_file="output.srt"):
    result = transcribe_audio(audio_path)

    index = 1
    lines = []

    words = []
    for segment in result["segments"]:
        words.extend(
            [{"word": clean_text(w["word"]), "start": w["start"], "end": w["end"]} for w in segment["words"]]
        )

    for chunk in chunk_words(words):
        start = chunk[0]["start"]
        end = chunk[-1]["end"]

        text = " ".join([w["word"] for w in chunk])
        start_str = srt_time(start)
        end_str = srt_time(end)

        block = f"{index}\n{start_str} --> {end_str}\n{text}\n"
        lines.append(block)
        index += 1

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"✅ SRT generated: {output_file}")


# =========================
# RUN
# =========================
if __name__ == "__main__":
    generate_srt("audio.mp3")
