# Offline Video Transcriber

This tool allows you to extract transcripts (TXT, SRT, VTT, JSON) from video files locally using the [faster-whisper](https://github.com/SYSTRAN/faster-whisper) model. It does not require an OpenAI API key and runs entirely on your machine.

## Prerequisites

1.  **Python 3.8+**: Ensure you have Python installed.
2.  **FFmpeg**: Required for media processing.
    *   **Mac (Homebrew)**: `brew install ffmpeg`
    *   **Windows**: [Download FFmpeg](https://ffmpeg.org/download.html) and add it to your PATH.
    *   **Linux**: `sudo apt install ffmpeg`

## Installation

1.  Navigate to the project directory:
    ```bash
    cd /path/to/extracted_folder
    ```

2.  Install the required Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the script by specifying the input root directory containing your video files:

```bash
python transcribe_local.py --input-root "/path/to/your/video_folder"
```

### Common Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--input-root` | **(Required)** Folder to scan for video files. | N/A |
| `--out-root` | Where to save the transcripts. | `transcripts` |
| `--model` | Model size: `tiny`, `base`, `small`, `medium`, `large-v3`. | `large-v3` |
| `--device` | Hardware to use: `auto`, `cpu`, or `cuda`. | `auto` |
| `--language` | Force a specific language (e.g., `en`). | Auto-detect |
| `--vad` | Enable Voice Activity Detection to filter silence. | Disabled |
| `--workers` | Number of parallel transcriptions (model instances). | `1` |

### Examples

**Process all videos in the current folder using the "medium" model:**
```bash
python transcribe_local.py --input-root . --model medium
```

**Transcribe videos in a specific folder and save outputs to "my_transcripts":**
```bash
python transcribe_local.py --input-root "./Videos" --out-root "./my_transcripts"
```

**Run on CPU with multiple workers (if you have enough RAM):**
```bash
python transcribe_local.py --input-root . --device cpu --workers 2
```

## Output

For each video file found, the script will generate:
*   `.txt`: Plain text transcript.
*   `.srt`: Subtitle file.
*   `.vtt`: WebVTT subtitle file.
*   `.json`: Detailed segment data with timestamps and language info.
