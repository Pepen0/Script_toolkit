# Script Toolkit

This repository contains a small collection of scripts and helpers for my vibe conding activities. It is organized into a couple of focused subfolders and includes a simple shell helper and a local transcription script.

Quick summary:

- `Extract_codebase/` : contains shell helpers for concatenating or preparing extracted code.
- `Extract_transcript/` : contains a Python transcription script and its `requirements.txt` for local speech-to-text usage.

Getting started
---------------

### To extract transcripts:

1. Install Python dependencies (for transcription features):

```zsh
python3 -m venv .venv
source .venv/bin/activate
pip install -r Extract_transcript/requirements.txt
```

2. Run the transcription script (example):

```zsh
# from repository root
python Extract_transcript/transcribe_local.py --help

# Example usage (adjust flags/args according to the script's options):
python Extract_transcript/transcribe_local.py --input path/to/audio.wav --output transcript.txt
```

### To extract code:

1. Use `concat.sh` to join multiple text or code files into one file:

```zsh
sh Extract_codebase/concat.sh combined.txt
```
