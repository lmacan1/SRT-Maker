# SRT-Maker

Web app to generate Premiere-ready SRT files from audio and video — self-hostable, runs Whisper locally with CUDA.

## Features

- Transcribes audio (MP3, WAV, M4A, OGG, FLAC) and video (MP4, MKV, WebM, MOV) files
- Uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper) large-v3 with GPU acceleration
- Pause-aware subtitle grouping — breaks cues at natural speech pauses and punctuation
- Gap-free timing — no loose frames between consecutive subtitle cues
- Drag-and-drop web UI with dark mode
- Automatic SRT file download

## How it works

1. Upload an audio or video file via the web UI
2. Audio is extracted from video files using FFmpeg
3. faster-whisper transcribes with word-level timestamps
4. Words are grouped into subtitle cues based on:
   - Natural speech pauses (>=300ms gaps)
   - Punctuation (sentences, commas, semicolons)
   - Dynamic word count (2-5 words depending on speech rhythm)
5. Gaps between cues are closed (up to 500ms) to eliminate loose frames
6. SRT file is returned for download

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Requires FFmpeg for video file support.

## Usage

```bash
python main.py
# or
uvicorn main:app --host 0.0.0.0 --port 8765
```

Then open `http://localhost:8765` in your browser.

## Deployment

The app runs as a systemd service behind a Cloudflare Tunnel at [srt.lukanet.work](https://srt.lukanet.work).
