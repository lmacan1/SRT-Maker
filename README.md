# SRT-Maker

Web app to generate Premiere-ready SRT files from audio and video — self-hostable, runs Whisper locally with CUDA.

## Features

- Transcribes audio (MP3, WAV, M4A, OGG, FLAC) and video (MP4, MKV, WebM, MOV) files
- Uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper) large-v3 (`int8_float16`) with GPU acceleration
- Batched inference — roughly 2.5x faster than sequential decoding on long audio
- Pause-aware subtitle grouping — breaks cues at natural speech pauses and punctuation
- Silence-aligned timing — cue endings snap to acoustically detected silences
- Drag-and-drop web UI with dark mode (drop files anywhere on the page)
- Automatic SRT file download, including filenames with non-ASCII characters

## How it works

1. Upload an audio or video file via the web UI (streamed to disk, 2 GB max)
2. Video is converted to 16 kHz mono WAV with FFmpeg (the format Whisper uses natively)
3. faster-whisper transcribes with word-level timestamps using `BatchedInferencePipeline` (batch size 16, beam size 5, VAD filter). FFmpeg silence detection runs on the CPU at the same time as GPU transcription
4. Words are grouped into subtitle cues (max 3 words) based on:
   - Natural speech pauses (>=300ms gaps)
   - Punctuation (sentences, commas, semicolons)

   Hyphens and dashes are removed from the text ("three-step" becomes "three step").
5. Cue ends are snapped to the detected silences (or lightly padded mid-phrase) to correct Whisper's timestamp drift
6. The SRT is returned for download and deleted from the server afterwards

## Performance and reliability

- **Warm model:** the model stays loaded in VRAM for 90 seconds after a request, so back-to-back files skip the load. It then unloads to free the shared GPU (`IDLE_UNLOAD_SECONDS` in `main.py`). The model loads from the local cache first, with no HuggingFace round-trip.
- **Queueing:** requests are serialized by a lock, so concurrent uploads wait their turn instead of competing for VRAM. Blocking work runs in threads, so the server stays responsive.
- **Preflight VRAM check:** requests get a 503 up front if less than 2 GB of VRAM is available.
- **Limits and timeouts:** uploads over 2 GB return 413, and FFmpeg calls time out after 10 minutes (504).
- **Error handling:** specific HTTP statuses for bad input (400), corrupt media (422), GPU out of memory (507), and busy GPU (503). The model is unloaded after a failed request to recover VRAM.
- **Cleanup:** temp files are removed after every request, and a background sweep (startup and every 15 minutes) deletes orphaned files in `uploads/` older than an hour.
- **Health check:** `GET /health` returns `{"status": "ok", "model_loaded": <bool>}`.

The app must run as a single worker process, because the GPU model is held in-process.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Requires FFmpeg (video support and silence detection) and an NVIDIA GPU with CUDA.

## Usage

```bash
python main.py
# or
uvicorn main:app --host 0.0.0.0 --port 8765
```

Then open `http://localhost:8765` in your browser.

## Deployment

The app runs as a systemd service behind a Cloudflare Tunnel at [srt.lukanet.work](https://srt.lukanet.work).
