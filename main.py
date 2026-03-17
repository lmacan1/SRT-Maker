"""
SRT-Maker: Web app for transcribing MP3 files to SRT subtitles using faster-whisper.
"""
import os
import uuid
import tempfile
from pathlib import Path
from typing import List, Tuple

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel

app = FastAPI(title="SRT-Maker")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize model lazily
_model = None

def get_model() -> WhisperModel:
    """Load the whisper model (lazy initialization)."""
    global _model
    if _model is None:
        print("Loading faster-whisper large-v3 model with CUDA...")
        _model = WhisperModel(
            "large-v3",
            device="cuda",
            compute_type="float16"
        )
        print("Model loaded successfully!")
    return _model


def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def split_segment_into_cues(
    text: str,
    start_time: float,
    end_time: float,
    max_words: int = 4
) -> List[Tuple[float, float, str]]:
    """
    Split a segment into cues with max_words per cue.
    Timestamps are interpolated proportionally based on word count.

    Returns list of (start, end, text) tuples.
    """
    words = text.strip().split()
    if not words:
        return []

    total_duration = end_time - start_time
    total_words = len(words)

    cues = []
    word_idx = 0

    while word_idx < total_words:
        # Get next chunk of words (up to max_words)
        chunk_words = words[word_idx:word_idx + max_words]
        chunk_word_count = len(chunk_words)

        # Calculate proportional timestamps
        chunk_start_ratio = word_idx / total_words
        chunk_end_ratio = (word_idx + chunk_word_count) / total_words

        chunk_start = start_time + (total_duration * chunk_start_ratio)
        chunk_end = start_time + (total_duration * chunk_end_ratio)

        chunk_text = " ".join(chunk_words)
        cues.append((chunk_start, chunk_end, chunk_text))

        word_idx += chunk_word_count

    return cues


def generate_srt(segments) -> str:
    """
    Generate SRT content from whisper segments.
    Splits segments into 3-4 word cues with interpolated timestamps.
    """
    all_cues = []

    for segment in segments:
        cues = split_segment_into_cues(
            segment.text,
            segment.start,
            segment.end,
            max_words=4
        )
        all_cues.extend(cues)

    # Build SRT content
    srt_lines = []
    for idx, (start, end, text) in enumerate(all_cues, 1):
        srt_lines.append(str(idx))
        srt_lines.append(f"{format_timestamp(start)} --> {format_timestamp(end)}")
        srt_lines.append(text)
        srt_lines.append("")  # Empty line between cues

    return "\n".join(srt_lines)


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main page."""
    return Path("static/index.html").read_text()


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """
    Transcribe an uploaded MP3 file and return an SRT file.
    """
    if not file.filename.lower().endswith(('.mp3', '.wav', '.m4a', '.ogg', '.flac')):
        raise HTTPException(status_code=400, detail="Please upload an audio file (MP3, WAV, M4A, OGG, FLAC)")

    # Save uploaded file temporarily
    temp_id = str(uuid.uuid4())
    temp_audio_path = Path("uploads") / f"{temp_id}_{file.filename}"

    try:
        # Write uploaded file to disk
        content = await file.read()
        temp_audio_path.write_bytes(content)

        # Load model and transcribe
        model = get_model()

        print(f"Transcribing: {file.filename}")
        segments, info = model.transcribe(
            str(temp_audio_path),
            language="en",
            beam_size=5,
            best_of=5,
            temperature=0,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
            ),
        )

        # Collect all segments (generator needs to be consumed)
        segments_list = list(segments)
        print(f"Transcription complete: {len(segments_list)} segments, duration {info.duration:.1f}s")

        # Generate SRT content
        srt_content = generate_srt(segments_list)

        # Save SRT file
        srt_filename = Path(file.filename).stem + ".srt"
        srt_path = Path("uploads") / f"{temp_id}_{srt_filename}"
        srt_path.write_text(srt_content, encoding="utf-8")

        return FileResponse(
            path=str(srt_path),
            filename=srt_filename,
            media_type="application/x-subrip",
            headers={"Content-Disposition": f'attachment; filename="{srt_filename}"'}
        )

    finally:
        # Clean up temp audio file
        if temp_audio_path.exists():
            temp_audio_path.unlink()


@app.on_event("startup")
async def startup_event():
    """Preload the model on startup."""
    # Preload in background - first request will still be fast
    import threading
    threading.Thread(target=get_model, daemon=True).start()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)
