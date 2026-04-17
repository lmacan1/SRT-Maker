"""
SRT-Maker: Web app for transcribing MP3 files to SRT subtitles using faster-whisper.
"""
import os
import logging
import uuid
import subprocess
from pathlib import Path
from typing import List, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("srt-maker")

VIDEO_EXTENSIONS = ('.mp4', '.mkv', '.webm', '.mov')

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel

app = FastAPI(title="SRT-Maker")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize model lazily
_model = None

def extract_audio(video_path: Path, output_path: Path) -> None:
    """Extract audio from video file using ffmpeg."""
    subprocess.run([
        'ffmpeg', '-i', str(video_path),
        '-vn', '-acodec', 'libmp3lame', '-q:a', '2',
        '-y', str(output_path)
    ], check=True, capture_output=True)


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


def unload_model():
    """Unload the whisper model to free GPU memory."""
    global _model
    if _model is not None:
        del _model
        _model = None
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass
        print("Model unloaded, GPU memory freed.")


def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def split_words_into_cues(
    words: List,
    min_words: int = 2,
    max_words: int = 5,
    pause_threshold: float = 0.3,
) -> List[Tuple[float, float, str]]:
    """
    Split words into cues using speech rhythm and punctuation.

    - Breaks at natural pauses (gaps >= pause_threshold between words)
    - Breaks after commas, sentence-ending punctuation
    - Allows up to max_words for fast speech, breaks earlier for slow speech
    - Always breaks at min_words if punctuation or pause demands it

    Returns list of (start, end, text) tuples.
    """
    if not words:
        return []

    cues = []
    current_words = []
    current_word_infos = []
    current_start = None

    for i, word_info in enumerate(words):
        word_text = word_info.word.strip()
        if not word_text:
            continue

        if current_start is None:
            current_start = word_info.start

        current_words.append(word_text)
        current_word_infos.append(word_info)
        current_end = word_info.end

        should_break = False
        n = len(current_words)

        # Hard break at max words
        if n >= max_words:
            should_break = True

        # Break at sentence-ending punctuation (always, even with 1 word)
        elif word_text[-1] in '.!?':
            should_break = True

        # Break at comma / semicolon / colon if we have enough words
        elif n >= min_words and word_text[-1] in ',;:':
            should_break = True

        # Break at natural pause before the *next* word
        elif n >= min_words and i + 1 < len(words):
            next_start = words[i + 1].start
            gap = next_start - current_end
            if gap >= pause_threshold:
                should_break = True

        if should_break:
            cue_text = " ".join(current_words)
            cues.append((current_start, current_end, cue_text))
            current_words = []
            current_word_infos = []
            current_start = None

    # Handle remaining words
    if current_words and current_start is not None:
        cue_text = " ".join(current_words)
        cues.append((current_start, current_end, cue_text))

    return cues


def close_gaps(cues: List[Tuple[float, float, str]], max_extend: float = 0.5) -> List[Tuple[float, float, str]]:
    """
    Extend each cue's end time to meet the next cue's start time,
    eliminating gaps that cause loose frames. Only extends if the
    gap is shorter than max_extend seconds to avoid stretching
    across intentional pauses.
    """
    if len(cues) <= 1:
        return cues

    closed = []
    for i in range(len(cues) - 1):
        start, end, text = cues[i]
        next_start = cues[i + 1][0]
        gap = next_start - end
        if 0 < gap <= max_extend:
            end = next_start
        closed.append((start, end, text))
    closed.append(cues[-1])
    return closed


def generate_srt(segments) -> str:
    """
    Generate SRT content from whisper segments with word-level timestamps.
    Splits into max 3 words per cue with accurate timestamps and no gaps.
    """
    all_words = []

    for segment in segments:
        if hasattr(segment, 'words') and segment.words:
            all_words.extend(segment.words)

    all_cues = split_words_into_cues(all_words, max_words=3)
    all_cues = close_gaps(all_cues)

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
    if not file.filename.lower().endswith(('.mp3', '.wav', '.m4a', '.ogg', '.flac', '.mp4', '.mkv', '.webm', '.mov')):
        raise HTTPException(status_code=400, detail="Please upload an audio or video file (MP3, WAV, M4A, OGG, FLAC, MP4, MKV, WebM, MOV)")

    # Save uploaded file temporarily
    temp_id = str(uuid.uuid4())
    temp_audio_path = Path("uploads") / f"{temp_id}_{file.filename}"

    extracted_audio_path = None

    try:
        # Write uploaded file to disk
        content = await file.read()
        temp_audio_path.write_bytes(content)

        # Extract audio if it's a video file
        transcribe_path = temp_audio_path
        if file.filename.lower().endswith(VIDEO_EXTENSIONS):
            print(f"Extracting audio from: {file.filename}")
            extracted_audio_path = Path("uploads") / f"{temp_id}_extracted.mp3"
            extract_audio(temp_audio_path, extracted_audio_path)
            transcribe_path = extracted_audio_path
            print("Audio extraction complete")

        # Load model and transcribe
        model = get_model()

        print(f"Transcribing: {file.filename}")
        segments, info = model.transcribe(
            str(transcribe_path),
            language="en",
            beam_size=5,
            best_of=5,
            temperature=0,
            word_timestamps=True,
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

    except HTTPException:
        raise
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode("utf-8", errors="replace") if e.stderr else ""
        logger.error("ffmpeg failed for %s: %s", file.filename, stderr)
        raise HTTPException(
            status_code=422,
            detail="Audio extraction failed — the file may be corrupt or in an unsupported codec.",
        )
    except RuntimeError as e:
        message = str(e).lower()
        if "out of memory" in message or "cuda" in message:
            logger.error("CUDA error on %s: %s", file.filename, e)
            raise HTTPException(
                status_code=507,
                detail="GPU out of memory — try a shorter file or restart the server.",
            )
        logger.exception("transcription runtime error on %s", file.filename)
        raise HTTPException(status_code=500, detail="Transcription failed — check server logs.")
    except Exception:
        logger.exception("unexpected transcription failure on %s", file.filename)
        raise HTTPException(status_code=500, detail="Transcription failed — check server logs.")
    finally:
        # Clean up temp files
        if temp_audio_path.exists():
            temp_audio_path.unlink()
        if extracted_audio_path and extracted_audio_path.exists():
            extracted_audio_path.unlink()
        # Free GPU memory after each transcription
        unload_model()


@app.on_event("startup")
async def startup_event():
    """Server started - model loads on first request, not at startup."""
    print("SRT-Maker ready. Model will load on first transcription request.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)
