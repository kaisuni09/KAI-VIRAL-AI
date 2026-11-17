# KAI VIRAL AI v2.0

A free, self-hosted starter app for turning an authorized video file into multiple vertical Shorts.

## Features
- YouTube URL metadata inspection via oEmbed (does not download the video)
- Faster-Whisper English speech transcription and SRT captions
- Heuristic scene/speech-based candidate ranking
- Hook and ending CTA suggestions
- OpenCV Haar face detection for an approximate vertical crop center
- FFmpeg 1080x1920 H.264/AAC export
- Multiple MP4 outputs

## Important limitations
- A YouTube URL alone is not a reliable or permitted way to obtain the full video stream. This app deliberately does not bypass YouTube download restrictions. Upload a video file you own or are authorized to edit.
- “Viral” cannot be guaranteed. Moment ranking is a simple heuristic based on scene changes, speech density and selected excitement words.
- Face tracking is approximate: it samples frames and selects a stable crop center; it is not continuous object tracking.
- Hooks and endings are suggestions, not generated from an LLM.
- First run downloads the Whisper model. CPU inference may be slow on low-end phones.
- Large video rendering needs storage, RAM and CPU. A free shared hosting tier may not support long videos.

## Run locally
Requires Python 3.11 and FFmpeg installed.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```
Open http://127.0.0.1:8000

## Run with Docker
```bash
docker build -t kai-viral-ai .
docker run --rm -p 8000:8000 -v "$PWD/exports:/app/exports" kai-viral-ai
```

## API
- `POST /inspect-url` form field `url`: reads public YouTube oEmbed metadata only.
- `POST /generate` multipart fields: `file` (authorized video), `shorts_count` (1–10).

## Before publishing
Review every clip, subtitle, hook, music/audio rights and platform policy compliance.
