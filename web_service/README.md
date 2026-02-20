# KaniTTS Web Service

A modern web interface for KaniTTS, featuring:
- Chat-style text-to-speech generation
- Voice cloning
- Project management (Chats)
- Dynamic model loading

## Setup

1. Install dependencies:
   ```bash
   pip install -e .[web]
   ```
   Or manually:
   ```bash
   pip install fastapi uvicorn python-multipart jinja2 aiofiles
   ```
   Ensure `kani_tts` and its dependencies are installed.

2. Run the server:
   ```bash
   uvicorn web_service.main:app --reload
   ```

3. Open your browser at `http://localhost:8000`.

## Features

- **Chats**: Organize your generations into distinct chat sessions.
- **Voice Cloning**: Upload reference audio to clone voices.
- **Model Loading**: Load `kani-tts` models dynamically.
- **Generation**: Generate speech with temperature, top_p, and repetition penalty controls.
