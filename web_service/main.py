from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import logging
from pathlib import Path
from typing import Optional

from .tts_manager import tts_manager
from .db import db
from .models import ChatCreate, MessageCreate, VoiceCreate

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="KaniTTS Web Service")

# Mount static files
app.mount("/static", StaticFiles(directory="web_service/static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="web_service/templates")

# CORS (useful if developing frontend separately, but good practice)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/voices", response_class=HTMLResponse)
async def read_voices(request: Request):
    return templates.TemplateResponse(request=request, name="voices.html")

# --- API Endpoints ---

@app.get("/api/model/status")
async def get_model_status():
    return tts_manager.get_status()

@app.post("/api/model/load")
async def load_model(request: Request):
    data = await request.json()
    model_name = data.get("model_name")
    if not model_name:
        raise HTTPException(status_code=400, detail="model_name is required")

    try:
        await tts_manager.load_model(model_name)
        return {"status": "success", "message": f"Model {model_name} loading started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/voices")
async def get_voices():
    return db.get_voices()

@app.post("/api/voices/clone")
async def clone_voice(name: str = Form(...), file: UploadFile = File(...)):
    # Save uploaded file temporarily
    temp_dir = Path("web_service/temp")
    temp_dir.mkdir(exist_ok=True)

    temp_file_path = temp_dir / file.filename
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Create embedding
        emb_filename = await tts_manager.create_voice_embedding(str(temp_file_path))

        # Add to DB
        voice = db.add_voice(name, emb_filename)

        # Cleanup temp file
        os.remove(temp_file_path)

        return voice
    except Exception as e:
        if temp_file_path.exists():
            os.remove(temp_file_path)
        logger.error(f"Voice cloning failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chats")
async def get_chats():
    return db.get_chats()

@app.post("/api/chats")
async def create_chat(chat_data: ChatCreate):
    return db.create_chat(chat_data.title)

@app.delete("/api/chats/{chat_id}")
async def delete_chat(chat_id: str):
    db.delete_chat(chat_id)
    return {"status": "success"}

@app.get("/api/chats/{chat_id}/messages")
async def get_messages(chat_id: str):
    return db.get_messages(chat_id)

@app.post("/api/chats/{chat_id}/generate")
async def generate_speech(chat_id: str, message_data: MessageCreate):
    # 1. Save user message
    db.add_message(
        chat_id=chat_id,
        role="user",
        content=message_data.content,
        voice_id=message_data.voice_id
    )

    # 2. Check model status
    status = tts_manager.get_status()
    if status["status"] != "READY":
        raise HTTPException(status_code=503, detail="Model is not ready. Please load a model first.")

    # 3. Get voice embedding path if provided
    voice_emb_path = None
    if message_data.voice_id:
        voice = db.get_voice(message_data.voice_id) # Synchronous call
        if voice:
            voice_emb_path = tts_manager.voices_dir / voice.filename
        else:
             logger.warning(f"Voice {message_data.voice_id} not found, using default voice.")

    # 4. Generate audio
    try:
        audio_filename, gen_text = await tts_manager.generate_audio(
            text=message_data.content,
            voice_emb_path=str(voice_emb_path) if voice_emb_path else None,
            temperature=message_data.temperature,
            top_p=message_data.top_p,
            repetition_penalty=message_data.repetition_penalty
        )

        # 5. Save assistant message
        audio_url = f"/static/audio/{audio_filename}"

        msg = db.add_message(
            chat_id=chat_id,
            role="assistant",
            content=gen_text, # Use generated text (might be same as input or modified)
            audio_url=audio_url,
            voice_id=message_data.voice_id
        )

        return msg
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
