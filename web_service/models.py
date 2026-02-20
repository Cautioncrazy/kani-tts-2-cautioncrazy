from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class Chat(BaseModel):
    id: str
    title: str
    created_at: datetime

class Message(BaseModel):
    id: str
    chat_id: str
    role: str  # "user" or "assistant"
    content: str
    audio_url: Optional[str] = None
    voice_id: Optional[str] = None
    timestamp: datetime

class Voice(BaseModel):
    id: str
    name: str
    filename: str  # Path to .pt file relative to voices dir
    created_at: datetime

class ChatCreate(BaseModel):
    title: str

class MessageCreate(BaseModel):
    chat_id: str
    content: str
    voice_id: Optional[str] = None
    temperature: float = 1.0
    top_p: float = 0.95
    repetition_penalty: float = 1.1

class VoiceCreate(BaseModel):
    name: str
