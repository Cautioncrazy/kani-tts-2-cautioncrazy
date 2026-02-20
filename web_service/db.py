import sqlite3
import uuid
from datetime import datetime
from typing import List, Optional
from pathlib import Path
from .models import Chat, Message, Voice

class Database:
    def __init__(self, db_path: str = "web_service/storage/data.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chats (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                chat_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                audio_url TEXT,
                voice_id TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (chat_id) REFERENCES chats (id) ON DELETE CASCADE
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS voices (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                filename TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()

    def create_chat(self, title: str) -> Chat:
        chat_id = str(uuid.uuid4())
        created_at = datetime.now()

        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO chats (id, title, created_at) VALUES (?, ?, ?)",
            (chat_id, title, created_at.isoformat())
        )
        conn.commit()
        conn.close()

        return Chat(id=chat_id, title=title, created_at=created_at)

    def get_chats(self) -> List[Chat]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM chats ORDER BY created_at DESC")
        rows = cursor.fetchall()
        conn.close()

        return [
            Chat(
                id=row["id"],
                title=row["title"],
                created_at=datetime.fromisoformat(str(row["created_at"])) if isinstance(row["created_at"], str) else row["created_at"]
            )
            for row in rows
        ]

    def get_chat(self, chat_id: str) -> Optional[Chat]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM chats WHERE id = ?", (chat_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return Chat(
                id=row["id"],
                title=row["title"],
                created_at=datetime.fromisoformat(str(row["created_at"])) if isinstance(row["created_at"], str) else row["created_at"]
            )
        return None

    def delete_chat(self, chat_id: str):
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
        conn.commit()
        conn.close()

    def add_message(self, chat_id: str, role: str, content: str, audio_url: Optional[str] = None, voice_id: Optional[str] = None) -> Message:
        msg_id = str(uuid.uuid4())
        timestamp = datetime.now()

        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO messages (id, chat_id, role, content, audio_url, voice_id, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (msg_id, chat_id, role, content, audio_url, voice_id, timestamp.isoformat())
        )
        conn.commit()
        conn.close()

        return Message(
            id=msg_id,
            chat_id=chat_id,
            role=role,
            content=content,
            audio_url=audio_url,
            voice_id=voice_id,
            timestamp=timestamp
        )

    def get_messages(self, chat_id: str) -> List[Message]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM messages WHERE chat_id = ? ORDER BY timestamp ASC", (chat_id,))
        rows = cursor.fetchall()
        conn.close()

        return [
            Message(
                id=row["id"],
                chat_id=row["chat_id"],
                role=row["role"],
                content=row["content"],
                audio_url=row["audio_url"],
                voice_id=row["voice_id"],
                timestamp=datetime.fromisoformat(str(row["timestamp"])) if isinstance(row["timestamp"], str) else row["timestamp"]
            )
            for row in rows
        ]

    def add_voice(self, name: str, filename: str) -> Voice:
        voice_id = str(uuid.uuid4())
        created_at = datetime.now()

        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO voices (id, name, filename, created_at) VALUES (?, ?, ?, ?)",
            (voice_id, name, filename, created_at.isoformat())
        )
        conn.commit()
        conn.close()

        return Voice(id=voice_id, name=name, filename=filename, created_at=created_at)

    def get_voices(self) -> List[Voice]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM voices ORDER BY created_at DESC")
        rows = cursor.fetchall()
        conn.close()

        return [
            Voice(
                id=row["id"],
                name=row["name"],
                filename=row["filename"],
                created_at=datetime.fromisoformat(str(row["created_at"])) if isinstance(row["created_at"], str) else row["created_at"]
            )
            for row in rows
        ]

    def get_voice(self, voice_id: str) -> Optional[Voice]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM voices WHERE id = ?", (voice_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return Voice(
                id=row["id"],
                name=row["name"],
                filename=row["filename"],
                created_at=datetime.fromisoformat(str(row["created_at"])) if isinstance(row["created_at"], str) else row["created_at"]
            )
        return None

db = Database()
