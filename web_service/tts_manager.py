import os
import torch
import logging
import asyncio
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import kani_tts. If not installed, we can't do much, but we assume it is.
try:
    from kani_tts import KaniTTS, SpeakerEmbedder
except ImportError:
    logger.error("kani_tts not found. Please install it.")
    KaniTTS = None
    SpeakerEmbedder = None

class TTSManager:
    def __init__(self, base_dir: str = "web_service"):
        self.model: Optional[Any] = None
        self.embedder: Optional[Any] = None
        self.model_name: Optional[str] = None
        self.status: str = "IDLE"  # IDLE, LOADING, READY, ERROR
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.base_dir = Path(base_dir)
        self.voices_dir = self.base_dir / "voices"
        self.static_dir = self.base_dir / "static"
        self.outputs_dir = self.static_dir / "audio"

        self.voices_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

        self._executor = ThreadPoolExecutor(max_workers=1)

    async def load_model(self, model_name: str):
        if self.status == "LOADING":
            raise RuntimeError("A model is already loading.")

        self.status = "LOADING"
        self.model_name = model_name

        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(self._executor, self._load_model_sync, model_name)
            self.status = "READY"
        except Exception as e:
            self.status = "ERROR"
            self.model_name = None # Reset if failed
            logger.error(f"Failed to load model {model_name}: {e}")
            raise e

    def _load_model_sync(self, model_name: str):
        logger.info(f"Loading model {model_name}...")

        if KaniTTS is None:
             raise ImportError("kani_tts library not found")

        # Initialize KaniTTS
        # We set suppress_logs=True to keep console clean
        self.model = KaniTTS(model_name, device_map="auto", suppress_logs=True)

        # Initialize Embedder if not already
        if self.embedder is None:
            logger.info("Loading SpeakerEmbedder...")
            self.embedder = SpeakerEmbedder(device=self.device)

        logger.info(f"Model {model_name} loaded successfully.")

    async def generate_audio(self, text: str, voice_emb_path: Optional[str] = None, **kwargs) -> Tuple[str, str]:
        """
        Generates audio and saves it to a file.
        Returns: (filename, generated_text)
        """
        if self.status != "READY" or self.model is None:
             raise RuntimeError("Model is not ready. Please load a model first.")

        loop = asyncio.get_event_loop()

        def _generate():
            speaker_emb = None
            if voice_emb_path:
                logger.info(f"Loading speaker embedding from {voice_emb_path}")
                speaker_emb = torch.load(voice_emb_path, map_location=self.device)

            logger.info(f"Generating audio for text: {text[:50]}...")
            audio, gen_text = self.model.generate(
                text,
                speaker_emb=speaker_emb,
                **kwargs
            )
            return audio, gen_text

        try:
            audio, gen_text = await loop.run_in_executor(self._executor, _generate)
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise e

        # Save audio
        # Generate a unique filename
        import uuid
        filename = f"gen_{uuid.uuid4().hex}.wav"
        filepath = self.outputs_dir / filename

        logger.info(f"Saving audio to {filepath}")
        sf.write(str(filepath), audio, self.model.sample_rate)

        # Return filename relative to static/audio so frontend can construct URL
        return filename, gen_text

    async def create_voice_embedding(self, audio_file_path: str) -> str:
        """
        Creates a speaker embedding from an audio file and saves it.
        Returns the filename of the saved embedding .pt file.
        """
        if SpeakerEmbedder is None:
             raise ImportError("kani_tts library not found")

        loop = asyncio.get_event_loop()

        def _process():
            if self.embedder is None:
                # Load embedder on demand if model not loaded yet
                self.embedder = SpeakerEmbedder(device=self.device)

            logger.info(f"Embedding audio from {audio_file_path}")
            emb = self.embedder.embed_audio_file(audio_file_path)

            import uuid
            filename = f"voice_{uuid.uuid4().hex}.pt"
            filepath = self.voices_dir / filename

            logger.info(f"Saving embedding to {filepath}")
            torch.save(emb, filepath)
            return filename

        return await loop.run_in_executor(self._executor, _process)

    def get_status(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "model_name": self.model_name,
            "device": self.device
        }

# Global instance
tts_manager = TTSManager()
