# app/services/asr_service.py
import whisper
from app.utils.logger import logger

# Global model cache (so Whisper loads once)
_model = None

def load_model(model_name: str = "base"):
    """
    Load the Whisper model (once).
    Available models: tiny, base, small, medium, large
    """
    global _model
    if _model is None:
        logger.info(f"Loading Whisper model: {model_name}")
        _model = whisper.load_model(model_name)
        logger.info(f"Whisper model '{model_name}' loaded successfully.")
    return _model

def transcribe_local(wav_path: str, model_name: str = "base", language: str = None):
    """
    Transcribe an audio file using the local Whisper model.
    Args:
        wav_path (str): Path to 16kHz mono WAV file
        model_name (str): Whisper model size ('tiny', 'base', 'small', etc.)
        language (str): Optional language hint (e.g., 'en')

    Returns:
        tuple: (transcript_text, metadata_dict)
    """
    model = load_model(model_name)
    logger.info(f"Transcribing audio: {wav_path}")

    # Perform transcription
    result = model.transcribe(wav_path, language=language)

    text = result["text"].strip()
    segments = result.get("segments", [])
    logger.info(f"Transcription complete — {len(text)} characters, {len(segments)} segments.")

    return text, {
        "model": model_name,
        "language": result.get("language"),
        "duration": result.get("duration"),
        "segments": segments
    }
