# app/utils/audio_utils.py
import subprocess
from app.utils.logger import logger

def normalize_to_wav(in_path: str, out_path: str, sr: int = 16000):
    # requires ffmpeg installed
    cmd = ["ffmpeg", "-y", "-i", in_path, "-ac", "1", "-ar", str(sr), out_path]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        logger.exception("ffmpeg failed")
        raise RuntimeError("Audio normalization failed") from e
    return out_path