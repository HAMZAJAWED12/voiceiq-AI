import os
import torch
import soundfile as sf
from typing import List, Dict
from app.utils.logger import logger
from huggingface_hub import login

# Try importing pyannote
try:
    from pyannote.audio import Pipeline
    _has_pyannote = True
except ImportError:
    _has_pyannote = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_diarization_pipeline = None


# ------------------------------------------------------------
# Load & Cache Pyannote Pipeline
# ------------------------------------------------------------
def load_diarization_pipeline():
    """
    Loads the Pyannote speaker diarization pipeline (cached).
    Compatible with pyannote.audio==3.3.x.
    """
    global _diarization_pipeline

    if _diarization_pipeline is not None:
        return _diarization_pipeline

    if not _has_pyannote:
        logger.warning("Pyannote not installed. Using mock diarization.")
        return None

    token = os.getenv("PYANNOTE_AUTH_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        logger.warning("No PYANNOTE_AUTH_TOKEN or HUGGINGFACE_TOKEN found. Using mock diarization.")
        return None

    logger.info("Loading Pyannote speaker-diarization pipeline...")

    try:
        login(token=token)

        _diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            use_auth_token=token
        ).to(DEVICE)

        # Optional: adjust clustering threshold
        try:
            _diarization_pipeline.instantiate({
                "clustering": {
                    "method": "centroid",
                    "threshold": 0.65  # tweak 0.6–0.75 depending on your use-case
                }
            })
            logger.info("Adjusted clustering threshold for enhanced speaker separation.")
        except Exception as e:
            logger.warning(f"Could not adjust clustering threshold: {e}")

        logger.info(f"Pyannote diarization pipeline loaded successfully on {DEVICE}.")

    except Exception as e:
        logger.error(f"Failed to load Pyannote pipeline: {e}")
        _diarization_pipeline = None

    return _diarization_pipeline


# ------------------------------------------------------------
# Mock fallback diarization
# ------------------------------------------------------------
def _mock_diarization(wav_path: str) -> List[Dict]:
    data, sr = sf.read(wav_path)
    duration = len(data) / sr
    logger.info(f"Mock diarization for {duration:.1f}s audio.")
    return [{
        "start": 0.0,
        "end": round(duration, 2),
        "speaker": "SPEAKER_00",
        "confidence": 1.0,
    }]


# ------------------------------------------------------------
# Segment smoothing helpers
# ------------------------------------------------------------
def _smooth_segments(
    segments: List[Dict],
    min_duration: float = 0.8,
    max_gap_merge: float = 0.4
) -> List[Dict]:
    """Merge tiny segments & same-speaker segments that are very close."""

    if not segments:
        return []

    segments = sorted(segments, key=lambda s: s["start"])
    smoothed: List[Dict] = []

    for seg in segments:
        duration = seg["end"] - seg["start"]

        if smoothed:
            last = smoothed[-1]
            gap = seg["start"] - last["end"]

            # Merge if same speaker and gap small
            if seg["speaker"] == last["speaker"] and gap <= max_gap_merge:
                last["end"] = max(last["end"], seg["end"])
                last["confidence"] = (last["confidence"] + seg["confidence"]) / 2.0
                continue

            # Merge tiny low-duration segments
            if duration < min_duration:
                last["end"] = max(last["end"], seg["end"])
                last["confidence"] = (last["confidence"] + seg["confidence"]) / 2.0
                continue

        smoothed.append(seg)

    return smoothed


# ------------------------------------------------------------
# Main Diarization Function
# ------------------------------------------------------------
def diarize_audio(wav_path: str) -> List[Dict]:
    """
    Run pyannote diarization (or fallback).
    Returns list of dicts:
    {
        "start": float,
        "end": float,
        "speaker": "SPEAKER_XX",
        "confidence": float
    }
    """
    logger.info(f"Running diarization for: {wav_path}")
    pipeline = load_diarization_pipeline()

    if pipeline is None:
        return _mock_diarization(wav_path)

    try:
        # Try forcing 2 speakers first (useful for conversations)
        try:
            diarization = pipeline(
                {"audio": wav_path},
                min_speakers=2,
                max_speakers=2
            )
        except Exception as e:
            logger.warning(f"Auto-speaker diarization fallback: {e}")
            diarization = pipeline({"audio": wav_path})

        raw_segments = []
        # itertracks yields (Segment, track_id, speaker_label)
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            raw_segments.append(
                {
                    "start": float(round(turn.start, 3)),
                    "end": float(round(turn.end, 3)),
                    "speaker": speaker,
                    "confidence": 1.0,  # Pyannote doesn't expose confidence per segment
                }
            )

        # First smoothing pass
        smoothed = _smooth_segments(raw_segments)

        logger.info(
            f"Raw segments: {len(raw_segments)}, "
            f"Smoothed segments: {len(smoothed)}, "
            f"Speakers detected: {len(set(s['speaker'] for s in smoothed))}"
        )

        return smoothed

    except Exception as e:
        logger.error(f"Diarization failed: {e}")
        return _mock_diarization(wav_path)