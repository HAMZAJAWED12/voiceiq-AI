# app/routes/process_audio.py

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import tempfile
import os
import uuid
import base64

from app.utils.audio_utils import normalize_to_wav
from app.utils.logger import logger

from app.services.asr_service import transcribe_local
from app.services.diarization_service import diarize_audio
from app.services.alignment_service import (
    align_transcript_with_speakers,
    build_conversation,
)

from app.services.metadata_service import MetadataExtractor
from app.services.sentiment_service import SentimentService
from app.services.keyword_service import KeywordService
from app.services.topic_service import TopicService
from app.services.summary_service import SummaryService
from app.services.gender_service import GenderService
from app.services.pdf_service import PDFService
from app.services.emotion_service import EmotionService
from app.services.intent_service import IntentService
from app.services.factcheck_service import FactCheckService
from app.services.flag_service import FlagService



router = APIRouter()


# --------------------------
# Response Models
# --------------------------

class ASRMeta(BaseModel):
    model: Optional[str]
    language: Optional[str]
    duration: Optional[float]
    segments: Optional[List[Dict]]


class SpeakerSegment(BaseModel):
    start: float
    end: float
    speaker: str
    text: Optional[str]

    sentiment: Optional[str] = None
    sentiment_score: Optional[float] = None
    keywords: Optional[List[str]] = None

    confidence: Optional[float] = None
    gender: Optional[str] = None
    gender_confidence: Optional[float] = None


class SpeakerStats(BaseModel):
    total_speaking_time: float
    segment_count: int
    total_words: int
    longest_monologue: float
    first_spoke_at: float
    last_spoke_at: float
    avg_segment_length: float
    wpm: float
    speaking_ratio: float
    word_ratio: float


class ConversationStats(BaseModel):
    total_duration: float
    total_segments: int
    total_words: int
    avg_turn_length: float
    speaker_count: int
    conversation_start: float
    conversation_end: float


class TopicInfo(BaseModel):
    topic: str
    confidence: float


# add near top with other models

class FlagItem(BaseModel):
    type: str
    speaker: str
    start: float
    end: float
    text: str
    score: float
    note: Optional[str] = None


class FactCheckItem(BaseModel):
    type: str
    value: str
    status: str
    source: Optional[str] = None
    note: Optional[str] = None


class ProcessAudioResponse(BaseModel):
    request_id: str
    transcript: str
    asr_meta: ASRMeta

    segments: List[Dict]
    speaker_segments: List[SpeakerSegment]
    conversation: List[SpeakerSegment]

    speaker_stats: Dict[str, SpeakerStats]
    conversation_stats: ConversationStats

    topic: TopicInfo

    summary: Optional[str] = None
    report_pdf_base64: Optional[str] = None

    # NEW
    intents_summary: Optional[Dict[str, int]] = None
    fact_checks: Optional[List[FactCheckItem]] = None
    flags: Optional[List[FlagItem]] = None
    timeline: Optional[List[Dict]] = None
    emotion_overview: Optional[Dict[str, Dict[str, float]]] = None


# --------------------------
# Main Route
# --------------------------

@router.post("/process-audio", response_model=ProcessAudioResponse)
async def process_audio(file: UploadFile = File(...)):
    request_id = str(uuid.uuid4())
    logger.info(f"[{request_id}] Received: {file.filename}")

    # Validate file type
    if not file.filename.lower().endswith((".mp3", ".wav", ".m4a", ".flac")):
        raise HTTPException(status_code=400, detail="Unsupported file format")

    # Temporary workspace
    tmpdir = tempfile.mkdtemp()
    in_path = os.path.join(tmpdir, file.filename)
    wav_path = os.path.join(tmpdir, "normalized.wav")

    try:
        # Save audio file
        with open(in_path, "wb") as f:
            f.write(await file.read())

        # Normalize audio
        normalize_to_wav(in_path, wav_path, sr=16000)

        # Step 1: Transcription
        text, meta = transcribe_local(wav_path)

        # Step 2: Diarization
        segments = diarize_audio(wav_path)

        # Step 3: Alignment
        try:
            asr_payload = {
                "text": text,
                "meta": meta,
                "segments": meta.get("segments", []),
            }
            aligned = align_transcript_with_speakers(asr_payload, segments)
            speaker_segments = aligned.get("speaker_segments", [])
        except Exception as e:
            logger.error(f"Alignment failed: {e}")
            speaker_segments = []

        # Step 4: Build conversation view
        try:
            conversation = build_conversation(
                {"text": text, "meta": meta, "segments": meta.get("segments", [])},
                segments,
            )
        except Exception as e:
            logger.error(f"Conversation build failed: {e}")
            conversation = []

        # Step 5: Stats
        speaker_stats = MetadataExtractor.compute_speaker_stats(speaker_segments)
        conversation_stats = MetadataExtractor.compute_conversation_stats(
            speaker_segments, segments
        )

        # Step 6: Sentiment + Keywords + Gender
        if speaker_segments:
            speaker_segments = SentimentService.analyze_speaker_segments(speaker_segments)
            speaker_segments = KeywordService.extract_keywords_per_segment(speaker_segments)
            speaker_segments = GenderService.add_gender_to_segments(speaker_segments, wav_path)

        # NEW Step 6b: Emotion (audio/text)
        if speaker_segments:
            speaker_segments = EmotionService.analyze_speaker_segments(wav_path, speaker_segments)
            emotion_overview = EmotionService.summarize_emotions(speaker_segments)
        else:
            emotion_overview = {}

        # Step 7: Topic detection
        topic = TopicService.classify(text or "")

        # Step X: Generate summary
        summary = SummaryService.generate_summary(text or "")

        # NEW: Intent classification (use conversation if available)
        if conversation:
            conversation_with_intents = IntentService.annotate_conversation(conversation)
        else:
            # fallback: build from speaker_segments
            conv_fallback = []
            for seg in speaker_segments or []:
                conv_fallback.append(
                    {
                        "start": seg.get("start", 0.0),
                        "end": seg.get("end", 0.0),
                        "speaker": seg.get("speaker", "UNKNOWN"),
                        "text": seg.get("text", ""),
                    }
                )
            conversation_with_intents = IntentService.annotate_conversation(conv_fallback)

        intents_summary = IntentService.summarize_intents(conversation_with_intents)

        # NEW: Fact-check candidates from transcript
        fact_checks = FactCheckService.fact_check(text or "")

        # NEW: Flags (hesitation, aggression, lie-risk)
        flags = FlagService.generate_flags(conversation_with_intents)

        # NEW: Visual timeline structure (for UI charts)
        timeline = []
        for turn in conversation_with_intents:
            timeline.append(
                {
                    "start": turn.get("start", 0.0),
                    "end": turn.get("end", 0.0),
                    "speaker": turn.get("speaker", "UNKNOWN"),
                    "text": turn.get("text", ""),
                    "intent": turn.get("intent", "other"),
                }
            )

        # Step X: Generate PDF (extended)
        pdf_bytes = PDFService.generate_pdf_report(
            transcript=text,
            speaker_segments=speaker_segments,
            summary=summary,
            topic=topic.get("topic", ""),
            conversation_stats=conversation_stats,
            speaker_stats=speaker_stats,
            emotion_overview=emotion_overview,
            intents_summary=intents_summary,
            flags=flags,
            fact_checks=fact_checks,
        )

        pdf_b64 = PDFService.to_base64(pdf_bytes)


        return {
            "request_id": request_id,
            "transcript": text or "",
            "asr_meta": meta,
            "segments": segments or [],
            "speaker_segments": speaker_segments,
            "conversation": conversation_with_intents,  # now includes 'intent'
            "speaker_stats": speaker_stats,
            "conversation_stats": conversation_stats,
            "topic": topic,
            "summary": summary,
            "report_pdf_base64": pdf_b64,
            "intents_summary": intents_summary,
            "fact_checks": fact_checks,
            "flags": flags,
            "timeline": timeline,
            "emotion_overview": emotion_overview,
        }


    finally:
        # Cleanup
        try:
            if os.path.exists(in_path): os.remove(in_path)
            if os.path.exists(wav_path): os.remove(wav_path)
            os.rmdir(tmpdir)
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")