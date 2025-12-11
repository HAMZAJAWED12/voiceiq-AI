# app/services/alignment_service.py

from typing import List, Dict, Any
import math
from app.utils.logger import logger


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _confidence_from_whisper(seg: Dict) -> float:
    """
    Convert Whisper segment metadata to a clean [0,1] confidence score.
    Uses:
      avg_logprob → typical ~ [-1,0]
      no_speech_prob → penalty
    """
    avg_logprob = seg.get("avg_logprob", -1.0)
    no_speech_prob = seg.get("no_speech_prob", 0.0)

    # logistic mapping from logprob
    conf = 1.0 / (1.0 + math.exp(-avg_logprob))
    conf *= (1.0 - no_speech_prob)

    return max(0.0, min(1.0, conf))


def _overlap(a_start, a_end, b_start, b_end) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _best_diar_for_asr(asr_seg: Dict, diar: List[Dict]) -> Dict:
    """
    Find which diarization segment overlaps the ASR segment the most.
    """
    best = None
    best_overlap = 0.0
    s_start = float(asr_seg["start"])
    s_end = float(asr_seg["end"])

    for d in diar:
        d_start, d_end = float(d["start"]), float(d["end"])
        ov = _overlap(s_start, s_end, d_start, d_end)
        if ov > best_overlap:
            best_overlap = ov
            best = d

    return best


# ------------------------------------------------------------
# Main Enhancer
# ------------------------------------------------------------
class EnhancedAligner:

    # --------------------------------------------------------
    # 1. Extract ASR Segments
    # --------------------------------------------------------
    @staticmethod
    def _extract_asr_segments(asr_result: Any) -> List[Dict]:
        """
        Normalise ASR output from Whisper into a list of segments:
        [
          { start, end, text, avg_logprob, no_speech_prob }
        ]
        """
        segs = None

        # Whisper dict style
        if isinstance(asr_result, dict):

            if isinstance(asr_result.get("segments"), list):
                segs = asr_result["segments"]

            elif isinstance(asr_result.get("meta"), dict) and isinstance(
                asr_result["meta"].get("segments"), list
            ):
                segs = asr_result["meta"]["segments"]

        # Already list
        elif isinstance(asr_result, list):
            segs = asr_result

        if not segs:
            return []

        norm = []
        for s in segs:
            if "start" not in s or "end" not in s:
                continue
            norm.append(
                {
                    "start": float(s["start"]),
                    "end": float(s["end"]),
                    "text": (s.get("text") or "").strip(),
                    "avg_logprob": s.get("avg_logprob", -1.0),
                    "no_speech_prob": s.get("no_speech_prob", 0.0),
                }
            )

        return norm

    # --------------------------------------------------------
    # 2. Word slicing (optional but safe)
    # --------------------------------------------------------
    @staticmethod
    def _to_word_segments(asr_segments: List[Dict]) -> List[Dict]:
        """
        Approximate word timings by uniformly slicing the segment duration.
        """
        words_all = []

        for seg in asr_segments:
            text = seg.get("text", "")
            if not text:
                continue

            start = seg["start"]
            end = seg["end"]
            duration = max(end - start, 1e-6)

            words = text.split()
            if not words:
                continue

            w_dur = duration / len(words)

            for i, w in enumerate(words):
                w_start = start + i * w_dur
                w_end = end if i == len(words) - 1 else start + (i + 1) * w_dur

                words_all.append(
                    {
                        "start": w_start,
                        "end": w_end,
                        "word": w,
                    }
                )

        return words_all

    # --------------------------------------------------------
    # 3. Map Words → Speakers (core step)
    # --------------------------------------------------------
    @staticmethod
    def _align_words_to_diarization(
        word_segments: List[Dict],
        diarization: List[Dict],
    ) -> List[Dict]:
        """
        Maps each diarization segment to the text produced by all words
        falling inside that segment's time span.
        """
        aligned = []

        if not word_segments or not diarization:
            return aligned

        # Sort for stability
        word_segments = sorted(word_segments, key=lambda x: x["start"])
        diarization = sorted(diarization, key=lambda x: x["start"])

        for d in diarization:
            d_start, d_end = d["start"], d["end"]
            speaker = d["speaker"]

            words = [
                w for w in word_segments
                if not (w["end"] <= d_start or w["start"] >= d_end)
            ]

            if not words:
                continue

            text = " ".join(w["word"] for w in words)

            aligned.append(
                {
                    "start": round(d_start, 3),
                    "end": round(d_end, 3),
                    "speaker": speaker,
                    "text": text.strip(),
                }
            )

        return aligned

    # --------------------------------------------------------
    # 4. Merge blocks
    # --------------------------------------------------------
    @staticmethod
    def _merge_blocks(segments: List[Dict], max_gap: float = 0.75) -> List[Dict]:
        if not segments:
            return []

        segs = sorted(segments, key=lambda x: x["start"])
        merged = [segs[0]]

        for seg in segs[1:]:
            last = merged[-1]
            gap = seg["start"] - last["end"]

            if seg["speaker"] == last["speaker"] and 0 <= gap <= max_gap:
                last["end"] = seg["end"]
                last["text"] = (last["text"] + " " + seg["text"]).strip()
            else:
                merged.append(seg)

        return merged

    # --------------------------------------------------------
    # 5. Public API: align()
    # --------------------------------------------------------
    @classmethod
    def align(
        cls,
        asr_result: Any,
        diarization_result: List[Dict],
    ) -> Dict[str, List[Dict]]:
        diar = diarization_result or []
        asr = cls._extract_asr_segments(asr_result)

        if not asr or not diar:
            logger.warning("Alignment failed: missing ASR or diarization segments.")
            return {"speaker_segments": []}

        # Word slicing
        words = cls._to_word_segments(asr)

        # Map → diar windows
        raw = cls._align_words_to_diarization(words, diar)

        # Merge small gaps
        merged = cls._merge_blocks(raw)

        # Attach ASR-based confidence
        for seg in merged:
            # best ASR segment that overlaps this diar block
            best = _best_diar_for_asr(
                {"start": seg["start"], "end": seg["end"]}, asr
            )
            if best:
                seg["confidence"] = _confidence_from_whisper(best)
            else:
                seg["confidence"] = 0.5

            # placeholders for gender (set later in pipeline)
            seg["gender"] = None
            seg["gender_confidence"] = None

        logger.info(
            f"EnhancedAligner: produced {len(merged)} speaker segments "
            f"from {len(asr)} ASR segments and {len(diar)} diarization segments."
        )

        return {"speaker_segments": merged}


# ------------------------------------------------------------
# Wrapper for external code (your FastAPI calls this)
# ------------------------------------------------------------
def align_transcript_with_speakers(asr_result: Any, diarization_result: List[Dict]):
    return EnhancedAligner.align(asr_result, diarization_result)


def build_conversation(asr_result: Any, diarization_result: List[Dict]) -> List[Dict]:
    """
    Higher-level timeline builder:
    Converts speaker_segments into clean "conversation turns".

    - Uses merged segments
    - Assigns CUSTOMER/AGENT roles automatically
    - Merges tiny gaps
    - Produces chronological blocks
    """
    aligned = EnhancedAligner.align(asr_result, diarization_result)
    segments = aligned.get("speaker_segments", [])
    if not segments:
        return []

    segments = sorted(segments, key=lambda s: s["start"])

    # Compute speaking time per speaker
    time_map = {}
    for seg in segments:
        dur = seg["end"] - seg["start"]
        time_map[seg["speaker"]] = time_map.get(seg["speaker"], 0) + dur

    speaker_order = sorted(time_map.items(), key=lambda x: x[1], reverse=True)

    if len(speaker_order) >= 2:
        customer = speaker_order[0][0]
        agent = speaker_order[1][0]
        roles = {customer: "CUSTOMER", agent: "AGENT"}
    else:
        roles = {speaker_order[0][0]: "CUSTOMER"}

    # Build conversation timeline
    conv = []
    current = None
    MAX_GAP = 0.75

    for seg in segments:
        spk = seg["speaker"]
        role = roles.get(spk, spk)

        if (
            current
            and current["speaker_raw"] == spk
            and seg["start"] - current["end"] <= MAX_GAP
        ):
            current["end"] = seg["end"]
            current["text"] += " " + seg["text"]
        else:
            if current:
                conv.append(current)
            current = {
                "start": seg["start"],
                "end": seg["end"],
                "speaker": role,
                "speaker_raw": spk,
                "text": seg["text"],
            }

    if current:
        conv.append(current)

    return conv