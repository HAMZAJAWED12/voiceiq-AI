# app/services/pdf_service.py

from typing import List, Dict, Optional
from fpdf import FPDF
from app.utils.logger import logger


# ============================================================
# SAFETY HELPERS
# ============================================================

PAGE_WIDTH = 190   # usable width (A4 with margins)

def break_long_words(text: str, max_len: int = 50) -> str:
    """
    Breaks extremely long 'words' to avoid overflow in FPDF.
    """
    parts = text.split()
    fixed = []
    for p in parts:
        if len(p) > max_len:
            for i in range(0, len(p), max_len):
                fixed.append(p[i:i + max_len])
        else:
            fixed.append(p)
    return " ".join(fixed)


def safe_text(text: str) -> str:
    """
    Cleans input:
      - remove newlines
      - split words
      - break long chunks
    """
    if not isinstance(text, str):
        return ""

    text = text.replace("\n", " ").strip()

    safe_words = []
    for w in text.split(" "):
        if len(w) > 40:
            chunks = [w[i:i + 40] for i in range(0, len(w), 40)]
            safe_words.extend(chunks)
        else:
            safe_words.append(w)

    return " ".join(safe_words)


def safe_multicell(pdf: FPDF, text: str, h: float = 5):
    """
    Protective wrapper around FPDF.multi_cell:
      - Always uses PAGE_WIDTH (190)
      - On overflow: splits
    """
    txt = safe_text(text)
    txt = break_long_words(txt)

    try:
        pdf.multi_cell(PAGE_WIDTH, h, txt)
        return
    except Exception:
        pass

    # fallback: break by words
    for line in txt.split(" "):
        try:
            pdf.multi_cell(PAGE_WIDTH, h, line)
        except Exception:
            # final fallback: character by character
            for ch in line:
                try:
                    pdf.multi_cell(PAGE_WIDTH, h, ch)
                except Exception:
                    pdf.multi_cell(PAGE_WIDTH, h, "?")


# ============================================================
# MAIN SERVICE
# ============================================================

class PDFService:

    @staticmethod
    def generate_pdf_report(
        transcript: str,
        speaker_segments: List[Dict],
        summary: str,
        topic: str,
        conversation_stats: Dict,
        speaker_stats: Dict,
        emotion_overview: Optional[Dict[str, Dict[str, float]]] = None,
        intents_summary: Optional[Dict[str, int]] = None,
        flags: Optional[List[Dict]] = None,
        fact_checks: Optional[List[Dict]] = None,
    ) -> bytes:

        logger.info("Generating PDF report with advanced analytics...")

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=10)
        pdf.set_left_margin(10)
        pdf.set_right_margin(10)

        # --------------------------------------------------------
        # TITLE
        # --------------------------------------------------------
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 18)
        safe_multicell(pdf, "Conversation Analysis Report")
        pdf.ln(4)

        # --------------------------------------------------------
        # SUMMARY + TOPIC
        # --------------------------------------------------------
        pdf.set_font("Helvetica", "B", 12)
        safe_multicell(pdf, f"Topic: {topic or 'N/A'}")
        pdf.ln(2)

        pdf.set_font("Helvetica", "B", 12)
        safe_multicell(pdf, "Summary")
        pdf.set_font("Helvetica", "", 10)
        safe_multicell(pdf, summary or "N/A")
        pdf.ln(4)

        # --------------------------------------------------------
        # CONVERSATION STATS
        # --------------------------------------------------------
        pdf.set_font("Helvetica", "B", 12)
        safe_multicell(pdf, "Conversation Statistics")
        pdf.set_font("Helvetica", "", 10)

        for k, v in (conversation_stats or {}).items():
            safe_multicell(pdf, f"- {k.replace('_', ' ').title()}: {v}")
        pdf.ln(3)

        # --------------------------------------------------------
        # SPEAKER STATS
        # --------------------------------------------------------
        pdf.set_font("Helvetica", "B", 12)
        safe_multicell(pdf, "Speaker Statistics")
        pdf.set_font("Helvetica", "", 10)

        for speaker, stats in (speaker_stats or {}).items():
            safe_multicell(pdf, f"{speaker}:")
            for k, v in (stats or {}).items():
                safe_multicell(pdf, f"   - {k.replace('_', ' ').title()}: {v}")
        pdf.ln(3)

        # --------------------------------------------------------
        # EMOTIONS
        # --------------------------------------------------------
        if emotion_overview:
            pdf.set_font("Helvetica", "B", 12)
            safe_multicell(pdf, "Emotion Overview (per Speaker)")
            pdf.set_font("Helvetica", "", 10)

            for spk, emos in emotion_overview.items():
                safe_multicell(pdf, f"{spk}:")
                line = ", ".join(f"{e}: {round(score * 100)}%" for e, score in emos.items())
                safe_multicell(pdf, "   " + line)
            pdf.ln(3)

        # --------------------------------------------------------
        # INTENTS
        # --------------------------------------------------------
        if intents_summary:
            pdf.set_font("Helvetica", "B", 12)
            safe_multicell(pdf, "Intent Distribution")
            pdf.set_font("Helvetica", "", 10)

            for intent, count in intents_summary.items():
                safe_multicell(pdf, f"- {intent}: {count}")
            pdf.ln(3)

        # --------------------------------------------------------
        # FLAGS
        # --------------------------------------------------------
        if flags:
            pdf.set_font("Helvetica", "B", 12)
            safe_multicell(pdf, "Flags (Hesitation / Aggression / Lie Risk)")
            pdf.set_font("Helvetica", "", 9)

            for f in flags:
                line = (
                    f"[{f.get('type')}] {f.get('speaker')} "
                    f"({f.get('start'):.1f}-{f.get('end'):.1f}s) "
                    f"score={f.get('score')}: {f.get('note')}"
                )
                safe_multicell(pdf, line)
            pdf.ln(3)

        # --------------------------------------------------------
        # FACT CHECKS
        # --------------------------------------------------------
        if fact_checks:
            pdf.set_font("Helvetica", "B", 12)
            safe_multicell(pdf, "Fact-Check Candidates")
            pdf.set_font("Helvetica", "", 9)

            for fc in fact_checks:
                safe_multicell(
                    pdf,
                    f"- [{fc.get('type')}] {fc.get('value')} => {fc.get('status')} ({fc.get('note')})"
                )
            pdf.ln(3)

        # --------------------------------------------------------
        # TRANSCRIPT
        # --------------------------------------------------------
        pdf.set_font("Helvetica", "B", 12)
        safe_multicell(pdf, "Transcript (truncated if long)")
        pdf.set_font("Helvetica", "", 8)

        t = transcript or ""
        max_len = 4000
        if len(t) > max_len:
            t = t[:max_len] + "... [truncated]"

        safe_multicell(pdf, t)

        pdf.ln(4)

        # --------------------------------------------------------
        # EXPORT
        # --------------------------------------------------------
        pdf_bytes = pdf.output(dest="S")
        if isinstance(pdf_bytes, bytearray):
            pdf_bytes = bytes(pdf_bytes)

        return pdf_bytes


    @staticmethod
    def to_base64(pdf_bytes: bytes) -> str:
        import base64
        return base64.b64encode(pdf_bytes).decode("ascii")