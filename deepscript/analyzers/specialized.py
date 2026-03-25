"""Specialized analyzers — podcast, therapy, medical, legal, fundraising, voice memo."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from deepscript.analyzers.base import AnalysisResult, BaseAnalyzer
from deepscript.analyzers.business import BusinessAnalyzer

if TYPE_CHECKING:
    from deepscript.llm.provider import LLMProvider


class PodcastAnalyzer(BaseAnalyzer):
    """Analyzes podcast/interview recordings for show notes."""

    def __init__(self, llm: Optional["LLMProvider"] = None) -> None:
        super().__init__(llm)
        self._business = BusinessAnalyzer(llm)

    @property
    def supported_types(self) -> list[str]:
        return ["podcast"]

    def analyze(self, transcript: dict[str, Any]) -> AnalysisResult:
        text = transcript.get("text", "")
        segments = transcript.get("segments", [])
        base = self._business.analyze(transcript)
        sections = dict(base.sections)

        # Extract key quotes (longest statements)
        quotes = []
        for seg in segments:
            words = len(seg.get("text", "").split())
            if words > 20:
                quotes.append({
                    "text": seg["text"].strip(),
                    "speaker": seg.get("speaker", "Unknown"),
                    "timestamp": seg.get("start"),
                })
        quotes.sort(key=lambda q: len(q["text"]), reverse=True)
        sections["key_quotes"] = quotes[:5]

        # Guest detection (speaker with less talk time in 2-speaker conversation)
        speakers = {}
        for seg in segments:
            spk = seg.get("speaker", "Unknown")
            speakers[spk] = speakers.get(spk, 0) + len(seg.get("text", "").split())
        if len(speakers) == 2:
            sorted_speakers = sorted(speakers.items(), key=lambda x: x[1])
            sections["guest"] = sorted_speakers[0][0]
            sections["host"] = sorted_speakers[1][0]

        return AnalysisResult(call_type="podcast", sections=sections)


class TherapyAnalyzer(BaseAnalyzer):
    """Analyzes therapy/counseling sessions. Private, client-only output."""

    def __init__(self, llm: Optional["LLMProvider"] = None) -> None:
        super().__init__(llm)
        self._business = BusinessAnalyzer(llm)

    @property
    def supported_types(self) -> list[str]:
        return ["therapy-session"]

    def analyze(self, transcript: dict[str, Any]) -> AnalysisResult:
        text = transcript.get("text", "")
        base = self._business.analyze(transcript)
        sections = dict(base.sections)
        text_lower = text.lower()

        # Coping strategy mentions
        coping = ["breathing", "mindfulness", "journal", "exercise", "meditation", "grounding", "self-care"]
        sections["coping_strategies"] = [kw for kw in coping if kw in text_lower]

        # Emotion tracking
        emotions = ["anxious", "depressed", "angry", "sad", "hopeful", "grateful", "calm", "overwhelmed"]
        sections["emotions_mentioned"] = [kw for kw in emotions if kw in text_lower]

        # Homework tracking
        homework = ["homework", "practice", "try this", "between sessions", "this week"]
        sections["homework_signals"] = [kw for kw in homework if kw in text_lower]

        return AnalysisResult(call_type="therapy-session", sections=sections)


class SimpleAnalyzer(BaseAnalyzer):
    """Minimal analyzer for voice memos and unclassified transcripts."""

    def __init__(self, llm: Optional["LLMProvider"] = None) -> None:
        super().__init__(llm)
        self._business = BusinessAnalyzer(llm)

    @property
    def supported_types(self) -> list[str]:
        return ["voice-memo", "medical-appointment", "legal-consultation",
                "fundraising-donor", "earnings-call", "due-diligence"]

    def analyze(self, transcript: dict[str, Any]) -> AnalysisResult:
        base = self._business.analyze(transcript)
        return base
