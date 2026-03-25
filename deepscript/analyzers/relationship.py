"""Relationship analyzer — communication health for family/partner/personal calls.

Privacy: Opt-in only. Requires --relationship-insights flag.
Framing: Positive growth suggestions, never judgmental.
Research: Gottman, NVC citations for every suggestion.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Optional

from deepscript.analyzers.base import AnalysisResult, BaseAnalyzer

if TYPE_CHECKING:
    from deepscript.llm.provider import LLMProvider

# Keywords for rule-based detection
VALIDATION_KEYWORDS = [
    "i understand", "that makes sense", "i hear you", "you're right",
    "good point", "i see what you mean", "that's fair", "i agree",
    "thank you", "thanks", "appreciate",
]

CRITICISM_KEYWORDS = [
    "you always", "you never", "why can't you", "you should",
    "what's wrong with you",
]

POSITIVE_KEYWORDS = [
    "love", "appreciate", "thank", "grateful", "wonderful",
    "amazing", "great", "proud", "happy", "glad", "enjoy",
]

NEGATIVE_KEYWORDS = [
    "annoying", "frustrated", "angry", "disappointed", "upset",
    "hate", "terrible", "awful", "stupid", "ridiculous",
]


class RelationshipAnalyzer(BaseAnalyzer):
    """Analyzes communication health in personal relationships."""

    def __init__(self, llm: Optional["LLMProvider"] = None) -> None:
        super().__init__(llm)

    @property
    def supported_types(self) -> list[str]:
        return ["family", "partner", "personal"]

    def analyze(self, transcript: dict[str, Any]) -> AnalysisResult:
        text = transcript.get("text", "")
        segments = transcript.get("segments", [])

        sections: dict[str, Any] = {}

        # Rule-based metrics (always available)
        sections["listening_balance"] = self._listening_balance(segments)
        sections["we_i_language"] = self._we_i_language(text, segments)
        sections["validation_moments"] = self._detect_validations(segments)
        sections["engagement_signals"] = self._engagement_signals(segments)

        # LLM-enhanced analysis
        if self.llm:
            llm_result = self._analyze_with_llm(text)
            if llm_result:
                sections["emotional_tone"] = llm_result.get("emotional_tone", {})
                sections["gottman_indicators"] = llm_result.get("gottman_indicators", [])
                sections["appreciation_ratio"] = llm_result.get("appreciation_ratio", {})
                sections["bids_for_connection"] = llm_result.get("bids_for_connection", [])
                sections["repair_attempts"] = llm_result.get("repair_attempts", [])
                sections["nvc_patterns"] = llm_result.get("nvc_patterns", {})
                sections["growth_suggestions"] = llm_result.get("growth_suggestions", [])
        else:
            # Rule-based approximations
            sections["appreciation_ratio"] = self._appreciation_ratio(text)
            sections["gottman_indicators"] = self._detect_horsemen_rule_based(segments)

        return AnalysisResult(call_type="relationship", sections=sections)

    def _listening_balance(
        self, segments: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Compute listening balance per speaker."""
        speaker_words: dict[str, int] = {}
        speaker_questions: dict[str, int] = {}

        for seg in segments:
            speaker = seg.get("speaker", "Unknown")
            text = seg.get("text", "").strip()
            words = len(text.split())
            speaker_words[speaker] = speaker_words.get(speaker, 0) + words
            if text.endswith("?"):
                speaker_questions[speaker] = speaker_questions.get(speaker, 0) + 1

        total = sum(speaker_words.values())
        result: dict[str, Any] = {"speakers": {}}
        for spk in sorted(speaker_words.keys()):
            result["speakers"][spk] = {
                "talk_ratio": round(speaker_words[spk] / total, 3) if total else 0,
                "questions_asked": speaker_questions.get(spk, 0),
            }
        return result

    def _we_i_language(
        self, text: str, segments: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Analyze we/I language ratio per speaker."""
        result: dict[str, dict[str, int]] = {}

        for seg in segments:
            speaker = seg.get("speaker", "Unknown")
            seg_text = seg.get("text", "").lower()
            if speaker not in result:
                result[speaker] = {"we": 0, "i": 0}

            result[speaker]["we"] += len(re.findall(r"\bwe\b", seg_text))
            result[speaker]["i"] += len(re.findall(r"\bi\b", seg_text))

        summary: dict[str, Any] = {}
        for spk, counts in result.items():
            total = counts["we"] + counts["i"]
            summary[spk] = {
                "we_count": counts["we"],
                "i_count": counts["i"],
                "collaborative_ratio": round(counts["we"] / total, 3) if total else 0,
            }
        return summary

    def _detect_validations(
        self, segments: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Detect validation moments."""
        validations: list[dict[str, Any]] = []
        for seg in segments:
            text = seg.get("text", "").lower()
            for kw in VALIDATION_KEYWORDS:
                if kw in text:
                    validations.append({
                        "text": seg.get("text", "").strip(),
                        "speaker": seg.get("speaker", "Unknown"),
                        "keyword": kw,
                        "timestamp": seg.get("start"),
                    })
                    break
        return validations

    def _engagement_signals(
        self, segments: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Track engagement via word count per turn trends."""
        speaker_turns: dict[str, list[int]] = {}
        for seg in segments:
            speaker = seg.get("speaker", "Unknown")
            words = len(seg.get("text", "").split())
            if speaker not in speaker_turns:
                speaker_turns[speaker] = []
            speaker_turns[speaker].append(words)

        result: dict[str, Any] = {}
        for spk, turns in speaker_turns.items():
            avg = sum(turns) / len(turns) if turns else 0
            one_word = sum(1 for t in turns if t <= 2)
            result[spk] = {
                "avg_words_per_turn": round(avg, 1),
                "one_word_answers": one_word,
                "total_turns": len(turns),
            }
        return result

    def _appreciation_ratio(self, text: str) -> dict[str, Any]:
        """Rule-based appreciation ratio from keyword counting."""
        text_lower = text.lower()
        positive = sum(1 for kw in POSITIVE_KEYWORDS if kw in text_lower)
        negative = sum(1 for kw in NEGATIVE_KEYWORDS if kw in text_lower)
        ratio = f"{positive}:{negative}" if negative > 0 else f"{positive}:0"

        if negative == 0:
            assessment = "healthy" if positive > 0 else "neutral"
        else:
            r = positive / negative
            if r >= 5:
                assessment = "healthy (5:1+)"
            elif r >= 3:
                assessment = "approaching (3:1-5:1)"
            else:
                assessment = "concerning (<3:1)"

        return {
            "positive_count": positive,
            "negative_count": negative,
            "ratio": ratio,
            "assessment": assessment,
        }

    def _detect_horsemen_rule_based(
        self, segments: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Rule-based Gottman Four Horsemen detection."""
        indicators: list[dict[str, Any]] = []
        for seg in segments:
            text = seg.get("text", "").strip()
            text_lower = text.lower()
            speaker = seg.get("speaker", "Unknown")

            for kw in CRITICISM_KEYWORDS:
                if kw in text_lower:
                    indicators.append({
                        "type": "criticism",
                        "quote": text,
                        "speaker": speaker,
                        "timestamp": seg.get("start"),
                    })
                    break

        return indicators

    def _analyze_with_llm(self, text: str) -> dict[str, Any] | None:
        """Full LLM-based relationship analysis."""
        if not self.llm:
            return None

        truncated = text if len(text.split()) <= 4000 else " ".join(text.split()[:4000])
        prompt = self.llm.render_prompt("relationship", transcript=truncated)
        return self.llm.complete_json(prompt)
