"""Business meeting analyzer — rule-based with optional LLM enhancement."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Optional

from deepscript.analyzers.base import AnalysisResult, BaseAnalyzer

if TYPE_CHECKING:
    from deepscript.llm.provider import LLMProvider

ACTION_PATTERNS = [
    re.compile(r"\b(?:action item|todo|to do|task)\b[:\s]*(.+)", re.IGNORECASE),
    re.compile(r"\b(?:will|going to|need to|should|must)\s+(.+?)(?:\.|$)", re.IGNORECASE),
    re.compile(r"\b(?:follow up|follow-up)\b[:\s]*(.+?)(?:\.|$)", re.IGNORECASE),
    re.compile(r"\b(?:let's|let us)\s+(.+?)(?:\.|$)", re.IGNORECASE),
]

DECISION_PATTERNS = [
    re.compile(r"\b(?:decided|agreed|decision is|we'll go with|the plan is)\b[:\s]*(.+?)(?:\.|$)", re.IGNORECASE),
    re.compile(r"\b(?:we're going to|we are going to|consensus is)\b[:\s]*(.+?)(?:\.|$)", re.IGNORECASE),
    re.compile(r"\b(?:approved|confirmed|finalized)\b[:\s]*(.+?)(?:\.|$)", re.IGNORECASE),
]

TRANSITION_PHRASES = [
    "moving on", "next topic", "let's talk about", "now regarding",
    "turning to", "let's move on", "another thing",
]


class BusinessAnalyzer(BaseAnalyzer):
    """Extracts meeting intelligence: summary, action items, decisions, questions."""

    classification_keywords = {
        "business-meeting": [
            "agenda", "action item", "follow up", "next steps", "let's discuss",
            "any questions", "moving on", "take away", "meeting", "quarterly",
            "review", "update", "status",
        ],
    }

    def __init__(self, llm: Optional["LLMProvider"] = None) -> None:
        super().__init__(llm)

    @property
    def supported_types(self) -> list[str]:
        return ["business-meeting", "standup", "unknown"]

    def analyze(self, transcript: dict[str, Any]) -> AnalysisResult:
        # Try single-call combined analysis first (saves 2-3 LLM calls)
        if self.llm:
            combined = self.analyze_combined(transcript, "business-meeting")
            if combined and combined.sections.get("summary"):
                # Add attendees (always rule-based, needs segments)
                combined.sections["attendees"] = self._extract_attendees(transcript.get("segments", []))
                return combined

        # Multi-call fallback
        text = transcript.get("text", "")
        segments = transcript.get("segments", [])

        sections: dict[str, Any] = {}
        sections["summary"] = self._extract_summary(text, segments)
        sections["action_items"] = self._extract_action_items(text, segments)
        sections["decisions"] = self._extract_decisions(text, segments)
        sections["questions"] = self._extract_questions(text, segments)
        sections["attendees"] = self._extract_attendees(segments)

        return AnalysisResult(call_type="business-meeting", sections=sections)

    def _extract_summary(
        self, text: str, segments: list[dict[str, Any]]
    ) -> dict[str, Any]:
        word_count = len(text.split())
        duration = None
        if segments:
            last_end = max((s.get("end", 0) for s in segments), default=0)
            first_start = min((s.get("start", 0) for s in segments), default=0)
            duration = round(last_end - first_start, 1)

        # LLM abstractive summary
        if self.llm:
            truncated = text if len(text.split()) <= 4000 else " ".join(text.split()[:4000]) + "\n[...truncated...]"
            prompt = self.llm.render_prompt("summarize", transcript=truncated)
            llm_summary = self.llm.complete(prompt, max_tokens=512)
            if llm_summary:
                return {
                    "text": llm_summary.strip(),
                    "word_count": word_count,
                    "duration_seconds": duration,
                    "method": "llm",
                }

        # Rule-based extractive summary
        sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        opening = " ".join(sentences[:3]) if sentences else text[:200]
        topic_shifts = sum(
            1 for s in sentences if any(tp in s.lower() for tp in TRANSITION_PHRASES)
        )

        return {
            "text": opening,
            "word_count": word_count,
            "duration_seconds": duration,
            "topic_shifts": topic_shifts,
            "method": "rule-based",
        }

    def _extract_action_items(
        self, text: str, segments: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        # LLM extraction
        if self.llm:
            truncated = text if len(text.split()) <= 4000 else " ".join(text.split()[:4000])
            prompt = self.llm.render_prompt("action_items", transcript=truncated)
            result = self.llm.complete_json(prompt)
            if result and isinstance(result, list):
                return result

        # Rule-based fallback
        return self._extract_action_items_rule_based(segments)

    def _extract_action_items_rule_based(
        self, segments: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        seen: set[str] = set()
        for seg in segments:
            seg_text = seg.get("text", "").strip()
            speaker = seg.get("speaker", "Unknown")
            timestamp = seg.get("start")
            for pattern in ACTION_PATTERNS:
                for match in pattern.finditer(seg_text):
                    action_text = match.group(1).strip()
                    if len(action_text) < 5:
                        continue
                    normalized = action_text.lower()[:50]
                    if normalized in seen:
                        continue
                    seen.add(normalized)
                    items.append({
                        "text": action_text,
                        "speaker": speaker,
                        "timestamp": timestamp,
                    })
        return items

    def _extract_decisions(
        self, text: str, segments: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        # LLM extraction
        if self.llm:
            truncated = text if len(text.split()) <= 4000 else " ".join(text.split()[:4000])
            prompt = self.llm.render_prompt("decisions", transcript=truncated)
            result = self.llm.complete_json(prompt)
            if result and isinstance(result, list):
                return result

        # Rule-based fallback
        decisions: list[dict[str, Any]] = []
        seen: set[str] = set()
        for seg in segments:
            seg_text = seg.get("text", "").strip()
            speaker = seg.get("speaker", "Unknown")
            timestamp = seg.get("start")
            for pattern in DECISION_PATTERNS:
                for match in pattern.finditer(seg_text):
                    decision_text = match.group(1).strip()
                    if len(decision_text) < 5:
                        continue
                    normalized = decision_text.lower()[:50]
                    if normalized in seen:
                        continue
                    seen.add(normalized)
                    decisions.append({
                        "text": decision_text,
                        "speaker": speaker,
                        "timestamp": timestamp,
                    })
        return decisions

    def _extract_questions(
        self, text: str, segments: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        questions: list[dict[str, Any]] = []
        for seg in segments:
            seg_text = seg.get("text", "").strip()
            if not seg_text.endswith("?"):
                continue
            speaker = seg.get("speaker", "Unknown")
            timestamp = seg.get("start")
            questions.append({
                "text": seg_text,
                "speaker": speaker,
                "timestamp": timestamp,
            })
        return questions

    def _extract_attendees(
        self, segments: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        speaker_words: dict[str, int] = {}
        speaker_segs: dict[str, int] = {}
        for seg in segments:
            speaker = seg.get("speaker", "Unknown")
            words = len(seg.get("text", "").split())
            speaker_words[speaker] = speaker_words.get(speaker, 0) + words
            speaker_segs[speaker] = speaker_segs.get(speaker, 0) + 1
        total_words = sum(speaker_words.values())
        attendees = []
        for spk in sorted(speaker_words.keys()):
            wc = speaker_words[spk]
            attendees.append({
                "speaker": spk,
                "word_count": wc,
                "segment_count": speaker_segs[spk],
                "talk_ratio": round(wc / total_words, 3) if total_words > 0 else 0.0,
            })
        return attendees
