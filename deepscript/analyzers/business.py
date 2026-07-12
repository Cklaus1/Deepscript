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
        return ["business-meeting", "unknown"]

    def analyze(self, transcript: dict[str, Any]) -> AnalysisResult:
        # Try single-call combined analysis first (saves 2-3 LLM calls)
        if self.llm:
            combined = self.analyze_combined(transcript, "business-meeting")
            if combined and combined.sections.get("summary"):
                # Add attendees (always rule-based, needs segments)
                combined.sections["attendees"] = self._extract_attendees(transcript.get("segments", []))
                # Enrich action items — if any lack enrichment, re-extract all
                items = combined.sections.get("action_items", [])
                items = [i for i in items if isinstance(i, dict)]
                if items and any(not i.get("importance") for i in items):
                    text = transcript.get("text", "")
                    segments = transcript.get("segments", [])
                    combined.sections["action_items"] = self._extract_action_items(text, segments)
                elif items:
                    combined.sections["action_items"] = [
                        self._normalize_action_item(i) for i in items if i.get("text")
                    ]
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

    _VALID_IMPORTANCE = {"critical", "high", "medium", "low"}
    _VALID_COMPLEXITY = {"trivial", "easy", "medium", "hard", "complex"}

    def _extract_action_items(
        self, text: str, segments: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        # LLM extraction with enriched fields
        if self.llm:
            truncated = text if len(text.split()) <= 4000 else " ".join(text.split()[:4000])
            prompt = self.llm.render_prompt("action_items", transcript=truncated)
            result = self.llm.complete_json(prompt)
            if result and isinstance(result, list):
                items = [i for i in result if isinstance(i, dict)]
                return [self._normalize_action_item(item) for item in items if item.get("text")]

        # Rule-based fallback
        return self._extract_action_items_rule_based(segments)

    @classmethod
    def _normalize_action_item(cls, item: dict[str, Any]) -> dict[str, Any]:
        """Validate and normalize enriched action item fields from LLM output."""
        normalized = {
            "text": str(item.get("text", "")).strip(),
            "assignee": item.get("assignee") or None,
            "deadline": item.get("deadline") or None,
            "deadline_reasoning": item.get("deadline_reasoning") or None,
            "speaker": item.get("speaker") or "Unknown",
            "importance": "medium",
            "importance_reasoning": item.get("importance_reasoning") or None,
            "complexity": "medium",
            "tags": [],
            "risk": item.get("risk") or None,
            "dependencies": [],
            "next_action": item.get("next_action") or None,
            "source_quote": item.get("source_quote") or None,
            "commitment_strength": 0.65,
            "company": item.get("company") or None,
        }

        # Validate importance
        imp = str(item.get("importance", "")).lower().strip()
        if imp in cls._VALID_IMPORTANCE:
            normalized["importance"] = imp

        # Validate complexity
        cplx = str(item.get("complexity", "")).lower().strip()
        if cplx in cls._VALID_COMPLEXITY:
            normalized["complexity"] = cplx

        # Validate tags — must be list of strings
        tags = item.get("tags")
        if isinstance(tags, list):
            normalized["tags"] = [str(t).strip().lower() for t in tags if t][:10]

        # Validate dependencies — must be list of strings
        deps = item.get("dependencies")
        if isinstance(deps, list):
            normalized["dependencies"] = [str(d).strip() for d in deps if d]

        # Validate commitment_strength — must be 0.0-1.0
        cs = item.get("commitment_strength")
        if isinstance(cs, (int, float)):
            normalized["commitment_strength"] = min(1.0, max(0.0, float(cs)))

        # Truncate source_quote to 150 chars
        if normalized["source_quote"] and len(normalized["source_quote"]) > 150:
            normalized["source_quote"] = normalized["source_quote"][:147] + "..."

        return normalized

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
