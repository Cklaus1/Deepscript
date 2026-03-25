"""Support call analyzer — issue classification, emotion trajectory, resolution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from deepscript.analyzers.base import AnalysisResult, BaseAnalyzer
from deepscript.analyzers.business import BusinessAnalyzer

if TYPE_CHECKING:
    from deepscript.llm.provider import LLMProvider

ISSUE_TYPE_KEYWORDS = {
    "bug": ["bug", "broken", "doesn't work", "error", "crash", "not working", "failing"],
    "feature_request": ["would be nice", "feature", "could you add", "wish it had", "missing"],
    "billing": ["charge", "invoice", "billing", "refund", "payment", "subscription", "plan"],
    "how_to": ["how do i", "how can i", "where do i", "can't find", "help me", "tutorial"],
    "outage": ["down", "outage", "unavailable", "can't access", "service disruption", "503"],
}

FRUSTRATION_KEYWORDS = ["frustrated", "angry", "unacceptable", "ridiculous", "terrible", "worst", "furious"]
SATISFACTION_KEYWORDS = ["thank you", "that helps", "perfect", "great", "appreciate", "solved", "wonderful"]
EMPATHY_KEYWORDS = ["i understand", "i'm sorry", "that must be", "let me help", "i appreciate your patience"]


class SupportAnalyzer(BaseAnalyzer):
    """Analyzes support calls for issue classification and resolution quality."""

    def __init__(self, llm: Optional["LLMProvider"] = None) -> None:
        super().__init__(llm)
        self._business = BusinessAnalyzer(llm)

    @property
    def supported_types(self) -> list[str]:
        return ["support-escalation", "support-call"]

    def analyze(self, transcript: dict[str, Any]) -> AnalysisResult:
        text = transcript.get("text", "")
        segments = transcript.get("segments", [])

        base = self._business.analyze(transcript)
        sections = dict(base.sections)

        if self.llm:
            llm_result = self._analyze_with_llm(text)
            if llm_result:
                sections["issue"] = llm_result.get("issue", {})
                sections["emotion_trajectory"] = llm_result.get("emotion_trajectory", {})
                sections["resolution"] = llm_result.get("resolution", {})
                sections["empathy_score"] = llm_result.get("empathy_score", {})
        else:
            sections.update(self._analyze_rule_based(text, segments))

        return AnalysisResult(call_type="support-escalation", sections=sections)

    def _analyze_with_llm(self, text: str) -> dict[str, Any] | None:
        if not self.llm:
            return None
        truncated = text if len(text.split()) <= 4000 else " ".join(text.split()[:4000])
        prompt = self.llm.render_prompt("support", transcript=truncated)
        return self.llm.complete_json(prompt)

    def _analyze_rule_based(
        self, text: str, segments: list[dict[str, Any]]
    ) -> dict[str, Any]:
        text_lower = text.lower()

        # Issue type detection
        issue_scores: dict[str, int] = {}
        for itype, keywords in ISSUE_TYPE_KEYWORDS.items():
            hits = sum(1 for kw in keywords if kw in text_lower)
            if hits:
                issue_scores[itype] = hits

        issue_type = max(issue_scores, key=issue_scores.get) if issue_scores else "other"

        # Emotion trajectory
        frustration_count = sum(1 for kw in FRUSTRATION_KEYWORDS if kw in text_lower)
        satisfaction_count = sum(1 for kw in SATISFACTION_KEYWORDS if kw in text_lower)

        if frustration_count > satisfaction_count:
            start_emotion = "frustrated"
            end_emotion = "frustrated" if satisfaction_count == 0 else "neutral"
        elif satisfaction_count > 0:
            start_emotion = "neutral"
            end_emotion = "satisfied"
        else:
            start_emotion = "neutral"
            end_emotion = "neutral"

        # Resolution detection
        resolution_keywords = ["resolved", "fixed", "solved", "working now", "that did it"]
        escalation_keywords = ["escalate", "manager", "supervisor", "engineering team"]
        resolved = any(kw in text_lower for kw in resolution_keywords)
        escalated = any(kw in text_lower for kw in escalation_keywords)

        if resolved:
            resolution_status = "resolved"
        elif escalated:
            resolution_status = "escalated"
        else:
            resolution_status = "unresolved"

        # Empathy score
        empathy_hits = sum(1 for kw in EMPATHY_KEYWORDS if kw in text_lower)
        empathy_score = min(empathy_hits * 2, 10)

        return {
            "issue": {
                "type": issue_type,
                "severity": "high" if frustration_count > 2 else "medium",
            },
            "emotion_trajectory": {
                "start": start_emotion,
                "end": end_emotion,
                "frustration_signals": frustration_count,
                "satisfaction_signals": satisfaction_count,
            },
            "resolution": {
                "status": resolution_status,
            },
            "empathy_score": {
                "score": empathy_score,
                "max": 10,
            },
            "method": "rule-based",
        }
