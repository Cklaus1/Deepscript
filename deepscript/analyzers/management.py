"""Management meeting analyzers — 1:1, performance review, cofounder, advisory, board."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from deepscript.analyzers.base import AnalysisResult, BaseAnalyzer
from deepscript.analyzers.business import BusinessAnalyzer

if TYPE_CHECKING:
    from deepscript.llm.provider import LLMProvider

CAREER_KEYWORDS = ["career", "growth", "promotion", "development", "goals", "aspirations", "learning"]
STATUS_KEYWORDS = ["update", "progress", "status", "working on", "completed", "shipped"]
BLOCKER_KEYWORDS = ["blocked", "stuck", "need help", "waiting on", "dependency", "can't proceed"]
BURNOUT_KEYWORDS = ["overwhelmed", "stressed", "too much", "exhausted", "burned out", "workload"]
ALIGNMENT_KEYWORDS = ["vision", "direction", "strategy", "priorities", "disagree", "aligned", "misaligned"]
ADVICE_KEYWORDS = ["suggest", "recommend", "have you tried", "you should", "in my experience", "introduction"]


class ManagementAnalyzer(BaseAnalyzer):
    """Analyzes management meetings (1:1s, reviews, cofounder, advisory, board)."""

    def __init__(self, llm: Optional["LLMProvider"] = None, meeting_type: str = "one-on-one") -> None:
        super().__init__(llm)
        self.meeting_type = meeting_type
        self._business = BusinessAnalyzer(llm)

    @property
    def supported_types(self) -> list[str]:
        return ["one-on-one", "performance-review", "cofounder-alignment", "advisory-call", "board-meeting"]

    def analyze(self, transcript: dict[str, Any]) -> AnalysisResult:
        text = transcript.get("text", "")
        base = self._business.analyze(transcript)
        sections = dict(base.sections)

        if self.llm:
            llm_result = self._analyze_with_llm(text)
            if llm_result:
                sections["topic_coverage"] = llm_result.get("topic_coverage", [])
                sections["dynamics"] = llm_result.get("dynamics", {})
                sections["key_items"] = llm_result.get("key_items", [])
                sections["management_concerns"] = llm_result.get("concerns", [])
                sections["management_assessment"] = llm_result.get("assessment", "")
        else:
            text_lower = text.lower()
            sections["topic_coverage"] = {
                "career": sum(1 for kw in CAREER_KEYWORDS if kw in text_lower),
                "status": sum(1 for kw in STATUS_KEYWORDS if kw in text_lower),
                "blockers": sum(1 for kw in BLOCKER_KEYWORDS if kw in text_lower),
            }
            sections["burnout_indicators"] = [kw for kw in BURNOUT_KEYWORDS if kw in text_lower]
            sections["alignment_signals"] = [kw for kw in ALIGNMENT_KEYWORDS if kw in text_lower]
            sections["advice_given"] = [kw for kw in ADVICE_KEYWORDS if kw in text_lower]

        return AnalysisResult(call_type=self.meeting_type, sections=sections)

    def _analyze_with_llm(self, text: str) -> dict[str, Any] | None:
        if not self.llm:
            return None
        truncated = text if len(text.split()) <= 4000 else " ".join(text.split()[:4000])
        prompt = self.llm.render_prompt("management", transcript=truncated, management_type=self.meeting_type)
        return self.llm.complete_json(prompt)
