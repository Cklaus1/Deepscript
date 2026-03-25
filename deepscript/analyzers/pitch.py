"""Investor pitch & update analyzer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from deepscript.analyzers.base import AnalysisResult, BaseAnalyzer
from deepscript.analyzers.business import BusinessAnalyzer

if TYPE_CHECKING:
    from deepscript.llm.provider import LLMProvider

INTEREST_KEYWORDS = [
    "tell me more", "how does", "what's the timeline", "send me",
    "data room", "introduce you to", "next steps", "follow up",
    "let me connect you", "term sheet", "due diligence",
]

CONCERN_KEYWORDS = [
    "worried about", "what if", "risk", "competition", "burn rate",
    "runway", "market size", "defensibility", "unit economics",
]


class PitchAnalyzer(BaseAnalyzer):
    """Analyzes investor pitch calls."""

    def __init__(self, llm: Optional["LLMProvider"] = None) -> None:
        super().__init__(llm)
        self._business = BusinessAnalyzer(llm)

    @property
    def supported_types(self) -> list[str]:
        return ["investor-pitch", "investor-update"]

    def analyze(self, transcript: dict[str, Any]) -> AnalysisResult:
        text = transcript.get("text", "")
        base = self._business.analyze(transcript)
        sections = dict(base.sections)

        if self.llm:
            llm_result = self._analyze_with_llm(text)
            if llm_result:
                sections["interest_signals"] = llm_result.get("interest_signals", [])
                sections["investor_questions"] = llm_result.get("investor_questions", [])
                sections["objection_handling"] = llm_result.get("objection_handling", [])
                sections["next_step"] = llm_result.get("next_step", {})
                sections["pitch_assessment"] = llm_result.get("pitch_assessment", {})
                sections["improvement_areas"] = llm_result.get("improvement_areas", [])
        else:
            text_lower = text.lower()
            sections["interest_signals"] = [
                {"signal": kw, "strength": "moderate"}
                for kw in INTEREST_KEYWORDS if kw in text_lower
            ]
            sections["investor_concerns"] = [
                kw for kw in CONCERN_KEYWORDS if kw in text_lower
            ]

        return AnalysisResult(call_type="investor-pitch", sections=sections)

    def _analyze_with_llm(self, text: str) -> dict[str, Any] | None:
        if not self.llm:
            return None
        truncated = text if len(text.split()) <= 4000 else " ".join(text.split()[:4000])
        prompt = self.llm.render_prompt("pitch", transcript=truncated)
        return self.llm.complete_json(prompt)
