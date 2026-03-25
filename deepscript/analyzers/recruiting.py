"""Recruiting analyzers — screen, reference check, offer negotiation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from deepscript.analyzers.base import AnalysisResult, BaseAnalyzer
from deepscript.analyzers.business import BusinessAnalyzer

if TYPE_CHECKING:
    from deepscript.llm.provider import LLMProvider

QUALIFICATION_KEYWORDS = [
    "years of experience", "background in", "worked with", "proficient",
    "certification", "degree", "skill", "expertise",
]
RED_FLAG_KEYWORDS = [
    "gap in resume", "short tenure", "fired", "let go", "conflict",
    "not a team player", "concerns about",
]
COMP_KEYWORDS = [
    "salary", "compensation", "equity", "stock", "bonus", "benefits",
    "base pay", "total comp", "offer", "package",
]


class RecruiterScreenAnalyzer(BaseAnalyzer):
    """Analyzes recruiter screening calls."""

    def __init__(self, llm: Optional["LLMProvider"] = None) -> None:
        super().__init__(llm)
        self._business = BusinessAnalyzer(llm)

    @property
    def supported_types(self) -> list[str]:
        return ["recruiter-screen", "reference-check", "offer-negotiation"]

    def analyze(self, transcript: dict[str, Any]) -> AnalysisResult:
        text = transcript.get("text", "")
        base = self._business.analyze(transcript)
        sections = dict(base.sections)

        if self.llm:
            llm_result = self._analyze_with_llm(text)
            if llm_result:
                sections["qualification"] = llm_result.get("qualification", {})
                sections["signals"] = llm_result.get("signals", [])
                sections["recommendation"] = llm_result.get("recommendation", "")
                sections["recommendation_reasoning"] = llm_result.get("recommendation_reasoning", "")
                sections["compensation"] = llm_result.get("compensation", {})
                sections["key_findings"] = llm_result.get("key_findings", [])
        else:
            text_lower = text.lower()
            sections["qualification_signals"] = [kw for kw in QUALIFICATION_KEYWORDS if kw in text_lower]
            sections["red_flags"] = [kw for kw in RED_FLAG_KEYWORDS if kw in text_lower]
            sections["compensation_discussed"] = any(kw in text_lower for kw in COMP_KEYWORDS)

        return AnalysisResult(call_type="recruiter-screen", sections=sections)

    def _analyze_with_llm(self, text: str) -> dict[str, Any] | None:
        if not self.llm:
            return None
        truncated = text if len(text.split()) <= 4000 else " ".join(text.split()[:4000])
        prompt = self.llm.render_prompt("recruiting", transcript=truncated, recruiting_type="recruiter screen")
        return self.llm.complete_json(prompt)
