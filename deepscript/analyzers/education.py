"""Education analyzers — classroom, lecture, coaching."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from deepscript.analyzers.base import AnalysisResult, BaseAnalyzer
from deepscript.analyzers.business import BusinessAnalyzer

if TYPE_CHECKING:
    from deepscript.llm.provider import LLMProvider

COMPREHENSION_KEYWORDS = ["understand", "makes sense", "got it", "clear", "i see"]
CONFUSION_KEYWORDS = ["confused", "don't get", "can you repeat", "what do you mean", "lost me"]
GROW_GOAL_KEYWORDS = ["goal", "objective", "want to achieve", "aim", "target"]
GROW_REALITY_KEYWORDS = ["current situation", "right now", "at the moment", "currently"]
GROW_OPTIONS_KEYWORDS = ["could", "option", "alternative", "what if", "possibility"]
GROW_WILL_KEYWORDS = ["commit", "will do", "by next", "action", "plan to"]


class EducationAnalyzer(BaseAnalyzer):
    """Analyzes educational sessions (classroom, lecture, coaching)."""

    def __init__(self, llm: Optional["LLMProvider"] = None, edu_type: str = "classroom") -> None:
        super().__init__(llm)
        self.edu_type = edu_type
        self._business = BusinessAnalyzer(llm)

    @property
    def supported_types(self) -> list[str]:
        return ["classroom", "lecture", "coaching-session"]

    def analyze(self, transcript: dict[str, Any]) -> AnalysisResult:
        text = transcript.get("text", "")
        segments = transcript.get("segments", [])
        base = self._business.analyze(transcript)
        sections = dict(base.sections)

        if self.llm:
            llm_result = self._analyze_with_llm(text)
            if llm_result:
                sections["concepts"] = llm_result.get("concepts", [])
                sections["edu_engagement"] = llm_result.get("engagement", {})
                sections["pacing"] = llm_result.get("pacing", "")
                sections["key_takeaways"] = llm_result.get("key_takeaways", [])
                sections["terminology"] = llm_result.get("terminology", [])
                sections["edu_assessment"] = llm_result.get("assessment", "")
        else:
            text_lower = text.lower()
            sections["comprehension_signals"] = sum(1 for kw in COMPREHENSION_KEYWORDS if kw in text_lower)
            sections["confusion_signals"] = sum(1 for kw in CONFUSION_KEYWORDS if kw in text_lower)

            if self.edu_type == "coaching-session":
                sections["grow_model"] = {
                    "goal": sum(1 for kw in GROW_GOAL_KEYWORDS if kw in text_lower),
                    "reality": sum(1 for kw in GROW_REALITY_KEYWORDS if kw in text_lower),
                    "options": sum(1 for kw in GROW_OPTIONS_KEYWORDS if kw in text_lower),
                    "will": sum(1 for kw in GROW_WILL_KEYWORDS if kw in text_lower),
                }

        return AnalysisResult(call_type=self.edu_type, sections=sections)

    def _analyze_with_llm(self, text: str) -> dict[str, Any] | None:
        if not self.llm:
            return None
        truncated = text if len(text.split()) <= 4000 else " ".join(text.split()[:4000])
        prompt = self.llm.render_prompt("education", transcript=truncated, edu_type=self.edu_type)
        return self.llm.complete_json(prompt)
