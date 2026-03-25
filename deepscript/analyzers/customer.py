"""Customer success analyzers — onboarding, churn, renewal, vendor evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from deepscript.analyzers.base import AnalysisResult, BaseAnalyzer
from deepscript.analyzers.business import BusinessAnalyzer

if TYPE_CHECKING:
    from deepscript.llm.provider import LLMProvider

CONFUSION_KEYWORDS = ["confused", "don't understand", "how do i", "where is", "can't find", "lost", "unclear"]
CHURN_REASON_KEYWORDS = ["cancel", "leaving", "switching to", "not renewing", "too expensive", "doesn't meet", "competitor"]
MILESTONE_KEYWORDS = ["set up", "configured", "first", "launched", "went live", "onboarded", "trained"]
VENDOR_KEYWORDS = ["requirement", "feature comparison", "pricing", "vendor", "evaluate", "demo", "criteria"]


class CustomerAnalyzer(BaseAnalyzer):
    """Analyzes customer success calls (onboarding, churn, renewal, vendor eval)."""

    def __init__(self, llm: Optional["LLMProvider"] = None, cs_type: str = "customer-onboarding") -> None:
        super().__init__(llm)
        self.cs_type = cs_type
        self._business = BusinessAnalyzer(llm)

    @property
    def supported_types(self) -> list[str]:
        return ["customer-onboarding", "churn-save", "renewal-expansion", "vendor-evaluation"]

    def analyze(self, transcript: dict[str, Any]) -> AnalysisResult:
        text = transcript.get("text", "")
        base = self._business.analyze(transcript)
        sections = dict(base.sections)

        if self.llm:
            llm_result = self._analyze_with_llm(text)
            if llm_result:
                sections["health_indicators"] = llm_result.get("health_indicators", [])
                sections["risk_flags"] = llm_result.get("risk_flags", [])
                sections["opportunities"] = llm_result.get("opportunities", [])
                sections["milestones"] = llm_result.get("milestones", [])
                sections["cs_findings"] = llm_result.get("key_findings", [])
                sections["cs_next_steps"] = llm_result.get("next_steps", [])
                sections["cs_assessment"] = llm_result.get("assessment", "")
        else:
            text_lower = text.lower()
            sections["confusion_points"] = [kw for kw in CONFUSION_KEYWORDS if kw in text_lower]
            sections["churn_signals"] = [kw for kw in CHURN_REASON_KEYWORDS if kw in text_lower]
            sections["milestones_detected"] = [kw for kw in MILESTONE_KEYWORDS if kw in text_lower]
            sections["vendor_criteria"] = [kw for kw in VENDOR_KEYWORDS if kw in text_lower]

        return AnalysisResult(call_type=self.cs_type, sections=sections)

    def _analyze_with_llm(self, text: str) -> dict[str, Any] | None:
        if not self.llm:
            return None
        truncated = text if len(text.split()) <= 4000 else " ".join(text.split()[:4000])
        prompt = self.llm.render_prompt("customer_success", transcript=truncated, cs_type=self.cs_type)
        return self.llm.complete_json(prompt)
