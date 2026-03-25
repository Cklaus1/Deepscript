"""QBR (Quarterly Business Review) analyzer — health score, expansion, churn risk."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from deepscript.analyzers.base import AnalysisResult, BaseAnalyzer
from deepscript.analyzers.business import BusinessAnalyzer

if TYPE_CHECKING:
    from deepscript.llm.provider import LLMProvider

EXPANSION_KEYWORDS = [
    "more seats", "additional users", "another team", "expand", "upgrade",
    "enterprise plan", "premium", "add-on", "grow", "scale",
]

CHURN_KEYWORDS = [
    "cancel", "downgrade", "not renewing", "looking at alternatives",
    "budget cuts", "considering other", "not sure we'll continue",
    "disappointed", "not getting value",
]

HEALTH_POSITIVE = [
    "love", "great", "value", "roi", "successful", "adoption",
    "team loves", "can't imagine without", "essential",
]

HEALTH_NEGATIVE = [
    "frustrated", "issue", "problem", "complaint", "bug",
    "slow", "unreliable", "missing", "confusing",
]


class QBRAnalyzer(BaseAnalyzer):
    """Analyzes QBR calls for customer health, expansion, and churn risk."""

    def __init__(self, llm: Optional["LLMProvider"] = None) -> None:
        super().__init__(llm)
        self._business = BusinessAnalyzer(llm)

    @property
    def supported_types(self) -> list[str]:
        return ["qbr"]

    def analyze(self, transcript: dict[str, Any]) -> AnalysisResult:
        text = transcript.get("text", "")
        segments = transcript.get("segments", [])

        base = self._business.analyze(transcript)
        sections = dict(base.sections)

        if self.llm:
            llm_result = self._analyze_with_llm(text)
            if llm_result:
                sections["health_score"] = llm_result.get("health_score", {})
                sections["expansion_signals"] = llm_result.get("expansion_signals", [])
                sections["churn_risk"] = llm_result.get("churn_risk", {})
                sections["value_realization"] = llm_result.get("value_realization", {})
                sections["renewal_outlook"] = llm_result.get("renewal_outlook", {})
        else:
            sections.update(self._analyze_rule_based(text, segments))

        return AnalysisResult(call_type="qbr", sections=sections)

    def _analyze_with_llm(self, text: str) -> dict[str, Any] | None:
        if not self.llm:
            return None
        truncated = text if len(text.split()) <= 4000 else " ".join(text.split()[:4000])
        prompt = self.llm.render_prompt("qbr", transcript=truncated)
        return self.llm.complete_json(prompt)

    def _analyze_rule_based(
        self, text: str, segments: list[dict[str, Any]]
    ) -> dict[str, Any]:
        text_lower = text.lower()

        # Expansion signals
        expansion = [kw for kw in EXPANSION_KEYWORDS if kw in text_lower]

        # Churn risk
        churn = [kw for kw in CHURN_KEYWORDS if kw in text_lower]

        # Health score
        positive = sum(1 for kw in HEALTH_POSITIVE if kw in text_lower)
        negative = sum(1 for kw in HEALTH_NEGATIVE if kw in text_lower)
        health_raw = positive - negative
        health_score = max(0, min(10, 5 + health_raw))

        # Churn risk level
        if len(churn) >= 3:
            churn_level = "high"
        elif len(churn) >= 1:
            churn_level = "medium"
        else:
            churn_level = "low"

        return {
            "health_score": {
                "score": health_score,
                "max": 10,
                "positive_signals": positive,
                "negative_signals": negative,
            },
            "expansion_signals": [
                {"signal": kw, "strength": "moderate"} for kw in expansion
            ],
            "churn_risk": {
                "level": churn_level,
                "indicators": [
                    {"indicator": kw, "severity": "medium"} for kw in churn
                ],
            },
            "method": "rule-based",
        }
