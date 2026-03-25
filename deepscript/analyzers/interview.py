"""Interview analyzer — STAR scoring, competency mapping, evidence strength."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Optional

from deepscript.analyzers.base import AnalysisResult, BaseAnalyzer
from deepscript.analyzers.business import BusinessAnalyzer

if TYPE_CHECKING:
    from deepscript.llm.provider import LLMProvider

# STAR component indicators
STAR_SITUATION = ["when i was", "at my previous", "in my last role", "there was a time", "the context was"]
STAR_TASK = ["my responsibility", "i was tasked", "i needed to", "the goal was", "i was supposed to"]
STAR_ACTION = ["i decided to", "i implemented", "i built", "i led", "i created", "what i did was", "i approached"]
STAR_RESULT = ["the result was", "we achieved", "this led to", "the outcome", "we improved", "increased by", "decreased by", "saved"]


class InterviewAnalyzer(BaseAnalyzer):
    """Analyzes interview transcripts for STAR completeness and competencies."""

    def __init__(
        self,
        llm: Optional["LLMProvider"] = None,
        interview_type: str = "behavioral",
    ) -> None:
        super().__init__(llm)
        self.interview_type = interview_type
        self._business = BusinessAnalyzer(llm)

    @property
    def supported_types(self) -> list[str]:
        return ["interview-behavioral", "interview-technical"]

    def analyze(self, transcript: dict[str, Any]) -> AnalysisResult:
        text = transcript.get("text", "")
        segments = transcript.get("segments", [])

        base = self._business.analyze(transcript)
        sections = dict(base.sections)

        if self.llm:
            llm_result = self._analyze_with_llm(text)
            if llm_result:
                sections["interview_answers"] = llm_result.get("answers", [])
                sections["overall_star_score"] = llm_result.get("overall_star_score", 0)
                sections["competencies_demonstrated"] = llm_result.get("competencies_demonstrated", [])
                sections["competency_gaps"] = llm_result.get("competency_gaps", [])
                sections["interview_strengths"] = llm_result.get("strengths", [])
                sections["interview_concerns"] = llm_result.get("concerns", [])
                sections["recommendation"] = llm_result.get("recommendation", "")
                sections["recommendation_reasoning"] = llm_result.get("recommendation_reasoning", "")
        else:
            sections.update(self._analyze_rule_based(text, segments))

        return AnalysisResult(
            call_type=f"interview-{self.interview_type}",
            sections=sections,
        )

    def _analyze_with_llm(self, text: str) -> dict[str, Any] | None:
        if not self.llm:
            return None
        truncated = text if len(text.split()) <= 4000 else " ".join(text.split()[:4000])
        prompt = self.llm.render_prompt(
            "interview", transcript=truncated, interview_type=self.interview_type,
        )
        return self.llm.complete_json(prompt)

    def _analyze_rule_based(
        self, text: str, segments: list[dict[str, Any]]
    ) -> dict[str, Any]:
        text_lower = text.lower()

        # Count STAR components
        situation = sum(1 for kw in STAR_SITUATION if kw in text_lower)
        task = sum(1 for kw in STAR_TASK if kw in text_lower)
        action = sum(1 for kw in STAR_ACTION if kw in text_lower)
        result = sum(1 for kw in STAR_RESULT if kw in text_lower)

        components_present = sum(1 for c in [situation, task, action, result] if c > 0)

        star_scores = {
            "situation": min(situation, 3),
            "task": min(task, 3),
            "action": min(action, 3),
            "result": min(result, 3),
        }

        overall_star = round(components_present / 4 * 10, 1)

        # Detect question count (interviewer questions)
        question_segments = [s for s in segments if s.get("text", "").strip().endswith("?")]

        return {
            "star_analysis": {
                "components": star_scores,
                "components_present": components_present,
                "total_possible": 4,
                "assessment": (
                    "complete" if components_present == 4
                    else "partial" if components_present >= 2
                    else "incomplete"
                ),
            },
            "overall_star_score": overall_star,
            "questions_asked": len(question_segments),
            "method": "rule-based",
        }
