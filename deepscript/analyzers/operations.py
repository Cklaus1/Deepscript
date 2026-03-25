"""Operations meeting analyzers — standup, all-hands, retro, postmortem."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from deepscript.analyzers.base import AnalysisResult, BaseAnalyzer
from deepscript.analyzers.business import BusinessAnalyzer

if TYPE_CHECKING:
    from deepscript.llm.provider import LLMProvider

BLOCKER_KEYWORDS = ["blocked", "stuck", "waiting", "dependency", "can't", "blocker", "impediment"]
RETRO_WELL_KEYWORDS = ["went well", "liked", "great", "good job", "proud", "success"]
RETRO_IMPROVE_KEYWORDS = ["improve", "didn't go well", "frustrating", "should have", "next time"]
ROOT_CAUSE_KEYWORDS = ["root cause", "because", "caused by", "due to", "contributing factor", "triggered by"]
BLAME_KEYWORDS = ["fault", "blame", "who did", "responsible for breaking", "should have known"]


class OperationsAnalyzer(BaseAnalyzer):
    """Analyzes operations meetings (standups, all-hands, retros, postmortems)."""

    def __init__(self, llm: Optional["LLMProvider"] = None, ops_type: str = "standup") -> None:
        super().__init__(llm)
        self.ops_type = ops_type
        self._business = BusinessAnalyzer(llm)

    @property
    def supported_types(self) -> list[str]:
        return ["standup", "all-hands", "sprint-retro", "postmortem"]

    def analyze(self, transcript: dict[str, Any]) -> AnalysisResult:
        text = transcript.get("text", "")
        segments = transcript.get("segments", [])
        base = self._business.analyze(transcript)
        sections = dict(base.sections)

        if self.llm:
            llm_result = self._analyze_with_llm(text)
            if llm_result:
                sections["blockers"] = llm_result.get("blockers", [])
                sections["themes"] = llm_result.get("themes", [])
                sections["announcements"] = llm_result.get("announcements", [])
                sections["unanswered_questions"] = llm_result.get("unanswered_questions", [])
                sections["ops_assessment"] = llm_result.get("assessment", {})
        else:
            text_lower = text.lower()
            sections["blockers_detected"] = [kw for kw in BLOCKER_KEYWORDS if kw in text_lower]

            if self.ops_type == "sprint-retro":
                sections["went_well"] = [kw for kw in RETRO_WELL_KEYWORDS if kw in text_lower]
                sections["to_improve"] = [kw for kw in RETRO_IMPROVE_KEYWORDS if kw in text_lower]
            elif self.ops_type == "postmortem":
                sections["root_cause_signals"] = [kw for kw in ROOT_CAUSE_KEYWORDS if kw in text_lower]
                sections["blame_signals"] = [kw for kw in BLAME_KEYWORDS if kw in text_lower]
                sections["blamelessness"] = "concerning" if sections["blame_signals"] else "healthy"

            # Time discipline for standups
            if self.ops_type == "standup" and segments:
                duration = segments[-1].get("end", 0) - segments[0].get("start", 0)
                sections["duration_seconds"] = round(duration, 1)
                sections["time_discipline"] = "good" if duration <= 900 else "over_time"

        return AnalysisResult(call_type=self.ops_type, sections=sections)

    def _analyze_with_llm(self, text: str) -> dict[str, Any] | None:
        if not self.llm:
            return None
        truncated = text if len(text.split()) <= 4000 else " ".join(text.split()[:4000])
        prompt = self.llm.render_prompt("operations", transcript=truncated, ops_type=self.ops_type)
        return self.llm.complete_json(prompt)
