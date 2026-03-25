"""Discovery call analyzer — Mom Test, JTBD, pain points, commitment signals."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from deepscript.analyzers.base import AnalysisResult, BaseAnalyzer
from deepscript.analyzers.business import BusinessAnalyzer

if TYPE_CHECKING:
    from deepscript.llm.provider import LLMProvider

FRAMEWORK_DESCRIPTIONS = {
    "mom_test": """The Mom Test (Rob Fitzpatrick):
Rules for good customer conversations:
1. Talk about their life, not your idea
2. Ask about specifics in the past, not generics or hypotheticals
3. Talk less, listen more
4. Never ask "would you buy this?" — ask about current behavior
5. Look for real commitment (time, reputation, money), not compliments

Bad questions: "Do you think it's a good idea?", "Would you use this?", "How much would you pay?"
Good questions: "What do you currently do?", "Walk me through the last time...", "What have you tried?"

Compliment traps: Enthusiasm without commitment is false validation.""",
    "jtbd": """Jobs-to-be-Done Framework:
Focus on understanding the "job" the customer is trying to accomplish.
Extract in format: "When [situation], I want [motivation], so I can [outcome]"
Look for: switching triggers, hiring/firing criteria, struggling moments.""",
    "problem_solution": """Problem-Solution Interview:
1. Identify the problem clearly
2. Understand current solutions/workarounds
3. Gauge severity and frequency
4. Explore willingness to change""",
}


class DiscoveryAnalyzer(BaseAnalyzer):
    """Analyzes discovery calls for pain points, validation quality, and insights."""

    def __init__(
        self,
        llm: Optional["LLMProvider"] = None,
        framework: str = "mom_test",
    ) -> None:
        super().__init__(llm)
        self.framework = framework
        self._business = BusinessAnalyzer(llm)

    @property
    def supported_types(self) -> list[str]:
        return ["discovery-call", "customer-discovery"]

    def analyze(self, transcript: dict[str, Any]) -> AnalysisResult:
        text = transcript.get("text", "")
        segments = transcript.get("segments", [])

        # Get base business meeting analysis
        base = self._business.analyze(transcript)
        sections = dict(base.sections)

        if self.llm:
            discovery_result = self._analyze_with_llm(text)
            if discovery_result:
                sections["framework_score"] = {
                    "framework": discovery_result.get("framework", self.framework),
                    "score": discovery_result.get("framework_score", 0),
                    "notes": discovery_result.get("framework_notes", ""),
                }
                sections["pain_points"] = discovery_result.get("pain_points", [])
                sections["jtbd"] = discovery_result.get("jtbd", [])
                sections["commitment_signals"] = discovery_result.get("commitment_signals", [])
                sections["compliment_traps"] = discovery_result.get("compliment_traps", [])
                sections["call_phases"] = discovery_result.get("call_phases", [])
                sections["hidden_opportunities"] = discovery_result.get("hidden_opportunities", [])
        else:
            # Rule-based fallback
            sections["pain_points"] = self._detect_pain_points_rule_based(text, segments)
            sections["commitment_signals"] = self._detect_commitments_rule_based(text)
            sections["hypothetical_questions"] = self._detect_hypotheticals(text)

        return AnalysisResult(call_type="discovery-call", sections=sections)

    def _analyze_with_llm(self, text: str) -> dict[str, Any] | None:
        """Full LLM-based discovery analysis."""
        if not self.llm:
            return None

        description = FRAMEWORK_DESCRIPTIONS.get(self.framework, "")
        truncated = text if len(text.split()) <= 4000 else " ".join(text.split()[:4000])

        prompt = self.llm.render_prompt(
            "discovery_score",
            transcript=truncated,
            framework=self.framework.replace("_", " ").title(),
            framework_description=description,
        )
        return self.llm.complete_json(prompt)

    def _detect_pain_points_rule_based(
        self, text: str, segments: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Detect pain points via keyword patterns."""
        pain_keywords = [
            "frustrating", "painful", "annoying", "waste of time", "takes too long",
            "biggest challenge", "struggle with", "problem is", "issue is",
            "can't figure out", "doesn't work", "breaks when", "hate",
        ]
        text_lower = text.lower()
        pains: list[dict[str, Any]] = []
        seen: set[str] = set()

        for seg in segments:
            seg_text = seg.get("text", "").strip()
            seg_lower = seg_text.lower()
            for kw in pain_keywords:
                if kw in seg_lower:
                    normalized = seg_text[:50].lower()
                    if normalized not in seen:
                        seen.add(normalized)
                        pains.append({
                            "pain": seg_text,
                            "keyword": kw,
                            "speaker": seg.get("speaker", "Unknown"),
                            "timestamp": seg.get("start"),
                        })
                    break

        return pains

    def _detect_commitments_rule_based(self, text: str) -> list[dict[str, Any]]:
        """Detect commitment signals."""
        signals: list[dict[str, Any]] = []
        text_lower = text.lower()

        commitment_keywords = [
            ("time", ["let's schedule", "happy to do a follow-up", "set up a demo", "meet again"]),
            ("reputation", ["introduce you to", "mention it to", "bring in my", "connect you with", "email an intro", "intro to"]),
            ("money", ["what's the pricing", "send me a proposal", "budget for this", "willing to pay"]),
        ]

        for ctype, keywords in commitment_keywords:
            for kw in keywords:
                if kw in text_lower:
                    signals.append({"type": ctype, "signal": kw, "strength": "moderate"})
                    break

        return signals

    def _detect_hypotheticals(self, text: str) -> list[str]:
        """Detect hypothetical questions (Mom Test violations)."""
        hypotheticals: list[str] = []
        hypothetical_patterns = [
            "would you", "do you think", "how much would you pay",
            "would you use", "would you buy", "would it be useful",
        ]
        text_lower = text.lower()
        for pattern in hypothetical_patterns:
            if pattern in text_lower:
                hypotheticals.append(pattern)
        return hypotheticals
