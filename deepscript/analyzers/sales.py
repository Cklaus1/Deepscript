"""Sales call analyzer — methodology scoring, signals, and phase detection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from deepscript.analyzers.base import AnalysisResult, BaseAnalyzer
from deepscript.analyzers.business import BusinessAnalyzer

if TYPE_CHECKING:
    from deepscript.llm.provider import LLMProvider

METHODOLOGY_DESCRIPTIONS = {
    "meddic": """MEDDIC Framework:
- Metrics: Quantifiable measures of success the buyer cares about
- Economic Buyer: The person with budget authority
- Decision Criteria: What factors will determine the decision
- Decision Process: Steps/timeline to reach a decision
- Identify Pain: The business pain driving the purchase
- Champion: Internal advocate pushing for the solution""",
    "bant": """BANT Framework:
- Budget: Does the prospect have budget allocated?
- Authority: Is this the decision maker?
- Need: Is there a clear, urgent need?
- Timeline: When do they need a solution?""",
    "spin": """SPIN Selling Framework:
- Situation: Questions about current state and context
- Problem: Questions uncovering difficulties and dissatisfactions
- Implication: Questions about consequences of the problems
- Need-Payoff: Questions about value of solving the problem""",
    "challenger": """Challenger Sale Framework:
- Teach: Did the rep share unique insights?
- Tailor: Was the message customized to the prospect?
- Take Control: Did the rep confidently guide the conversation?""",
}


class SalesAnalyzer(BaseAnalyzer):
    """Analyzes sales calls for methodology compliance, signals, and phases."""

    classification_keywords = {
        "sales-call": [
            "pricing", "proposal", "contract", "deal", "close", "discount",
            "competitor", "budget", "decision maker", "timeline", "objection",
            "demo", "pilot", "roi", "implementation",
        ],
    }

    def __init__(
        self,
        llm: Optional["LLMProvider"] = None,
        methodology: str = "meddic",
        competitors: list[str] | None = None,
    ) -> None:
        super().__init__(llm)
        self.methodology = methodology
        self.competitors = competitors or []
        self._business = BusinessAnalyzer(llm)

    @property
    def supported_types(self) -> list[str]:
        return ["sales-call", "sales-discovery", "sales-demo", "sales-negotiation"]

    def _combined_prompt_extras(self, call_type: str) -> tuple[str, str]:
        desc = METHODOLOGY_DESCRIPTIONS.get(self.methodology, "")
        instructions = f"This is a sales call. Score using {self.methodology.upper()}:\n{desc}\nAlso identify buying signals, risk signals, and call phases."
        schema = (
            '"methodology_score": {{"methodology": "...", "scores": {{}}, "total_score": 0, "strengths": [], "gaps": []}},\n'
            '  "buying_signals": [{{"signal": "...", "quote": "...", "strength": "strong|moderate|weak"}}],\n'
            '  "risk_signals": [{{"signal": "...", "quote": "...", "severity": "high|medium|low"}}],\n'
            '  "call_phases": [{{"phase": "Intro|Discovery|Demo|Close", "summary": "..."}}]'
        )
        return instructions, schema

    def analyze(self, transcript: dict[str, Any]) -> AnalysisResult:
        text = transcript.get("text", "")

        # Try single combined LLM call (1 call instead of 4-5)
        if self.llm:
            combined = self.analyze_combined(transcript, "sales-call")
            if combined and combined.sections.get("summary"):
                combined.sections["attendees"] = self._business._extract_attendees(transcript.get("segments", []))
                return combined

        # Multi-call fallback
        base = self._business.analyze(transcript)
        sections = dict(base.sections)

        # Sales-specific sections
        if self.llm:
            methodology_score = self._score_methodology(text)
            if methodology_score:
                sections["methodology_score"] = methodology_score

            signals = self._analyze_signals(text)
            if signals:
                sections["buying_signals"] = signals.get("buying_signals", [])
                sections["risk_signals"] = signals.get("risk_signals", [])
                sections["next_steps"] = signals.get("next_steps", {})
                sections["call_phases"] = signals.get("call_phases", [])
        else:
            # Rule-based signal detection
            sections["buying_signals"] = self._detect_buying_signals_rule_based(text)
            sections["risk_signals"] = self._detect_risk_signals_rule_based(text)
            sections["competitor_mentions"] = self._detect_competitors(text)

        return AnalysisResult(call_type="sales-call", sections=sections)

    def _score_methodology(self, text: str) -> dict[str, Any] | None:
        """Score the call using the configured sales methodology."""
        if not self.llm:
            return None

        description = METHODOLOGY_DESCRIPTIONS.get(self.methodology, "")
        truncated = text if len(text.split()) <= 4000 else " ".join(text.split()[:4000])

        prompt = self.llm.render_prompt(
            "sales_score",
            transcript=truncated,
            methodology=self.methodology.upper(),
            methodology_description=description,
        )
        return self.llm.complete_json(prompt)

    def _analyze_signals(self, text: str) -> dict[str, Any] | None:
        """Analyze buying/risk signals and call phases via LLM."""
        if not self.llm:
            return None

        truncated = text if len(text.split()) <= 4000 else " ".join(text.split()[:4000])
        prompt = self.llm.render_prompt("sales_signals", transcript=truncated)
        return self.llm.complete_json(prompt)

    def _detect_buying_signals_rule_based(self, text: str) -> list[dict[str, Any]]:
        """Simple keyword-based buying signal detection."""
        signals: list[dict[str, Any]] = []
        text_lower = text.lower()

        buying_keywords = [
            ("ownership language", ["when we implement", "our team would", "we could use", "we'd want"]),
            ("implementation interest", ["how would we integrate", "what's the setup", "onboarding process"]),
            ("timeline specificity", ["by q", "by end of", "before next", "this quarter"]),
            ("budget discussion", ["what's the pricing", "what does it cost", "within our budget"]),
        ]

        for signal_type, keywords in buying_keywords:
            for kw in keywords:
                if kw in text_lower:
                    signals.append({"signal": signal_type, "keyword": kw, "strength": "moderate"})
                    break

        return signals

    def _detect_risk_signals_rule_based(self, text: str) -> list[dict[str, Any]]:
        """Simple keyword-based risk signal detection."""
        signals: list[dict[str, Any]] = []
        text_lower = text.lower()

        risk_keywords = [
            ("stall language", ["need to think about", "get back to you", "not sure yet", "let me check"]),
            ("vague next steps", ["i'll follow up", "let's touch base", "we'll be in touch"]),
            ("missing authority", ["need to check with", "have to ask my", "not my decision"]),
            ("price objection", ["too expensive", "over budget", "can't afford", "cheaper alternative"]),
        ]

        for signal_type, keywords in risk_keywords:
            for kw in keywords:
                if kw in text_lower:
                    signals.append({"signal": signal_type, "keyword": kw, "severity": "medium"})
                    break

        return signals

    def _detect_competitors(self, text: str) -> list[str]:
        """Detect configured competitor mentions."""
        text_lower = text.lower()
        return [c for c in self.competitors if c.lower() in text_lower]
