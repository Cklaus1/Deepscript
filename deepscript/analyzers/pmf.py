"""PMF (Product-Market Fit) analyzer — 8-dimension scoring from call transcripts.

This is whitespace. No existing tool measures PMF from transcripts.
Sean Ellis / Rahul Vohra framework on top of conversation analysis.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from deepscript.analyzers.base import AnalysisResult, BaseAnalyzer
from deepscript.analyzers.business import BusinessAnalyzer

if TYPE_CHECKING:
    from deepscript.llm.provider import LLMProvider

# Rule-based signal keyword clusters
PMF_SIGNALS = {
    "dependency": [
        "can't live without", "critical", "essential", "rely on",
        "depend on", "couldn't go back", "need this",
    ],
    "evangelism": [
        "told", "recommended", "showed", "referred", "mentioned it to",
        "our team loves", "everyone uses",
    ],
    "workflow_depth": [
        "every day", "every morning", "every week", "part of our process",
        "built into", "integrated with", "workflow", "daily standup",
    ],
    "switching_cost": [
        "built around", "integrated", "migrated", "customized",
        "trained the team", "can't imagine switching",
    ],
    "wtp": [
        "worth the price", "great value", "roi", "saves us",
        "pay more for", "worth every penny", "budget",
    ],
    "anti_pmf": [
        "nice to have", "also using", "alternative", "spreadsheet",
        "workaround", "not sure we need", "could live without",
        "haven't used it much", "forgot about",
    ],
}

FEATURE_REQUEST_INCREMENTAL = [
    "export", "filter", "sort", "integration", "api", "format",
    "customize", "theme", "shortcut", "bulk",
]

FEATURE_REQUEST_FUNDAMENTAL = [
    "completely redesign", "doesn't work for", "missing core",
    "can't do basic", "need a different", "wrong approach",
]


class PMFAnalyzer(BaseAnalyzer):
    """Analyzes calls for Product-Market Fit signals."""

    def __init__(self, llm: Optional["LLMProvider"] = None) -> None:
        super().__init__(llm)
        self._business = BusinessAnalyzer(llm)

    @property
    def supported_types(self) -> list[str]:
        return ["pmf-call"]

    def analyze(self, transcript: dict[str, Any]) -> AnalysisResult:
        text = transcript.get("text", "")
        segments = transcript.get("segments", [])

        # Base business analysis
        base = self._business.analyze(transcript)
        sections = dict(base.sections)

        if self.llm:
            llm_result = self._analyze_with_llm(text)
            if llm_result:
                sections["pmf_score"] = llm_result.get("pmf_score", 0)
                sections["pmf_dimensions"] = llm_result.get("dimensions", {})
                sections["ellis_classification"] = llm_result.get("ellis_classification", "")
                sections["ellis_reasoning"] = llm_result.get("ellis_reasoning", "")
                sections["strongest_signals"] = llm_result.get("strongest_signals", [])
                sections["anti_pmf_flags"] = llm_result.get("anti_pmf_flags", [])
                sections["key_quotes"] = llm_result.get("key_quotes", [])
        else:
            # Rule-based PMF analysis
            rule_result = self._analyze_rule_based(text, segments)
            sections.update(rule_result)

        return AnalysisResult(call_type="pmf-call", sections=sections)

    def _analyze_with_llm(self, text: str) -> dict[str, Any] | None:
        if not self.llm:
            return None
        truncated = text if len(text.split()) <= 4000 else " ".join(text.split()[:4000])
        prompt = self.llm.render_prompt("pmf_score", transcript=truncated)
        return self.llm.complete_json(prompt)

    def _analyze_rule_based(
        self, text: str, segments: list[dict[str, Any]]
    ) -> dict[str, Any]:
        text_lower = text.lower()

        # Score each signal category
        signal_scores: dict[str, dict[str, Any]] = {}
        for category, keywords in PMF_SIGNALS.items():
            hits = [kw for kw in keywords if kw in text_lower]
            score = min(len(hits) * 2.5, 10.0) if hits else 0
            signal_scores[category] = {
                "score": round(score, 1),
                "hits": hits,
            }

        # Feature request quality
        incremental = sum(1 for kw in FEATURE_REQUEST_INCREMENTAL if kw in text_lower)
        fundamental = sum(1 for kw in FEATURE_REQUEST_FUNDAMENTAL if kw in text_lower)
        if incremental + fundamental > 0:
            feature_quality = round(incremental / (incremental + fundamental) * 10, 1)
        else:
            feature_quality = 5.0  # Neutral if no feature requests

        # Build dimension scores from signals
        dimensions = {
            "emotional_intensity": {
                "score": signal_scores.get("dependency", {}).get("score", 0),
                "evidence": ", ".join(signal_scores.get("dependency", {}).get("hits", [])),
            },
            "workflow_integration": {
                "score": signal_scores.get("workflow_depth", {}).get("score", 0),
                "evidence": ", ".join(signal_scores.get("workflow_depth", {}).get("hits", [])),
            },
            "referral_evangelism": {
                "score": signal_scores.get("evangelism", {}).get("score", 0),
                "evidence": ", ".join(signal_scores.get("evangelism", {}).get("hits", [])),
            },
            "switching_cost": {
                "score": signal_scores.get("switching_cost", {}).get("score", 0),
                "evidence": ", ".join(signal_scores.get("switching_cost", {}).get("hits", [])),
            },
            "feature_request_quality": {
                "score": feature_quality,
                "evidence": f"{incremental} incremental, {fundamental} fundamental requests",
            },
            "willingness_to_pay": {
                "score": signal_scores.get("wtp", {}).get("score", 0),
                "evidence": ", ".join(signal_scores.get("wtp", {}).get("hits", [])),
            },
            "urgency": {"score": 0, "evidence": "requires LLM"},
            "unprompted_praise": {"score": 0, "evidence": "requires LLM"},
        }

        # Composite PMF score
        dim_scores = [d["score"] for d in dimensions.values()]
        pmf_score = round(sum(dim_scores) / len(dim_scores), 1) if dim_scores else 0

        # Ellis classification: "How would you feel if you could no longer use this product?"
        # "very_disappointed" = STRONG PMF (customer depends on it)
        # "not_disappointed" = NO PMF (customer doesn't care)
        anti_pmf_score = signal_scores.get("anti_pmf", {}).get("score", 0)
        if pmf_score >= 7 and anti_pmf_score <= 2:
            ellis = "very_disappointed"
        elif pmf_score >= 4:
            ellis = "somewhat_disappointed"
        else:
            ellis = "not_disappointed"

        # Anti-PMF flags
        anti_flags = signal_scores.get("anti_pmf", {}).get("hits", [])

        return {
            "pmf_score": pmf_score,
            "pmf_dimensions": dimensions,
            "ellis_classification": ellis,
            "ellis_reasoning": f"Composite score {pmf_score}/10, anti-PMF signals: {len(anti_flags)}",
            "strongest_signals": [
                f"{cat}: {', '.join(data['hits'])}"
                for cat, data in signal_scores.items()
                if data["hits"] and cat != "anti_pmf"
            ][:5],
            "anti_pmf_flags": anti_flags,
            "method": "rule-based",
        }
