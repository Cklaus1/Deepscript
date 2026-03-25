"""Transcript type classification — rule-based with optional LLM enhancement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from deepscript.llm.provider import LLMProvider

# Keyword clusters for classification.
CLASSIFICATION_KEYWORDS: dict[str, list[str]] = {
    "business-meeting": [
        "agenda", "action item", "follow up", "next steps", "let's discuss",
        "any questions", "moving on", "take away", "meeting", "quarterly",
        "review", "update", "status",
    ],
    "sales-call": [
        "pricing", "proposal", "contract", "deal", "close", "discount",
        "competitor", "budget", "decision maker", "timeline", "objection",
        "demo", "pilot", "roi", "implementation",
    ],
    "discovery-call": [
        "pain point", "challenge", "currently using", "workflow", "frustrat",
        "problem", "how do you", "what happens when", "walk me through",
        "tell me about", "biggest challenge",
    ],
    "standup": [
        "standup", "blocker", "yesterday", "today I", "sprint",
        "pull request", "pr review", "blocked on",
    ],
    "interview-behavioral": [
        "interview", "behavioral", "tell me about a time", "describe a situation",
        "competency", "star", "give me an example", "candidate", "hiring",
    ],
    "interview-technical": [
        "interview", "technical", "coding", "algorithm", "system design",
        "whiteboard", "data structure", "complexity", "candidate",
    ],
    "support-escalation": [
        "support", "ticket", "issue", "escalat", "resolution", "workaround",
        "outage", "incident", "sla", "customer complaint",
    ],
    "qbr": [
        "quarterly business review", "qbr", "renewal", "adoption", "usage",
        "health score", "churn", "expansion", "success metric", "roi",
    ],
    "pmf-call": [
        "product", "feature", "workflow", "use case", "adoption",
        "can't live without", "disappointed", "alternative", "switching",
        "recommend", "love", "critical", "essential",
    ],
    "investor-pitch": [
        "pitch", "investor", "raise", "funding", "valuation", "term sheet",
        "cap table", "runway", "burn rate", "traction", "series",
    ],
    "one-on-one": [
        "one on one", "1:1", "career", "development", "feedback",
        "how are you doing", "blockers", "growth", "promotion",
    ],
    "cofounder-alignment": [
        "cofounder", "co-founder", "vision", "equity split", "roles",
        "alignment", "direction", "company", "strategy",
    ],
    "advisory-call": [
        "advisor", "mentor", "advice", "introduce you to", "in my experience",
        "have you considered", "recommendation", "guidance",
    ],
    "board-meeting": [
        "board", "directors", "governance", "fiduciary", "resolution",
        "quorum", "minutes", "vote", "motion",
    ],
    "customer-onboarding": [
        "onboarding", "setup", "getting started", "welcome", "first steps",
        "training", "walkthrough", "configuration",
    ],
    "sprint-retro": [
        "retro", "retrospective", "went well", "improve", "start doing",
        "stop doing", "keep doing", "action item",
    ],
    "postmortem": [
        "postmortem", "post-mortem", "incident", "root cause", "outage",
        "severity", "timeline", "contributing factor", "prevention",
    ],
    "podcast": [
        "podcast", "episode", "listeners", "audience", "show notes",
        "guest", "host", "subscribe",
    ],
    "classroom": [
        "class", "student", "lesson", "homework", "assignment",
        "curriculum", "exam", "quiz", "lecture",
    ],
    "coaching-session": [
        "coaching", "coach", "goal", "accountability", "growth",
        "self-awareness", "commitment", "breakthrough",
    ],
    "performance-review": [
        "performance", "review", "rating", "objectives", "kpi",
        "improvement plan", "feedback", "evaluation",
    ],
    "churn-save": [
        "cancel", "churn", "leaving", "not renewing", "downgrade",
        "save", "retention", "win back",
    ],
    "vendor-evaluation": [
        "vendor", "evaluation", "rfp", "comparison", "demo",
        "requirement", "pricing", "procurement",
    ],
    "voice-memo": [],
}


@dataclass
class Classification:
    """Result of transcript classification."""

    call_type: str
    confidence: float
    scores: dict[str, float]
    reasoning: str | None = None


def classify_transcript(
    transcript: dict[str, Any],
    custom_classifications: dict[str, Any] | None = None,
    llm: Optional["LLMProvider"] = None,
) -> Classification:
    """Classify a transcript. Uses LLM if available, falls back to keywords."""
    # Try LLM classification first
    if llm is not None:
        llm_result = _classify_with_llm(transcript, llm)
        if llm_result is not None:
            return llm_result

    return _classify_rule_based(transcript, custom_classifications)


def _classify_with_llm(
    transcript: dict[str, Any], llm: "LLMProvider"
) -> Classification | None:
    """Classify using LLM. Returns None on failure."""
    text = transcript.get("text", "")
    # Truncate to ~4000 words to control costs
    words = text.split()
    if len(words) > 4000:
        text = " ".join(words[:4000]) + "\n[...truncated...]"

    prompt = llm.render_prompt("classify", transcript=text)
    result = llm.complete_json(prompt, max_tokens=256)

    if result and "call_type" in result:
        return Classification(
            call_type=result["call_type"],
            confidence=float(result.get("confidence", 0.9)),
            scores={result["call_type"]: float(result.get("confidence", 0.9))},
            reasoning=result.get("reasoning"),
        )
    return None


def _classify_rule_based(
    transcript: dict[str, Any],
    custom_classifications: dict[str, Any] | None = None,
) -> Classification:
    """Classify using keyword matching.

    Merges keywords from: built-in list + auto-discovered analyzers + custom config.
    """
    text = transcript.get("text", "").lower()
    segments = transcript.get("segments", [])

    keywords = dict(CLASSIFICATION_KEYWORDS)

    # Merge auto-discovered keywords from analyzer classes
    try:
        from deepscript.analyzers import collect_keywords
        for ktype, kwords in collect_keywords().items():
            if ktype not in keywords:
                keywords[ktype] = kwords
    except Exception:
        pass  # Auto-discovery is optional

    if custom_classifications:
        for ctype, config in custom_classifications.items():
            kw = config.get("keywords", []) if isinstance(config, dict) else []
            keywords[ctype] = [k.lower() for k in kw]

    scores: dict[str, float] = {}
    for ctype, patterns in keywords.items():
        if not patterns:
            continue
        hits = sum(1 for p in patterns if p in text)
        scores[ctype] = hits / len(patterns) if patterns else 0.0

    # Voice memo heuristic
    speakers = {s.get("speaker") for s in segments if s.get("speaker")}
    word_count = len(text.split())
    best_keyword_score = max(scores.values()) if scores else 0
    if len(speakers) == 1 and word_count < 200 and best_keyword_score < 0.3:
        scores["voice-memo"] = 0.7

    if not scores or max(scores.values()) == 0:
        return Classification(call_type="unknown", confidence=0.0, scores=scores)

    best_type = max(scores, key=lambda k: scores[k])
    best_score = scores[best_type]
    confidence = min(best_score, 1.0)

    return Classification(
        call_type=best_type,
        confidence=round(confidence, 3),
        scores={k: round(v, 3) for k, v in sorted(scores.items(), key=lambda x: -x[1])},
    )
