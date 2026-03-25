"""CallEpisode model — maps DeepScript analysis to BTask CMS episode format."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from deepscript.analyzers.base import AnalysisResult
from deepscript.core.classifier import Classification
from deepscript.core.communication import CommunicationMetrics


def _gen_id() -> str:
    return f"ep_{uuid.uuid4().hex[:12]}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class CallEpisode:
    """CMS-compatible episode for a call analysis."""

    episode_id: str = field(default_factory=_gen_id)
    timestamp: str = field(default_factory=_now_iso)
    mode: str = "coding"

    # Context
    task_id: str = ""
    task_type: str = ""  # Classification call_type
    session_id: str = ""
    run_id: str = ""
    generation: int = 0

    # Execution
    model: str = ""
    provider: str = ""
    execution_time_ms: int = 0
    source_file: str = ""

    # Outcome
    status: str = "completed"
    scores: dict[str, float] = field(default_factory=dict)
    findings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    confidence: float = 0.0
    category: str = "approach"

    def to_cms_dict(self) -> dict[str, Any]:
        """Convert to CMS Episode JSONL format."""
        return {
            "episode_id": self.episode_id,
            "timestamp": self.timestamp,
            "mode": self.mode,
            "context": {
                "task_id": self.task_id,
                "task_type": self.task_type,
                "session_id": self.session_id,
                "run_id": self.run_id,
                "generation": self.generation,
            },
            "execution": {
                "model": self.model,
                "provider": self.provider,
                "tool_executions": [],
                "files_modified": [self.source_file] if self.source_file else [],
                "token_usage": {"prompt": 0, "completion": 0},
                "execution_time_ms": self.execution_time_ms,
                "error_category": None,
            },
            "outcome": {
                "status": self.status,
                "scores": self.scores,
                "gate_decision": None,
                "delta": 0.0,
                "is_winner": False,
                "findings": self.findings,
                "errors": self.errors,
            },
            "confidence": self.confidence,
            "category": self.category,
        }


def build_episode(
    classification: Classification,
    analysis: AnalysisResult | None,
    communication: CommunicationMetrics | None = None,
    source_file: str = "",
    model: str = "",
    execution_time_ms: int = 0,
) -> CallEpisode:
    """Build a CallEpisode from analysis results."""
    findings: list[str] = []
    scores: dict[str, float] = {}

    # Extract findings from analysis sections
    if analysis:
        sections = analysis.sections

        # Action items count
        actions = sections.get("action_items", [])
        if actions:
            findings.append(f"{len(actions)} action items extracted")

        # Decisions
        decisions = sections.get("decisions", [])
        if decisions:
            findings.append(f"{len(decisions)} key decisions identified")

        # Methodology score
        ms = sections.get("methodology_score")
        if ms and isinstance(ms, dict):
            total = ms.get("total_score", 0)
            max_p = ms.get("max_possible", 1)
            scores["methodology"] = round(total / max_p * 10, 1) if max_p else 0
            findings.append(f"Methodology score: {total}/{max_p}")

        # Buying/risk signals
        buying = sections.get("buying_signals", [])
        risk = sections.get("risk_signals", [])
        if buying:
            findings.append(f"{len(buying)} buying signals detected")
            scores["buying_signals"] = min(len(buying) * 2.5, 10.0)
        if risk:
            findings.append(f"{len(risk)} risk signals detected")
            scores["risk_signals"] = max(10.0 - len(risk) * 2.5, 0.0)

        # Pain points
        pains = sections.get("pain_points", [])
        if pains:
            findings.append(f"{len(pains)} pain points identified")

        # Framework score
        fw = sections.get("framework_score")
        if fw and isinstance(fw, dict):
            scores["framework"] = float(fw.get("score", 0))

        # Commitment signals
        commits = sections.get("commitment_signals", [])
        if commits:
            findings.append(f"{len(commits)} commitment signals detected")

    # Communication scores
    if communication:
        scores["speaking_balance"] = round(communication.speaking_balance * 10, 1)
        findings.append(f"{communication.total_speakers} speakers, balance: {communication.speaking_balance}")

    # Overall composite score
    if scores:
        scores["overall"] = round(sum(scores.values()) / len(scores), 1)

    return CallEpisode(
        task_id=source_file or _gen_id(),
        task_type=classification.call_type,
        model=model,
        provider="anthropic" if model else "",
        execution_time_ms=execution_time_ms,
        source_file=source_file,
        scores=scores,
        findings=findings,
        confidence=classification.confidence,
        category="pattern",
    )
