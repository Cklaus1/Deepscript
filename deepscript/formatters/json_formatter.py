"""JSON output formatter for DeepScript analysis results."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from deepscript.analyzers.base import AnalysisResult
from deepscript.core.classifier import Classification
from deepscript.core.communication import CommunicationMetrics
from deepscript.core.topic_segmenter import Topic


def format_json(
    classification: Classification,
    communication: CommunicationMetrics | None,
    analysis: AnalysisResult | None,
    topics: list[Topic] | None = None,
    source_file: str | None = None,
    sections_filter: str | list[str] = "all",
) -> dict[str, Any]:
    """Build structured JSON output from analysis components.

    Args:
        sections_filter: "all" or list of section names to include
            (e.g., ["summary", "action_items", "communication"]).
    """
    result: dict[str, Any] = {}

    if source_file:
        result["source"] = source_file

    result["classification"] = {
        "call_type": classification.call_type,
        "confidence": classification.confidence,
        "scores": classification.scores,
    }
    if classification.reasoning:
        result["classification"]["reasoning"] = classification.reasoning

    # Apply sections filter
    include_all = sections_filter == "all"
    allowed = set(sections_filter) if isinstance(sections_filter, list) else set()

    if communication and (include_all or "communication" in allowed):
        result["communication"] = {
            "total_speakers": communication.total_speakers,
            "total_words": communication.total_words,
            "total_segments": communication.total_segments,
            "total_questions": communication.total_questions,
            "speaking_balance": communication.speaking_balance,
            "speaker_switches_per_segment": communication.speaker_switches_per_segment,
            "speakers": [asdict(s) for s in communication.speakers],
        }

    if topics and (include_all or "topics" in allowed):
        result["topics"] = [asdict(t) for t in topics]

    if analysis:
        if include_all:
            result["analysis"] = analysis.to_dict()
        else:
            # Filter to only requested sections
            filtered = {"call_type": analysis.call_type}
            for key, value in analysis.sections.items():
                if key in allowed:
                    filtered[key] = value
            if len(filtered) > 1:  # More than just call_type
                result["analysis"] = filtered

    return result
