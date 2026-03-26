"""Chunk-aware transcript handling.

When AudioScript chunks a long recording (>80K chars) for LLM analysis,
the transcript contains pre-analyzed chunk data. DeepScript should use
this data instead of re-analyzing from scratch.

Key contract:
- transcript["llm_analysis"]["chunked"] == True means this is a chunked call
- transcript["llm_analysis"]["chunks"] contains per-chunk: title, summary, topics, actions
- Chunks are topic segments within ONE call, not separate calls
- DeepScript merges chunk intel into its own analysis, supplementing (not replacing) its pipeline
"""

from __future__ import annotations

import logging
from typing import Any

from deepscript.core.topic_segmenter import Topic

logger = logging.getLogger(__name__)


def is_chunked(transcript: dict[str, Any]) -> bool:
    """Check if this transcript was chunked by AudioScript."""
    llm = transcript.get("llm_analysis", {})
    return llm.get("chunked", False) and llm.get("chunk_count", 0) > 1


def extract_chunk_topics(transcript: dict[str, Any]) -> list[Topic]:
    """Extract topic segments from AudioScript's chunk data.

    Each chunk becomes a topic — the chunk title is the topic name,
    the chunk summary is the topic summary.
    """
    llm = transcript.get("llm_analysis", {})
    chunks = llm.get("chunks", [])
    if not chunks:
        return []

    topics: list[Topic] = []
    for chunk in chunks:
        title = chunk.get("title") or f"Part {len(topics) + 1}"
        summary = chunk.get("summary", "")
        start = chunk.get("start_time", 0.0)
        end = chunk.get("end_time", 0.0)
        speakers = chunk.get("speaker_labels", []) or chunk.get("speakers_identified", [])

        topics.append(Topic(
            name=title,
            start_seconds=start,
            end_seconds=end,
            speakers=speakers,
            summary=summary,
        ))

    return topics


def extract_chunk_actions(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract action items from AudioScript's chunk analysis."""
    llm = transcript.get("llm_analysis", {})
    chunks = llm.get("chunks", [])

    actions: list[dict[str, Any]] = []
    seen: set[str] = set()

    for chunk in chunks:
        for item in chunk.get("action_items", []):
            # Deduplicate
            if isinstance(item, dict):
                text = item.get("text", item.get("action", str(item)))
            else:
                text = str(item)

            normalized = text.lower().strip()[:60]
            if normalized not in seen:
                seen.add(normalized)
                actions.append(item if isinstance(item, dict) else {"text": text})

    return actions


def extract_chunk_summary(transcript: dict[str, Any]) -> str:
    """Get the merged summary from AudioScript's chunk analysis."""
    llm = transcript.get("llm_analysis", {})
    return llm.get("summary", "")


def extract_chunk_classification(transcript: dict[str, Any]) -> str | None:
    """Get AudioScript's LLM classification if available."""
    llm = transcript.get("llm_analysis", {})
    classification = llm.get("classification")
    if isinstance(classification, dict):
        return classification.get("type") or classification.get("call_type")
    if isinstance(classification, str):
        return classification
    return None


def get_chunk_metadata(transcript: dict[str, Any]) -> dict[str, Any]:
    """Get chunk metadata for inclusion in DeepScript output."""
    llm = transcript.get("llm_analysis", {})
    if not llm.get("chunked"):
        return {}

    return {
        "chunked": True,
        "chunk_count": llm.get("chunk_count", 0),
        "total_duration": sum(
            c.get("duration_seconds", 0) for c in llm.get("chunks", [])
        ),
        "chunk_titles": [
            c.get("title", f"Part {i+1}")
            for i, c in enumerate(llm.get("chunks", []))
        ],
    }
