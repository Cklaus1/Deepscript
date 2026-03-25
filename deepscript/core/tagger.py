"""Tag and property generation for MiNotes/Obsidian frontmatter."""

from __future__ import annotations

from typing import Any

from deepscript.core.classifier import Classification
from deepscript.core.communication import CommunicationMetrics
from deepscript.core.topic_segmenter import Topic


def generate_tags(
    classification: Classification,
    communication: CommunicationMetrics | None = None,
    topics: list[Topic] | None = None,
    source_file: str | None = None,
) -> dict[str, Any]:
    """Generate tags and properties for frontmatter.

    Returns:
        dict with "tags" (list[str]) and "properties" (dict).
    """
    tags: list[str] = []
    properties: dict[str, Any] = {}

    # Classification tags
    tags.append(f"call/{classification.call_type}")
    properties["call_type"] = classification.call_type
    properties["classification_confidence"] = classification.confidence

    if source_file:
        properties["source"] = source_file

    # Communication properties
    if communication:
        properties["speakers"] = communication.total_speakers
        properties["total_words"] = communication.total_words
        properties["speaking_balance"] = communication.speaking_balance

        # Add speaker names as tags
        for s in communication.speakers:
            tags.append(f"speaker/{s.speaker}")

        # Duration from segments
        properties["questions_asked"] = communication.total_questions

    # Topic tags
    if topics:
        properties["topic_count"] = len(topics)
        for t in topics:
            # Sanitize topic name for tag use
            tag_name = t.name.lower().replace(" ", "-")[:30]
            tags.append(f"topic/{tag_name}")

    return {"tags": sorted(set(tags)), "properties": properties}
