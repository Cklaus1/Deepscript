"""Topic segmentation — break transcripts into named sections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from deepscript.llm.provider import LLMProvider

TRANSITION_PHRASES = [
    "moving on", "next topic", "let's talk about", "now regarding",
    "turning to", "let's move on", "another thing", "next up",
    "let's discuss", "shifting to", "on the topic of",
]


@dataclass
class Topic:
    """A detected topic segment."""

    name: str
    start_seconds: float
    end_seconds: float
    speakers: list[str]
    summary: str


def segment_topics(
    transcript: dict[str, Any],
    llm: Optional["LLMProvider"] = None,
    min_duration: int = 60,
    max_topics: int = 20,
    method: str = "hybrid",
) -> list[Topic]:
    """Segment transcript into topics.

    Args:
        transcript: Transcript with segments.
        llm: Optional LLM provider for naming/refinement.
        min_duration: Minimum topic duration in seconds.
        max_topics: Maximum number of topics to return.
        method: "rule" (boundaries only), "llm" (full LLM), "hybrid" (rules + LLM naming).
    """
    segments = transcript.get("segments", [])
    if not segments:
        return []

    if method == "llm" and llm:
        result = _segment_with_llm(transcript, llm, max_topics)
        if result:
            return result

    if method == "hybrid" and llm:
        boundaries = _detect_boundaries(segments, min_duration)
        return _name_topics_with_llm(transcript, boundaries, llm, max_topics, min_duration)

    # Rule-based fallback
    return _segment_rule_based(segments, min_duration, max_topics)


def _detect_boundaries(
    segments: list[dict[str, Any]], min_duration: int
) -> list[tuple[int, int]]:
    """Detect topic boundaries via pauses, speaker changes, and transition phrases.

    Returns list of (start_segment_idx, end_segment_idx) tuples.
    """
    if not segments:
        return []

    boundaries: list[int] = [0]  # Always start with segment 0

    for i in range(1, len(segments)):
        prev = segments[i - 1]
        curr = segments[i]

        # Check for long pause (>5 seconds gap)
        gap = curr.get("start", 0) - prev.get("end", 0)
        is_pause = gap > 5.0

        # Check for transition phrase
        text = curr.get("text", "").lower()
        is_transition = any(tp in text for tp in TRANSITION_PHRASES)

        if is_pause or is_transition:
            # Check minimum duration since last boundary
            last_boundary_time = segments[boundaries[-1]].get("start", 0)
            current_time = curr.get("start", 0)
            if current_time - last_boundary_time >= min_duration:
                boundaries.append(i)

    # Convert to (start, end) pairs
    pairs: list[tuple[int, int]] = []
    for i in range(len(boundaries)):
        start = boundaries[i]
        end = boundaries[i + 1] - 1 if i + 1 < len(boundaries) else len(segments) - 1
        pairs.append((start, end))

    return pairs


def _segment_rule_based(
    segments: list[dict[str, Any]], min_duration: int, max_topics: int
) -> list[Topic]:
    """Pure rule-based segmentation with generic topic names."""
    pairs = _detect_boundaries(segments, min_duration)

    topics: list[Topic] = []
    for idx, (start_idx, end_idx) in enumerate(pairs[:max_topics]):
        seg_slice = segments[start_idx : end_idx + 1]
        speakers = sorted({s.get("speaker", "Unknown") for s in seg_slice})
        start_time = seg_slice[0].get("start", 0.0)
        end_time = seg_slice[-1].get("end", 0.0)

        # Generate a basic name from first segment text
        first_text = seg_slice[0].get("text", "").strip()
        name = first_text[:50].rstrip(".") if first_text else f"Topic {idx + 1}"

        # Summary from first sentence
        full_text = " ".join(s.get("text", "") for s in seg_slice)
        sentences = full_text.split(".")
        summary = (sentences[0].strip() + ".") if sentences else ""

        topics.append(Topic(
            name=name,
            start_seconds=start_time,
            end_seconds=end_time,
            speakers=speakers,
            summary=summary,
        ))

    return topics


def _segment_with_llm(
    transcript: dict[str, Any],
    llm: "LLMProvider",
    max_topics: int,
) -> list[Topic] | None:
    """Full LLM-based topic segmentation."""
    text = _format_transcript_with_timestamps(transcript)
    prompt = llm.render_prompt("topics", transcript=text, max_topics=str(max_topics))
    result = llm.complete_json(prompt)

    if not result or not isinstance(result, list):
        return None

    return [
        Topic(
            name=t.get("name", "Unknown"),
            start_seconds=float(t.get("start_seconds", 0)),
            end_seconds=float(t.get("end_seconds", 0)),
            speakers=t.get("speakers", []),
            summary=t.get("summary", ""),
        )
        for t in result
    ]


def _name_topics_with_llm(
    transcript: dict[str, Any],
    boundaries: list[tuple[int, int]],
    llm: "LLMProvider",
    max_topics: int,
    min_duration: int = 60,
) -> list[Topic]:
    """Use rule-based boundaries but LLM for naming."""
    segments = transcript.get("segments", [])
    if not boundaries:
        return []

    text = _format_transcript_with_timestamps(transcript)
    prompt = llm.render_prompt("topics", transcript=text, max_topics=str(max_topics))
    llm_topics = llm.complete_json(prompt)

    if llm_topics and isinstance(llm_topics, list):
        return [
            Topic(
                name=t.get("name", "Unknown"),
                start_seconds=float(t.get("start_seconds", 0)),
                end_seconds=float(t.get("end_seconds", 0)),
                speakers=t.get("speakers", []),
                summary=t.get("summary", ""),
            )
            for t in llm_topics[:max_topics]
        ]

    # LLM failed, fall back to rule-based
    return _segment_rule_based(segments, min_duration, max_topics)


def _format_transcript_with_timestamps(transcript: dict[str, Any]) -> str:
    """Format transcript segments with timestamps for LLM context."""
    segments = transcript.get("segments", [])
    lines: list[str] = []
    for seg in segments:
        start = seg.get("start", 0)
        speaker = seg.get("speaker", "Unknown")
        text = seg.get("text", "").strip()
        mins = int(start // 60)
        secs = int(start % 60)
        lines.append(f"[{mins:02d}:{secs:02d}] {speaker}: {text}")

    result = "\n".join(lines)
    # Truncate if too long
    if len(result) > 15000:
        result = result[:15000] + "\n[...truncated...]"
    return result
