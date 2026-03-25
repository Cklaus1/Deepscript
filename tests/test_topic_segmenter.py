"""Tests for topic segmentation."""

import json
from pathlib import Path

from deepscript.core.topic_segmenter import segment_topics, _detect_boundaries

FIXTURES = Path(__file__).parent / "fixtures"


def test_segment_topics_rule_based():
    with open(FIXTURES / "sample_transcript.json") as f:
        transcript = json.load(f)

    topics = segment_topics(transcript, llm=None, min_duration=30, method="rule")
    assert isinstance(topics, list)
    # Should find at least one topic
    assert len(topics) >= 1
    # Each topic should have required fields
    for t in topics:
        assert t.name
        assert t.start_seconds >= 0
        assert t.end_seconds >= t.start_seconds
        assert isinstance(t.speakers, list)


def test_segment_topics_empty_transcript():
    transcript = {"text": "", "segments": []}
    topics = segment_topics(transcript, llm=None, method="rule")
    assert topics == []


def test_detect_boundaries_with_transitions():
    segments = [
        {"start": 0, "end": 30, "text": "Let's discuss Q1 results.", "speaker": "A"},
        {"start": 30, "end": 60, "text": "Revenue was up 15%.", "speaker": "A"},
        {"start": 60, "end": 90, "text": "Good numbers.", "speaker": "B"},
        {"start": 90, "end": 120, "text": "Let's move on to the roadmap.", "speaker": "A"},
        {"start": 120, "end": 150, "text": "We're planning a redesign.", "speaker": "A"},
    ]
    boundaries = _detect_boundaries(segments, min_duration=30)
    assert len(boundaries) >= 1


def test_segment_topics_respects_max():
    with open(FIXTURES / "sample_transcript.json") as f:
        transcript = json.load(f)

    topics = segment_topics(transcript, llm=None, min_duration=10, max_topics=2, method="rule")
    assert len(topics) <= 2
