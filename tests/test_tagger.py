"""Tests for tag generation."""

from deepscript.core.classifier import Classification
from deepscript.core.communication import CommunicationMetrics, SpeakerStats
from deepscript.core.tagger import generate_tags
from deepscript.core.topic_segmenter import Topic


def test_tags_from_classification():
    classification = Classification(call_type="sales-call", confidence=0.9, scores={})
    result = generate_tags(classification)
    assert "call/sales-call" in result["tags"]
    assert result["properties"]["call_type"] == "sales-call"


def test_tags_include_speakers():
    classification = Classification(call_type="business-meeting", confidence=0.8, scores={})
    communication = CommunicationMetrics(
        total_speakers=2, total_words=100, total_segments=10, total_questions=3,
        speaking_balance=0.9, speaker_switches_per_segment=0.5,
        speakers=[
            SpeakerStats("Alice", 60, 6, 2, 0.6, 10.0, 30),
            SpeakerStats("Bob", 40, 4, 1, 0.4, 10.0, 20),
        ],
    )
    result = generate_tags(classification, communication=communication)
    assert "speaker/Alice" in result["tags"]
    assert "speaker/Bob" in result["tags"]
    assert result["properties"]["speakers"] == 2


def test_tags_include_topics():
    classification = Classification(call_type="business-meeting", confidence=0.8, scores={})
    topics = [
        Topic("Q1 Review", 0, 60, ["Alice"], "Reviewed Q1 results."),
        Topic("Product Roadmap", 60, 120, ["Bob"], "Discussed roadmap."),
    ]
    result = generate_tags(classification, topics=topics)
    assert "topic/q1-review" in result["tags"]
    assert result["properties"]["topic_count"] == 2
