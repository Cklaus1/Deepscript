"""Tests for communication metrics."""

import json
from pathlib import Path

from deepscript.core.communication import analyze_communication

FIXTURES = Path(__file__).parent / "fixtures"


def test_communication_metrics_from_fixture():
    with open(FIXTURES / "sample_transcript.json") as f:
        transcript = json.load(f)
    metrics = analyze_communication(transcript)

    assert metrics.total_speakers == 3
    assert metrics.total_words > 0
    assert metrics.total_segments == 21
    assert metrics.total_questions > 0
    assert 0 < metrics.speaking_balance <= 1.0

    # Alice should have the highest talk ratio
    alice = next(s for s in metrics.speakers if s.speaker == "Alice")
    assert alice.talk_ratio > 0.5


def test_communication_empty_segments():
    transcript = {"text": "hello", "segments": []}
    metrics = analyze_communication(transcript)
    assert metrics.total_speakers == 0
    assert metrics.total_words == 0
    assert metrics.speaking_balance == 0.0


def test_communication_single_speaker():
    transcript = {
        "text": "test",
        "segments": [
            {"start": 0, "end": 5, "text": "Hello world.", "speaker": "Alice"},
            {"start": 5, "end": 10, "text": "More words here.", "speaker": "Alice"},
        ],
    }
    metrics = analyze_communication(transcript)
    assert metrics.total_speakers == 1
    assert metrics.speaking_balance == 1.0
    assert metrics.speaker_switches_per_segment == 0.0


def test_communication_question_count():
    transcript = {
        "text": "",
        "segments": [
            {"start": 0, "end": 5, "text": "What time is it?", "speaker": "Alice"},
            {"start": 5, "end": 10, "text": "Three o'clock.", "speaker": "Bob"},
            {"start": 10, "end": 15, "text": "Are you sure?", "speaker": "Alice"},
        ],
    }
    metrics = analyze_communication(transcript)
    assert metrics.total_questions == 2
    alice = next(s for s in metrics.speakers if s.speaker == "Alice")
    assert alice.question_count == 2


def test_communication_speaker_switches():
    transcript = {
        "text": "",
        "segments": [
            {"start": 0, "end": 5, "text": "Hello.", "speaker": "A"},
            {"start": 5, "end": 10, "text": "Hi.", "speaker": "B"},
            {"start": 10, "end": 15, "text": "How are you?", "speaker": "A"},
            {"start": 15, "end": 20, "text": "Good.", "speaker": "B"},
        ],
    }
    metrics = analyze_communication(transcript)
    # 3 switches out of 4 segments
    assert metrics.speaker_switches_per_segment == 0.75
