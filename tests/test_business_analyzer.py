"""Tests for business meeting analyzer."""

import json
from pathlib import Path

from deepscript.analyzers.business import BusinessAnalyzer

FIXTURES = Path(__file__).parent / "fixtures"


def test_business_analyzer_from_fixture():
    with open(FIXTURES / "sample_transcript.json") as f:
        transcript = json.load(f)

    analyzer = BusinessAnalyzer()
    result = analyzer.analyze(transcript)

    assert result.call_type == "business-meeting"
    assert "summary" in result.sections
    assert "action_items" in result.sections
    assert "decisions" in result.sections
    assert "questions" in result.sections
    assert "attendees" in result.sections


def test_action_item_extraction():
    with open(FIXTURES / "sample_transcript.json") as f:
        transcript = json.load(f)

    analyzer = BusinessAnalyzer()
    result = analyzer.analyze(transcript)

    action_items = result.sections["action_items"]
    assert len(action_items) > 0
    # Should find "Sarah will draft the mobile app requirements"
    texts = [a["text"].lower() for a in action_items]
    assert any("sarah" in t or "mobile" in t or "draft" in t for t in texts)


def test_decision_extraction():
    with open(FIXTURES / "sample_transcript.json") as f:
        transcript = json.load(f)

    analyzer = BusinessAnalyzer()
    result = analyzer.analyze(transcript)

    decisions = result.sections["decisions"]
    assert len(decisions) > 0
    # Should find "decided to prioritize the mobile app redesign"
    texts = [d["text"].lower() for d in decisions]
    assert any("prioritize" in t or "mobile" in t for t in texts)


def test_question_extraction():
    with open(FIXTURES / "sample_transcript.json") as f:
        transcript = json.load(f)

    analyzer = BusinessAnalyzer()
    result = analyzer.analyze(transcript)

    questions = result.sections["questions"]
    assert len(questions) > 0
    # Should find question segments ending with "?"
    assert all(q["text"].endswith("?") for q in questions)


def test_attendee_extraction():
    with open(FIXTURES / "sample_transcript.json") as f:
        transcript = json.load(f)

    analyzer = BusinessAnalyzer()
    result = analyzer.analyze(transcript)

    attendees = result.sections["attendees"]
    speakers = {a["speaker"] for a in attendees}
    assert speakers == {"Alice", "Bob", "Sarah"}
    # Talk ratios should sum to ~1.0
    total_ratio = sum(a["talk_ratio"] for a in attendees)
    assert 0.99 < total_ratio < 1.01


def test_summary_has_duration():
    with open(FIXTURES / "sample_transcript.json") as f:
        transcript = json.load(f)

    analyzer = BusinessAnalyzer()
    result = analyzer.analyze(transcript)

    summary = result.sections["summary"]
    assert summary["duration_seconds"] == 94.0
    assert summary["word_count"] > 0


def test_empty_transcript():
    transcript = {"text": "", "segments": []}
    analyzer = BusinessAnalyzer()
    result = analyzer.analyze(transcript)

    assert result.sections["action_items"] == []
    assert result.sections["decisions"] == []
    assert result.sections["questions"] == []
