"""Tests for interview analyzer — rule-based (no LLM)."""

import json
from pathlib import Path

from deepscript.analyzers.interview import InterviewAnalyzer

FIXTURES = Path(__file__).parent / "fixtures"


def test_interview_analyzer_from_fixture():
    with open(FIXTURES / "interview_transcript.json") as f:
        transcript = json.load(f)

    analyzer = InterviewAnalyzer(llm=None, interview_type="behavioral")
    result = analyzer.analyze(transcript)

    assert result.call_type == "interview-behavioral"
    assert "star_analysis" in result.sections
    assert "overall_star_score" in result.sections


def test_star_components_detected():
    with open(FIXTURES / "interview_transcript.json") as f:
        transcript = json.load(f)

    analyzer = InterviewAnalyzer(llm=None)
    result = analyzer.analyze(transcript)

    star = result.sections["star_analysis"]
    components = star["components"]

    # The transcript has explicit STAR structure
    assert components["situation"] > 0  # "at my previous company"
    assert components["action"] > 0  # "what I did was", "I set up"
    assert components["result"] > 0  # "the result was"


def test_star_score_range():
    with open(FIXTURES / "interview_transcript.json") as f:
        transcript = json.load(f)

    analyzer = InterviewAnalyzer(llm=None)
    result = analyzer.analyze(transcript)

    score = result.sections["overall_star_score"]
    assert 0 <= score <= 10


def test_interview_includes_business_sections():
    with open(FIXTURES / "interview_transcript.json") as f:
        transcript = json.load(f)

    analyzer = InterviewAnalyzer(llm=None)
    result = analyzer.analyze(transcript)

    assert "attendees" in result.sections
    assert "questions" in result.sections


def test_technical_interview_type():
    transcript = {
        "text": "Let's start with a coding question. Can you implement a binary search?",
        "segments": [
            {"start": 0, "end": 10, "text": "Let's start with a coding question.", "speaker": "Interviewer"},
        ],
    }
    analyzer = InterviewAnalyzer(llm=None, interview_type="technical")
    result = analyzer.analyze(transcript)

    assert result.call_type == "interview-technical"
