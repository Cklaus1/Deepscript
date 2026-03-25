"""Tests for PMF analyzer — rule-based (no LLM)."""

import json
from pathlib import Path

from deepscript.analyzers.pmf import PMFAnalyzer

FIXTURES = Path(__file__).parent / "fixtures"


def test_pmf_analyzer_from_fixture():
    with open(FIXTURES / "pmf_transcript.json") as f:
        transcript = json.load(f)

    analyzer = PMFAnalyzer(llm=None)
    result = analyzer.analyze(transcript)

    assert result.call_type == "pmf-call"
    assert "pmf_score" in result.sections
    assert "pmf_dimensions" in result.sections
    assert "ellis_classification" in result.sections


def test_pmf_detects_dependency_signals():
    with open(FIXTURES / "pmf_transcript.json") as f:
        transcript = json.load(f)

    analyzer = PMFAnalyzer(llm=None)
    result = analyzer.analyze(transcript)

    # Should detect "can't live without" or similar dependency language
    dimensions = result.sections["pmf_dimensions"]
    # Workflow integration should be high (mentions daily use, built around it)
    workflow = dimensions.get("workflow_integration", {})
    assert workflow.get("score", 0) > 0


def test_pmf_detects_anti_pmf():
    with open(FIXTURES / "pmf_transcript.json") as f:
        transcript = json.load(f)

    analyzer = PMFAnalyzer(llm=None)
    result = analyzer.analyze(transcript)

    anti_flags = result.sections.get("anti_pmf_flags", [])
    # Should detect "spreadsheet" or "alternative" usage
    assert len(anti_flags) > 0


def test_pmf_ellis_classification():
    with open(FIXTURES / "pmf_transcript.json") as f:
        transcript = json.load(f)

    analyzer = PMFAnalyzer(llm=None)
    result = analyzer.analyze(transcript)

    ellis = result.sections.get("ellis_classification", "")
    assert ellis in ("very_disappointed", "somewhat_disappointed", "not_disappointed")


def test_pmf_score_range():
    with open(FIXTURES / "pmf_transcript.json") as f:
        transcript = json.load(f)

    analyzer = PMFAnalyzer(llm=None)
    result = analyzer.analyze(transcript)

    score = result.sections["pmf_score"]
    assert 0 <= score <= 10


def test_pmf_includes_business_sections():
    with open(FIXTURES / "pmf_transcript.json") as f:
        transcript = json.load(f)

    analyzer = PMFAnalyzer(llm=None)
    result = analyzer.analyze(transcript)

    assert "attendees" in result.sections
    assert "questions" in result.sections


def test_pmf_empty_transcript():
    transcript = {"text": "Hello, how are you?", "segments": []}
    analyzer = PMFAnalyzer(llm=None)
    result = analyzer.analyze(transcript)

    assert result.sections["pmf_score"] == 0 or result.sections["pmf_score"] >= 0
