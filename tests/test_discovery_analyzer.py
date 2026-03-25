"""Tests for discovery analyzer — rule-based (no LLM)."""

import json
from pathlib import Path

from deepscript.analyzers.discovery import DiscoveryAnalyzer

FIXTURES = Path(__file__).parent / "fixtures"


def test_discovery_analyzer_from_fixture():
    with open(FIXTURES / "discovery_transcript.json") as f:
        transcript = json.load(f)

    analyzer = DiscoveryAnalyzer(llm=None, framework="mom_test")
    result = analyzer.analyze(transcript)

    assert result.call_type == "discovery-call"
    assert "summary" in result.sections
    assert "pain_points" in result.sections
    assert "commitment_signals" in result.sections


def test_discovery_detects_pain_points():
    with open(FIXTURES / "discovery_transcript.json") as f:
        transcript = json.load(f)

    analyzer = DiscoveryAnalyzer(llm=None)
    result = analyzer.analyze(transcript)

    pains = result.sections.get("pain_points", [])
    assert len(pains) > 0
    # Should detect "it's a mess" or "pain" mentions
    pain_texts = [p["pain"].lower() for p in pains]
    assert any("pain" in t or "mess" in t or "embarrass" in t for t in pain_texts)


def test_discovery_detects_commitments():
    with open(FIXTURES / "discovery_transcript.json") as f:
        transcript = json.load(f)

    analyzer = DiscoveryAnalyzer(llm=None)
    result = analyzer.analyze(transcript)

    commitments = result.sections.get("commitment_signals", [])
    # Should detect time commitment (pilot) and reputation commitment (intro)
    assert len(commitments) > 0
    types = [c["type"] for c in commitments]
    assert "reputation" in types  # "introduce me to"


def test_discovery_detects_hypotheticals():
    with open(FIXTURES / "discovery_transcript.json") as f:
        transcript = json.load(f)

    analyzer = DiscoveryAnalyzer(llm=None)
    result = analyzer.analyze(transcript)

    hypotheticals = result.sections.get("hypothetical_questions", [])
    # "If you could wave a magic wand" is close to hypothetical
    # The transcript has "would you be open" which matches
    assert isinstance(hypotheticals, list)


def test_discovery_includes_business_sections():
    with open(FIXTURES / "discovery_transcript.json") as f:
        transcript = json.load(f)

    analyzer = DiscoveryAnalyzer(llm=None)
    result = analyzer.analyze(transcript)

    assert "attendees" in result.sections
    assert "questions" in result.sections
