"""Tests for sales analyzer — rule-based (no LLM)."""

import json
from pathlib import Path

from deepscript.analyzers.sales import SalesAnalyzer

FIXTURES = Path(__file__).parent / "fixtures"


def test_sales_analyzer_from_fixture():
    with open(FIXTURES / "sales_transcript.json") as f:
        transcript = json.load(f)

    analyzer = SalesAnalyzer(llm=None, methodology="meddic", competitors=["Asana", "Monday.com"])
    result = analyzer.analyze(transcript)

    assert result.call_type == "sales-call"
    assert "summary" in result.sections
    assert "action_items" in result.sections
    assert "buying_signals" in result.sections
    assert "risk_signals" in result.sections


def test_sales_detects_buying_signals():
    with open(FIXTURES / "sales_transcript.json") as f:
        transcript = json.load(f)

    analyzer = SalesAnalyzer(llm=None)
    result = analyzer.analyze(transcript)

    buying = result.sections.get("buying_signals", [])
    signal_types = [s["signal"] for s in buying]
    # Should detect implementation interest ("when we implement")
    assert any("implementation" in s or "ownership" in s for s in signal_types)


def test_sales_detects_risk_signals():
    with open(FIXTURES / "sales_transcript.json") as f:
        transcript = json.load(f)

    analyzer = SalesAnalyzer(llm=None)
    result = analyzer.analyze(transcript)

    risk = result.sections.get("risk_signals", [])
    # Should detect "need to check with" or "get back to you"
    signal_types = [s["signal"] for s in risk]
    assert len(risk) > 0


def test_sales_detects_competitors():
    with open(FIXTURES / "sales_transcript.json") as f:
        transcript = json.load(f)

    analyzer = SalesAnalyzer(llm=None, competitors=["Asana", "Monday.com"])
    result = analyzer.analyze(transcript)

    competitors = result.sections.get("competitor_mentions", [])
    assert "Asana" in competitors
    assert "Monday.com" in competitors


def test_sales_includes_business_sections():
    with open(FIXTURES / "sales_transcript.json") as f:
        transcript = json.load(f)

    analyzer = SalesAnalyzer(llm=None)
    result = analyzer.analyze(transcript)

    # Should include base business analyzer sections
    assert "attendees" in result.sections
    assert "questions" in result.sections
