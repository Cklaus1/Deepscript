"""Tests for Tier 3 analyzers — podcast, therapy, simple."""

from deepscript.analyzers.specialized import PodcastAnalyzer, TherapyAnalyzer, SimpleAnalyzer


def _make_transcript(text, segments=None):
    return {"text": text, "segments": segments or []}


def test_podcast_analyzer():
    segments = [
        {"start": 0, "end": 30, "text": "Welcome to the show! Today we have a great guest.", "speaker": "Host"},
        {"start": 30, "end": 60, "text": "Thanks for having me. I'm excited to share my story about building a startup from zero to acquisition.", "speaker": "Guest"},
        {"start": 60, "end": 90, "text": "Tell us about the early days.", "speaker": "Host"},
        {"start": 90, "end": 150, "text": "We started in a garage with just three people and a dream. The first year was brutal but we learned everything about our customers by doing things that don't scale.", "speaker": "Guest"},
    ]
    transcript = _make_transcript("", segments)
    transcript["text"] = " ".join(s["text"] for s in segments)

    analyzer = PodcastAnalyzer(llm=None)
    result = analyzer.analyze(transcript)

    assert result.call_type == "podcast"
    assert len(result.sections.get("key_quotes", [])) > 0
    assert result.sections.get("guest") == "Host"  # Host talks less in this fixture
    assert result.sections.get("host") == "Guest"  # Guest talks more


def test_therapy_analyzer():
    transcript = _make_transcript(
        "I've been feeling anxious about work lately. "
        "I've been trying the breathing exercises you suggested. "
        "The journaling homework has helped me process my emotions. "
        "I'm feeling more hopeful about things improving."
    )
    analyzer = TherapyAnalyzer(llm=None)
    result = analyzer.analyze(transcript)

    assert result.call_type == "therapy-session"
    assert "breathing" in result.sections.get("coping_strategies", [])
    assert "anxious" in result.sections.get("emotions_mentioned", [])
    assert len(result.sections.get("homework_signals", [])) > 0


def test_simple_analyzer():
    transcript = _make_transcript(
        "Note to self: remember to buy groceries and call the dentist.",
        [{"start": 0, "end": 5, "text": "Note to self: remember to buy groceries and call the dentist.", "speaker": "Me"}],
    )
    analyzer = SimpleAnalyzer(llm=None)
    result = analyzer.analyze(transcript)

    # Should produce basic business sections
    assert "summary" in result.sections
    assert "attendees" in result.sections
