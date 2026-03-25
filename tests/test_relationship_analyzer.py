"""Tests for relationship analyzer — rule-based (no LLM)."""

from deepscript.analyzers.relationship import RelationshipAnalyzer


def _make_transcript(segments):
    text = " ".join(s["text"] for s in segments)
    return {"text": text, "segments": segments}


def test_relationship_analyzer_basic():
    segments = [
        {"start": 0, "end": 5, "text": "I had such a weird day today.", "speaker": "Partner A"},
        {"start": 5, "end": 12, "text": "Oh really? Tell me about it, I want to hear.", "speaker": "Partner B"},
        {"start": 12, "end": 20, "text": "Well, my boss said something that frustrated me.", "speaker": "Partner A"},
        {"start": 20, "end": 26, "text": "That sounds tough. I understand how you feel.", "speaker": "Partner B"},
        {"start": 26, "end": 32, "text": "Thanks, I appreciate you listening.", "speaker": "Partner A"},
    ]
    transcript = _make_transcript(segments)

    analyzer = RelationshipAnalyzer(llm=None)
    result = analyzer.analyze(transcript)

    assert result.call_type == "relationship"
    assert "listening_balance" in result.sections
    assert "we_i_language" in result.sections
    assert "validation_moments" in result.sections
    assert "engagement_signals" in result.sections


def test_validation_detection():
    segments = [
        {"start": 0, "end": 5, "text": "I understand where you're coming from.", "speaker": "A"},
        {"start": 5, "end": 10, "text": "That makes sense to me.", "speaker": "B"},
        {"start": 10, "end": 15, "text": "You're right about that.", "speaker": "A"},
    ]
    transcript = _make_transcript(segments)

    analyzer = RelationshipAnalyzer(llm=None)
    result = analyzer.analyze(transcript)

    validations = result.sections["validation_moments"]
    assert len(validations) >= 2


def test_appreciation_ratio():
    segments = [
        {"start": 0, "end": 5, "text": "I love spending time with you.", "speaker": "A"},
        {"start": 5, "end": 10, "text": "I appreciate everything you do.", "speaker": "B"},
        {"start": 10, "end": 15, "text": "Thank you for being so wonderful.", "speaker": "A"},
        {"start": 15, "end": 20, "text": "That was a bit annoying though.", "speaker": "B"},
    ]
    transcript = _make_transcript(segments)

    analyzer = RelationshipAnalyzer(llm=None)
    result = analyzer.analyze(transcript)

    ratio = result.sections["appreciation_ratio"]
    assert ratio["positive_count"] > ratio["negative_count"]
    assert "healthy" in ratio["assessment"].lower() or "approaching" in ratio["assessment"].lower()


def test_we_i_language():
    segments = [
        {"start": 0, "end": 5, "text": "We should plan our vacation together.", "speaker": "A"},
        {"start": 5, "end": 10, "text": "I think we could go somewhere warm.", "speaker": "B"},
        {"start": 10, "end": 15, "text": "I want to go to the beach.", "speaker": "A"},
    ]
    transcript = _make_transcript(segments)

    analyzer = RelationshipAnalyzer(llm=None)
    result = analyzer.analyze(transcript)

    we_i = result.sections["we_i_language"]
    assert "A" in we_i
    assert "B" in we_i
    assert we_i["B"]["we_count"] > 0


def test_gottman_horsemen_detection():
    segments = [
        {"start": 0, "end": 5, "text": "You always forget to do the dishes.", "speaker": "A"},
        {"start": 5, "end": 10, "text": "That's not true, you never notice when I do.", "speaker": "B"},
    ]
    transcript = _make_transcript(segments)

    analyzer = RelationshipAnalyzer(llm=None)
    result = analyzer.analyze(transcript)

    gottman = result.sections["gottman_indicators"]
    assert len(gottman) >= 1
    assert any(g["type"] == "criticism" for g in gottman)


def test_engagement_signals():
    segments = [
        {"start": 0, "end": 5, "text": "How was school today?", "speaker": "Parent"},
        {"start": 5, "end": 7, "text": "Fine.", "speaker": "Kid"},
        {"start": 7, "end": 12, "text": "Did anything interesting happen?", "speaker": "Parent"},
        {"start": 12, "end": 14, "text": "Not really.", "speaker": "Kid"},
        {"start": 14, "end": 20, "text": "Tell me about your coding project.", "speaker": "Parent"},
        {"start": 20, "end": 30, "text": "Oh yeah! So I built this really cool thing where you can make a character move around the screen and it was so fun!", "speaker": "Kid"},
    ]
    transcript = _make_transcript(segments)

    analyzer = RelationshipAnalyzer(llm=None)
    result = analyzer.analyze(transcript)

    engagement = result.sections["engagement_signals"]
    assert engagement["Kid"]["one_word_answers"] >= 1
    assert engagement["Kid"]["total_turns"] == 3
