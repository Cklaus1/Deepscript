"""Tests for support analyzer — rule-based (no LLM)."""

from deepscript.analyzers.support import SupportAnalyzer


def _make_transcript(text, segments=None):
    return {"text": text, "segments": segments or []}


def test_support_detects_bug():
    transcript = _make_transcript(
        "The page is broken and keeps crashing when I click submit. "
        "This is the third time this week. It doesn't work at all."
    )
    analyzer = SupportAnalyzer(llm=None)
    result = analyzer.analyze(transcript)

    assert result.sections["issue"]["type"] == "bug"


def test_support_detects_billing():
    transcript = _make_transcript(
        "I was charged twice on my invoice this month. "
        "I need a refund for the duplicate payment."
    )
    analyzer = SupportAnalyzer(llm=None)
    result = analyzer.analyze(transcript)

    assert result.sections["issue"]["type"] == "billing"


def test_support_emotion_trajectory():
    transcript = _make_transcript(
        "This is unacceptable! I'm furious about this outage. "
        "Oh, that actually helps. Thank you so much, that's perfect."
    )
    analyzer = SupportAnalyzer(llm=None)
    result = analyzer.analyze(transcript)

    emotion = result.sections["emotion_trajectory"]
    assert emotion["frustration_signals"] > 0
    assert emotion["satisfaction_signals"] > 0


def test_support_resolution_detected():
    transcript = _make_transcript(
        "Let me try that... yes, it's working now! That fixed it. Thank you."
    )
    analyzer = SupportAnalyzer(llm=None)
    result = analyzer.analyze(transcript)

    assert result.sections["resolution"]["status"] == "resolved"


def test_support_escalation_detected():
    transcript = _make_transcript(
        "I need to escalate this to your manager. Can I speak with a supervisor?"
    )
    analyzer = SupportAnalyzer(llm=None)
    result = analyzer.analyze(transcript)

    assert result.sections["resolution"]["status"] == "escalated"


def test_support_empathy_score():
    transcript = _make_transcript(
        "I understand your frustration. I'm sorry about the inconvenience. "
        "Let me help you resolve this right away. I appreciate your patience."
    )
    analyzer = SupportAnalyzer(llm=None)
    result = analyzer.analyze(transcript)

    assert result.sections["empathy_score"]["score"] > 0
