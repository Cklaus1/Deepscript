"""Tests for transcript classifier."""

import json
from pathlib import Path

from deepscript.core.classifier import classify_transcript


FIXTURES = Path(__file__).parent / "fixtures"


def test_classify_business_meeting():
    with open(FIXTURES / "sample_transcript.json") as f:
        transcript = json.load(f)
    result = classify_transcript(transcript)
    assert result.call_type == "business-meeting"
    assert result.confidence > 0


def test_classify_sales_call():
    transcript = {
        "text": "Let's talk about pricing for the enterprise plan. What's your budget? "
        "We can offer a discount if you sign the contract this quarter. "
        "Who is the decision maker? What's your timeline for implementation? "
        "Our competitor Gong charges more for less. Let me show you a demo.",
        "segments": [
            {"start": 0, "end": 5, "text": "Let's talk about pricing.", "speaker": "Rep"},
            {"start": 5, "end": 10, "text": "What's your budget?", "speaker": "Rep"},
        ],
    }
    result = classify_transcript(transcript)
    assert result.call_type == "sales-call"
    assert result.confidence > 0


def test_classify_voice_memo():
    transcript = {
        "text": "Note to self: pick up groceries and call the dentist.",
        "segments": [
            {"start": 0, "end": 5, "text": "Note to self: pick up groceries and call the dentist.", "speaker": "Me"},
        ],
    }
    result = classify_transcript(transcript)
    assert result.call_type == "voice-memo"


def test_classify_with_custom_keywords():
    transcript = {
        "text": "The board of directors has reached quorum. Let's begin with the governance review. "
        "The fiduciary responsibility resolution is on the agenda.",
        "segments": [],
    }
    custom = {
        "board-meeting": {
            "keywords": ["board", "directors", "governance", "fiduciary", "resolution", "quorum"]
        }
    }
    result = classify_transcript(transcript, custom_classifications=custom)
    assert result.call_type == "board-meeting"


def test_classify_unknown():
    transcript = {"text": "Hello.", "segments": []}
    result = classify_transcript(transcript)
    assert result.call_type == "unknown"
    assert result.confidence == 0.0


def test_classification_scores_all_types():
    transcript = {
        "text": "Let's discuss the agenda and action items. Also review the proposal pricing.",
        "segments": [],
    }
    result = classify_transcript(transcript)
    assert isinstance(result.scores, dict)
    assert len(result.scores) > 0
