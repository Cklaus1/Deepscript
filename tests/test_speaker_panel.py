"""Tests for speaker panel (LLM multi-expert identification)."""

from deepscript.core.speaker_panel import (
    PanelResult,
    LabelResult,
    _gather_speech_samples,
    _format_speech_samples,
    _format_candidates,
    _parse_panel_response,
)


def test_panel_result_to_dict():
    """Convert PanelResult to dict."""
    result = PanelResult(
        cluster_id="spk_00",
        name="John Doe",
        confidence=85,
        reasoning="Self-ID in opening",
        key_evidence=["Self-ID in opening"],
    )
    d = result.to_dict()
    assert d["cluster_id"] == "spk_00"
    assert d["name"] == "John Doe"
    assert d["confidence"] == 85


def test_label_result_to_dict():
    """Convert LabelResult to dict."""
    result = LabelResult(
        cluster_id="spk_00",
        label="Estate Attorney",
        category="advisor",
        reasoning="Legal vocabulary detected",
    )
    d = result.to_dict()
    assert d["cluster_id"] == "spk_00"
    assert d["label"] == "Estate Attorney"
    assert d["category"] == "advisor"


def test_gather_speech_samples():
    """Gather speech samples for a speaker."""
    transcripts = [
        (
            type("P", (), {"stem": "test"})(),
            {
                "diarization": {
                    "speakers_resolved": [
                        {"speaker_cluster_id": "spk_00"},
                        {"speaker_cluster_id": "spk_01"},
                    ],
                },
                "segments": [
                    {"speaker_cluster_id": "spk_00", "text": "Hello, I'm John Doe and I work at Acme Corp", "start": 0, "end": 5},
                    {"speaker_cluster_id": "spk_01", "text": "Hi there", "start": 5, "end": 10},
                    {"speaker_cluster_id": "spk_00", "text": "How are you today? I wanted to discuss the project timeline", "start": 10, "end": 15},
                ],
            },
        ),
    ]
    samples = _gather_speech_samples("spk_00", transcripts)
    assert len(samples) == 1
    assert samples[0]["call"] == "test"
    assert len(samples[0]["segments"]) == 2


def test_gather_speech_samples_empty():
    """Return empty list when no segments match."""
    transcripts = [
        (
            type("P", (), {"stem": "test"})(),
            {
                "diarization": {
                    "speakers_resolved": [
                        {"speaker_cluster_id": "spk_00"},
                        {"speaker_cluster_id": "spk_01"},
                    ],
                },
                "segments": [
                    {"speaker_cluster_id": "spk_01", "text": "Hello", "start": 0, "end": 5},
                ],
            },
        ),
    ]
    samples = _gather_speech_samples("spk_00", transcripts)
    assert samples == []


def test_gather_speech_samples_filters_short():
    """Filter out segments shorter than 10 chars."""
    transcripts = [
        (
            type("P", (), {"stem": "test"})(),
            {
                "diarization": {
                    "speakers_resolved": [
                        {"speaker_cluster_id": "spk_00"},
                    ],
                },
                "segments": [
                    {"speaker_cluster_id": "spk_00", "text": "Yeah", "start": 0, "end": 5},
                    {"speaker_cluster_id": "spk_00", "text": "This is a longer segment with enough text to pass the filter", "start": 10, "end": 15},
                ],
            },
        ),
    ]
    samples = _gather_speech_samples("spk_00", transcripts)
    assert len(samples) == 1
    assert len(samples[0]["segments"]) == 1


def test_format_speech_samples():
    """Format speech samples as text."""
    samples = [
        {"call": "Meeting 1", "segments": [{"text": "Hello, I'm John", "start": 0}]},
    ]
    formatted = _format_speech_samples(samples)
    assert "Hello, I'm John" in formatted
    assert "Meeting 1" in formatted


def test_format_candidates():
    """Format candidate list as text."""
    candidates = [
        {"name": "John Doe", "score": 0.85},
        {"name": "Jane Smith", "score": 0.3},
    ]
    formatted = _format_candidates(candidates)
    assert "John Doe" in formatted
    assert "Jane Smith" in formatted


def test_parse_panel_response_valid():
    """Parse valid panel response."""
    response = '{"consensus": {"name": "John Doe", "confidence": 85, "reasoning": "Self-ID", "key_evidence": ["Self-ID in opening"]}, "analyst_1": {"top_pick": "John Doe"}, "analyst_2": {"top_pick": "John Doe"}, "analyst_3": {"top_pick": "John Doe"}}'
    result = _parse_panel_response("spk_00", response)
    assert result is not None
    assert result.name == "John Doe"
    assert result.confidence == 85


def test_parse_panel_response_unknown():
    """Return None when name is UNKNOWN."""
    response = '{"consensus": {"name": "UNKNOWN", "confidence": 0, "reasoning": "", "key_evidence": []}}'
    result = _parse_panel_response("spk_00", response)
    assert result is None


def test_parse_panel_response_invalid():
    """Return None for invalid JSON."""
    result = _parse_panel_response("spk_00", "not json")
    assert result is None