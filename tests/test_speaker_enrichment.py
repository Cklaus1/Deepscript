"""Tests for speaker enrichment — mapping diarization to segments."""

from deepscript.core.speaker_enrichment import (
    enrich_speakers,
    get_speaker_context,
    load_speaker_db,
)


def test_enrich_speakers_already_labeled():
    """No-op when segments already have speaker labels."""
    transcript = {
        "text": "hello",
        "segments": [
            {"start": 0, "end": 5, "text": "hello", "speaker": "Alice"},
        ],
    }
    enrich_speakers(transcript)
    assert transcript["segments"][0]["speaker"] == "Alice"


def test_enrich_speakers_from_diarization():
    """Map speakers from diarization metadata to segments."""
    transcript = {
        "text": "hello world",
        "segments": [
            {"start": 0, "end": 5, "text": "hello"},
            {"start": 6, "end": 10, "text": "world"},
            {"start": 15, "end": 20, "text": "goodbye"},
        ],
        "diarization": {
            "speakers_resolved": [
                {
                    "local_label": "SPEAKER_00",
                    "speaker_cluster_id": "spk_abc123",
                    "display_name": "Alice",
                    "status": "confirmed",
                    "confidence": 0.95,
                },
                {
                    "local_label": "SPEAKER_01",
                    "speaker_cluster_id": "spk_def456",
                    "display_name": None,
                    "status": "unknown",
                    "confidence": 1.0,
                },
            ],
        },
    }
    enrich_speakers(transcript)

    # Should have speaker labels now (heuristic assignment)
    speakers = {s.get("speaker") for s in transcript["segments"]}
    assert "Unknown" not in speakers or len(speakers) > 1  # At least some labeled
    # All segments should have cluster_ids
    for seg in transcript["segments"]:
        assert seg.get("speaker") is not None


def test_enrich_speakers_no_diarization():
    """Leave segments unchanged when no diarization data."""
    transcript = {
        "text": "hello",
        "segments": [{"start": 0, "end": 5, "text": "hello"}],
    }
    enrich_speakers(transcript)
    assert transcript["segments"][0].get("speaker") is None


def test_enrich_speakers_empty_segments():
    """Handle empty segments list."""
    transcript = {"text": "hello", "segments": []}
    enrich_speakers(transcript)
    assert transcript["segments"] == []


def test_enrich_speakers_uses_cluster_id_when_no_name():
    """Use cluster ID as speaker name when display_name is null."""
    transcript = {
        "text": "hello",
        "segments": [{"start": 0, "end": 5, "text": "hello"}],
        "diarization": {
            "speakers_resolved": [
                {
                    "local_label": "SPEAKER_00",
                    "speaker_cluster_id": "spk_abc123",
                    "display_name": None,
                    "status": "unknown",
                    "confidence": 1.0,
                },
            ],
        },
    }
    enrich_speakers(transcript)
    assert transcript["segments"][0]["speaker"] == "spk_abc123"


def test_get_speaker_context():
    """Build speaker context from diarization metadata."""
    transcript = {
        "diarization": {
            "speakers_resolved": [
                {
                    "local_label": "SPEAKER_00",
                    "speaker_cluster_id": "spk_abc",
                    "display_name": "Alice",
                    "status": "confirmed",
                    "confidence": 0.95,
                    "is_new": False,
                },
            ],
        },
    }
    ctx = get_speaker_context(transcript)
    assert ctx["total_speakers"] == 1
    assert ctx["speakers"][0]["cluster_id"] == "spk_abc"
    assert ctx["speakers"][0]["display_name"] == "Alice"


def test_get_speaker_context_with_db():
    """Enrich with speaker DB data."""
    transcript = {
        "diarization": {
            "speakers_resolved": [
                {
                    "local_label": "SPEAKER_00",
                    "speaker_cluster_id": "spk_abc",
                    "display_name": None,
                    "status": "unknown",
                    "confidence": 1.0,
                    "is_new": False,
                },
            ],
        },
    }
    mock_db = {
        "identities": {
            "spk_abc": {
                "canonical_name": "Bob Smith",
                "total_calls": 5,
                "total_speaking_seconds": 3600,
                "first_seen": "2026-01-01",
                "last_seen": "2026-03-25",
                "typical_co_speakers": ["spk_def"],
            },
        },
    }
    ctx = get_speaker_context(transcript, speaker_db=mock_db)
    assert ctx["is_returning"] is True
    assert ctx["speakers"][0]["canonical_name"] == "Bob Smith"
    assert ctx["speakers"][0]["total_calls"] == 5


def test_real_transcript_enrichment():
    """Test with actual AudioScript transcript."""
    import json
    from pathlib import Path

    fixture = Path(__file__).parent / "fixtures" / "real_product_discussion.json"
    if not fixture.exists():
        return  # Skip if fixture not available

    with open(fixture) as f:
        transcript = json.load(f)

    # Before enrichment
    speakers_before = {s.get("speaker") for s in transcript["segments"][:10]}

    enrich_speakers(transcript)

    # After enrichment — should have speaker labels
    speakers_after = {s.get("speaker") for s in transcript["segments"][:10]}
    assert None not in speakers_after  # All segments should be labeled
    assert len(speakers_after) >= 2  # Should detect multiple speakers
