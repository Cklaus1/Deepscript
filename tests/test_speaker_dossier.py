"""Tests for speaker dossier generation."""

from deepscript.core.speaker_dossier import (
    SpeakerDossier,
    _find_best_clip_segment,
    _save_dossier_md,
    _save_summary,
)


def test_speaker_dossier_to_dict():
    """Convert dossier to dict."""
    dossier = SpeakerDossier(
        cluster_id="spk_00",
        label="John Doe",
        facts=["Lives in Florida", "Works at Acme"],
        co_speakers=["Jane Smith"],
        total_calls=5,
    )
    d = dossier.to_dict()
    assert d["cluster_id"] == "spk_00"
    assert d["label"] == "John Doe"
    assert len(d["facts"]) == 2
    assert d["co_speakers"] == ["Jane Smith"]
    assert d["total_calls"] == 5


def test_speaker_dossier_defaults():
    """Dossier with minimal fields."""
    dossier = SpeakerDossier(cluster_id="spk_00", label="Unknown", facts=[])
    d = dossier.to_dict()
    assert d["clip_path"] is None
    assert d["clip_transcript"] == ""
    assert d["category"] == ""


def test_find_best_clip_segment_picks_longest():
    """Prefer longer segments with more words."""
    transcripts = [
        (
            type("P", (), {"stem": "test"})(),
            {
                "segments": [
                    {"speaker_cluster_id": "spk_00", "start": 0, "end": 15, "confidence": 0.9, "text": "This is a long segment with many words to score well"},
                    {"speaker_cluster_id": "spk_00", "start": 20, "end": 22, "confidence": 0.95, "text": "Yeah"},
                    {"speaker_cluster_id": "spk_01", "start": 0, "end": 30, "confidence": 0.9, "text": "Other speaker long segment"},
                ],
            },
        ),
    ]
    best_seg, best_call = _find_best_clip_segment("spk_00", transcripts, clip_duration=10)
    assert best_seg is not None
    assert best_seg["text"].startswith("This is a long")


def test_find_best_clip_segment_filters_short():
    """Skip segments with fewer than 5 words."""
    transcripts = [
        (
            type("P", (), {"stem": "test"})(),
            {
                "segments": [
                    {"speaker_cluster_id": "spk_00", "start": 0, "end": 5, "confidence": 0.9, "text": "Yeah"},
                    {"speaker_cluster_id": "spk_00", "start": 10, "end": 20, "confidence": 0.9, "text": "This is a longer segment with enough words to pass the filter"},
                ],
            },
        ),
    ]
    best_seg, _ = _find_best_clip_segment("spk_00", transcripts, clip_duration=10)
    assert best_seg is not None
    assert "Yeah" not in best_seg["text"]


def test_find_best_clip_segment_no_match():
    """Return None when no segments match."""
    transcripts = [
        (
            type("P", (), {"stem": "test"})(),
            {
                "segments": [
                    {"speaker_cluster_id": "spk_01", "start": 0, "end": 30, "confidence": 0.9, "text": "Other speaker"},
                ],
            },
        ),
    ]
    best_seg, best_call = _find_best_clip_segment("spk_00", transcripts, clip_duration=10)
    assert best_seg is None
    assert best_call is None


def test_save_dossier_md(tmp_path):
    """Save dossier as markdown."""
    dossier = SpeakerDossier(
        cluster_id="spk_00",
        label="John Doe",
        facts=["Lives in Florida"],
        clip_path="/tmp/clip.mp3",
        clip_transcript="Hello world",
        clip_source="test",
        clip_timestamp="01:30",
        co_speakers=["Jane Smith"],
        total_calls=3,
    )
    _save_dossier_md(dossier, tmp_path)
    md_file = tmp_path / "John Doe.md"
    assert md_file.exists()
    content = md_file.read_text()
    assert "John Doe" in content
    assert "spk_00" in content
    assert "Lives in Florida" in content


def test_save_dossier_md_sanitizes_name(tmp_path):
    """Sanitize special characters in name."""
    dossier = SpeakerDossier(
        cluster_id="spk_00",
        label="John <Doe> / O'Brien",
        facts=[],
    )
    _save_dossier_md(dossier, tmp_path)
    # Should create a file with sanitized name
    files = list(tmp_path.glob("*.md"))
    assert len(files) == 1
    # Name should be sanitized (no < > /)
    assert "<" not in files[0].name


def test_save_summary(tmp_path):
    """Save summary index of all dossiers."""
    dossiers = [
        SpeakerDossier(cluster_id="spk_00", label="John", facts=[], total_calls=3, co_speakers=["Jane"]),
        SpeakerDossier(cluster_id="spk_01", label="Jane", facts=[], total_calls=5, co_speakers=["John"]),
    ]
    _save_summary(dossiers, tmp_path)
    index = tmp_path / "_Index.md"
    assert index.exists()
    content = index.read_text()
    assert "John" in content
    assert "Jane" in content
    assert "| 1 |" in content
    assert "| 2 |" in content