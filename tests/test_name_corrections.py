"""Tests for name correction system."""

from deepscript.core.name_corrections import (
    apply_corrections,
    apply_corrections_to_segments,
    compile_corrections,
    load_corrections,
)


def test_compile_empty_corrections():
    """Empty corrections returns None pattern."""
    pattern, lookup = compile_corrections({})
    assert pattern is None
    assert lookup == {}


def test_compile_corrections_sorts_by_length():
    """Longer patterns match first."""
    corrections = {"a news": "Anuj", "a nooj": "Anuj"}
    pattern, lookup = compile_corrections(corrections)
    assert pattern is not None
    assert lookup["a news"] == "Anuj"
    assert lookup["a nooj"] == "Anuj"


def test_apply_corrections_basic():
    """Replace mangled names in text."""
    corrections = {"a news": "Anuj", "a nooj": "Anuj"}
    result = apply_corrections("Hello a news, how are you?", corrections)
    assert "Anuj" in result
    assert "a news" not in result


def test_apply_corrections_case_insensitive():
    """Corrections work regardless of case."""
    corrections = {"a news": "Anuj"}
    result = apply_corrections("Hello A NEWS, how are you?", corrections)
    assert "Anuj" in result


def test_apply_corrections_no_match():
    """Text without corrections is unchanged."""
    corrections = {"a news": "Anuj"}
    result = apply_corrections("Hello world", corrections)
    assert result == "Hello world"


def test_apply_corrections_with_compiled():
    """Pre-compiled pattern is used correctly."""
    corrections = {"a news": "Anuj"}
    compiled = compile_corrections(corrections)
    result = apply_corrections("Hello a news", corrections, compiled=compiled)
    assert "Anuj" in result


def test_apply_corrections_empty_text():
    """Empty text returns unchanged."""
    corrections = {"a news": "Anuj"}
    assert apply_corrections("", corrections) == ""


def test_apply_corrections_to_segments():
    """Corrections applied in-place to segments."""
    corrections = {"a news": "Anuj"}
    segments = [
        {"text": "Hello a news"},
        {"text": "No change here"},
        {"text": "a news is here"},
    ]
    count = apply_corrections_to_segments(segments, corrections)
    assert count == 2
    assert segments[0]["text"] == "Hello Anuj"
    assert segments[1]["text"] == "No change here"
    assert segments[2]["text"] == "Anuj is here"


def test_apply_corrections_to_segments_empty():
    """No corrections returns zero count."""
    segments = [{"text": "hello"}]
    count = apply_corrections_to_segments(segments, {})
    assert count == 0


def test_load_corrections_from_file(tmp_path):
    """Load corrections from a name_corrections.txt file."""
    corrections_file = tmp_path / "name_corrections.txt"
    corrections_file.write_text("a news -> Anuj\na nooj -> Anuj\n# comment\n")
    result = load_corrections(transcript_dir=tmp_path)
    assert result["a news"] == "Anuj"
    assert result["a nooj"] == "Anuj"


def test_load_corrections_from_file_with_arrow(tmp_path):
    """Load corrections using => separator."""
    corrections_file = tmp_path / "name_corrections.txt"
    corrections_file.write_text("a news => Anuj\n")
    result = load_corrections(transcript_dir=tmp_path)
    assert result["a news"] == "Anuj"


def test_load_corrections_none_sources():
    """No sources returns empty dict."""
    result = load_corrections()
    assert result == {}


def test_load_corrections_from_contacts():
    """Build corrections from contact list with non-English names."""
    contacts = [
        {"displayName": "Anuj Grover"},
        {"displayName": "John Smith"},  # English name, should be skipped
    ]
    result = load_corrections(contacts=contacts)
    # Should have corrections for Anuj variants
    assert any("Anuj" in v for v in result.values())
    # John Smith should not generate corrections (no non-English patterns)
    john_corrections = {k: v for k, v in result.items() if "john" in k}
    assert len(john_corrections) == 0