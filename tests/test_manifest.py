"""Tests for processing manifest."""

import json
import tempfile
from pathlib import Path

from deepscript.utils.manifest import ProcessingManifest, get_file_hash


def test_manifest_save_and_load(tmp_path):
    manifest = ProcessingManifest()
    fixture = Path(__file__).parent / "fixtures" / "sample_transcript.json"
    manifest.record(fixture, "completed", call_type="business-meeting")

    manifest_path = tmp_path / ".deepscript-manifest.json"
    manifest.save(manifest_path)

    loaded = ProcessingManifest.load(manifest_path)
    assert str(fixture) in loaded.entries
    assert loaded.entries[str(fixture)].status == "completed"


def test_manifest_is_processed(tmp_path):
    manifest = ProcessingManifest()
    fixture = Path(__file__).parent / "fixtures" / "sample_transcript.json"
    manifest.record(fixture, "completed")

    assert manifest.is_processed(fixture)


def test_manifest_not_processed_if_hash_changed(tmp_path):
    manifest = ProcessingManifest()
    # Record with a fake file
    test_file = tmp_path / "test.json"
    test_file.write_text('{"text": "hello"}')
    manifest.record(test_file, "completed")

    # Change file content
    test_file.write_text('{"text": "changed"}')
    assert not manifest.is_processed(test_file)


def test_manifest_load_missing_file(tmp_path):
    manifest = ProcessingManifest.load(tmp_path / "nonexistent.json")
    assert len(manifest.entries) == 0


def test_file_hash_deterministic():
    fixture = Path(__file__).parent / "fixtures" / "sample_transcript.json"
    h1 = get_file_hash(fixture)
    h2 = get_file_hash(fixture)
    assert h1 == h2
    assert len(h1) == 16
