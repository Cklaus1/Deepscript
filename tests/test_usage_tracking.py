"""Tests for persistent usage tracking."""

import json
from pathlib import Path
from unittest.mock import patch

from deepscript.llm.cost_tracker import (
    CostTracker,
    UsageEntry,
    clear_usage,
    load_usage_history,
    rotate_usage_log,
    usage_summary,
)


def test_cost_tracker_persist(tmp_path):
    usage_file = tmp_path / "usage.jsonl"

    with patch("deepscript.llm.cost_tracker.USAGE_DIR", tmp_path), \
         patch("deepscript.llm.cost_tracker.USAGE_FILE", usage_file):
        tracker = CostTracker(budget_limit=50.0)
        tracker.record("claude-sonnet-4-6", input_tokens=1000, output_tokens=500)
        tracker.record("claude-sonnet-4-6", input_tokens=2000, output_tokens=800)
        tracker.persist(source_file="test.json", call_type="sales-call")

    assert usage_file.exists()
    with open(usage_file) as f:
        lines = f.readlines()
    assert len(lines) == 2

    entry = json.loads(lines[0])
    assert entry["model"] == "claude-sonnet-4-6"
    assert entry["input_tokens"] == 1000
    assert entry["source_file"] == "test.json"
    assert entry["call_type"] == "sales-call"


def test_load_usage_history(tmp_path):
    usage_file = tmp_path / "usage.jsonl"
    entries = [
        {"timestamp": "2026-03-25T10:00:00+00:00", "model": "claude-sonnet-4-6",
         "input_tokens": 1000, "output_tokens": 500, "cost_usd": 0.0105,
         "source_file": "a.json", "call_type": "sales-call"},
        {"timestamp": "2026-03-25T11:00:00+00:00", "model": "claude-sonnet-4-6",
         "input_tokens": 2000, "output_tokens": 800, "cost_usd": 0.018,
         "source_file": "b.json", "call_type": "discovery-call"},
    ]
    with open(usage_file, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    with patch("deepscript.llm.cost_tracker.USAGE_FILE", usage_file):
        result = load_usage_history()
    assert len(result) == 2


def test_load_usage_history_filter_month(tmp_path):
    usage_file = tmp_path / "usage.jsonl"
    entries = [
        {"timestamp": "2026-02-15T10:00:00+00:00", "model": "m", "input_tokens": 100,
         "output_tokens": 50, "cost_usd": 0.001, "source_file": "", "call_type": ""},
        {"timestamp": "2026-03-25T10:00:00+00:00", "model": "m", "input_tokens": 200,
         "output_tokens": 100, "cost_usd": 0.002, "source_file": "", "call_type": ""},
    ]
    with open(usage_file, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    with patch("deepscript.llm.cost_tracker.USAGE_FILE", usage_file):
        result = load_usage_history(month="2026-03")
    assert len(result) == 1
    assert result[0].input_tokens == 200


def test_usage_summary():
    entries = [
        UsageEntry("2026-03-25T10:00:00", "claude-sonnet-4-6", 1000, 500, 0.0105, "a.json", "sales-call"),
        UsageEntry("2026-03-25T11:00:00", "claude-sonnet-4-6", 2000, 800, 0.018, "b.json", "discovery-call"),
    ]
    result = usage_summary(entries, budget_limit=50.0)

    assert result["calls"] == 2
    assert result["total_input_tokens"] == 3000
    assert result["total_output_tokens"] == 1300
    assert result["total_cost_usd"] == 0.0285
    assert result["budget_remaining_usd"] == 50.0 - 0.0285
    assert "claude-sonnet-4-6" in result["models"]
    assert result["models"]["claude-sonnet-4-6"]["calls"] == 2
    assert "sales-call" in result["by_call_type"]
    assert "discovery-call" in result["by_call_type"]


def test_usage_summary_empty():
    result = usage_summary([], budget_limit=50.0)
    assert result["calls"] == 0
    assert result["budget_remaining_usd"] == 50.0


def test_clear_usage(tmp_path):
    usage_file = tmp_path / "usage.jsonl"
    usage_file.write_text('{"line": 1}\n{"line": 2}\n')

    with patch("deepscript.llm.cost_tracker.USAGE_FILE", usage_file):
        count = clear_usage()
    assert count == 2
    assert not usage_file.exists()


def test_clear_usage_no_file(tmp_path):
    usage_file = tmp_path / "nonexistent.jsonl"
    with patch("deepscript.llm.cost_tracker.USAGE_FILE", usage_file):
        count = clear_usage()
    assert count == 0


def _write_entries(path, n, start=0):
    with open(path, "w") as f:
        for i in range(start, start + n):
            f.write(json.dumps({
                "timestamp": "2026-03-25T10:00:00+00:00", "model": "m",
                "input_tokens": 1, "output_tokens": 1, "cost_usd": 0.0,
                "source_file": str(i), "call_type": "",
            }) + "\n")


def test_rotate_usage_log_prunes_to_keep_lines(tmp_path):
    usage_file = tmp_path / "usage.jsonl"
    _write_entries(usage_file, 1000)

    with patch("deepscript.llm.cost_tracker.USAGE_FILE", usage_file):
        rotated = rotate_usage_log(max_bytes=1000, keep_lines=100)

    assert rotated is True
    lines = usage_file.read_text().splitlines()
    assert len(lines) == 100
    # Keeps the *newest* entries (900..999), drops the oldest.
    assert json.loads(lines[0])["source_file"] == "900"
    assert json.loads(lines[-1])["source_file"] == "999"
    # Atomic temp file is cleaned up.
    assert not usage_file.with_suffix(".jsonl.tmp").exists()


def test_rotate_usage_log_noop_under_cap(tmp_path):
    usage_file = tmp_path / "usage.jsonl"
    _write_entries(usage_file, 10)
    before = usage_file.read_text()

    with patch("deepscript.llm.cost_tracker.USAGE_FILE", usage_file):
        rotated = rotate_usage_log(max_bytes=10**9, keep_lines=5)

    assert rotated is False
    assert usage_file.read_text() == before  # untouched


def test_rotate_usage_log_no_file(tmp_path):
    usage_file = tmp_path / "nonexistent.jsonl"
    with patch("deepscript.llm.cost_tracker.USAGE_FILE", usage_file):
        assert rotate_usage_log(max_bytes=1) is False


def test_persist_triggers_rotation(tmp_path):
    usage_file = tmp_path / "usage.jsonl"
    _write_entries(usage_file, 500)  # pre-existing oversized log

    with patch("deepscript.llm.cost_tracker.USAGE_DIR", tmp_path), \
         patch("deepscript.llm.cost_tracker.USAGE_FILE", usage_file), \
         patch("deepscript.llm.cost_tracker.ROTATE_MAX_BYTES", 1000), \
         patch("deepscript.llm.cost_tracker.ROTATE_KEEP_LINES", 50):
        tracker = CostTracker(budget_limit=50.0)
        tracker.record("m", input_tokens=1, output_tokens=1)
        tracker.persist(source_file="new", call_type="y")

    lines = usage_file.read_text().splitlines()
    assert len(lines) == 50  # bounded by keep_lines after append
    # The just-recorded entry survives (it's among the newest).
    assert json.loads(lines[-1])["source_file"] == "new"
