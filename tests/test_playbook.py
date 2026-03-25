"""Tests for CMS playbook generation and PMF dashboard."""

import json
from pathlib import Path

from deepscript.cms_bridge.playbook import generate_playbook, load_episodes
from deepscript.cms_bridge.dashboard import generate_pmf_dashboard
from deepscript.cms_bridge.working_memory import assemble_working_memory, format_prep_markdown


def _create_episodes(tmp_path, call_type, count=5):
    """Helper to create test episodes."""
    episodes_dir = tmp_path / "episodes" / "coding"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    jsonl = episodes_dir / f"{call_type}.jsonl"
    with open(jsonl, "w") as f:
        for i in range(count):
            ep = {
                "episode_id": f"ep_test_{i}",
                "timestamp": f"2026-03-{20+i}T10:00:00+00:00",
                "context": {"task_type": call_type},
                "outcome": {
                    "status": "completed",
                    "scores": {"overall": 6.0 + i * 0.5, "methodology": 7.0},
                    "findings": [
                        f"{i+1} action items extracted",
                        f"{i} buying signals detected",
                        "Risk: budget not confirmed" if i % 2 else "Strong MEDDIC compliance",
                    ],
                },
            }
            f.write(json.dumps(ep) + "\n")


def test_load_episodes(tmp_path):
    _create_episodes(tmp_path, "sales-call", 3)
    episodes = load_episodes("sales-call", str(tmp_path))
    assert len(episodes) == 3


def test_load_episodes_missing_file(tmp_path):
    episodes = load_episodes("nonexistent", str(tmp_path))
    assert episodes == []


def test_generate_playbook(tmp_path):
    _create_episodes(tmp_path, "sales-call", 5)
    md = generate_playbook("sales-call", str(tmp_path))

    assert "Sales Call Playbook" in md
    assert "5 analyzed calls" in md
    assert "What Works" in md or "What to Watch" in md
    assert "Benchmarks" in md

    # Check file was saved
    playbook_path = tmp_path / "semantic" / "playbooks" / "sales-call.md"
    assert playbook_path.exists()


def test_generate_playbook_empty(tmp_path):
    md = generate_playbook("nonexistent", str(tmp_path))
    assert "No episodes found" in md


def test_pmf_dashboard(tmp_path):
    episodes_dir = tmp_path / "episodes" / "coding"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    jsonl = episodes_dir / "pmf-call.jsonl"
    with open(jsonl, "w") as f:
        for i in range(10):
            ep = {
                "episode_id": f"ep_pmf_{i}",
                "timestamp": f"2026-03-{15+i}T10:00:00+00:00",
                "context": {"task_type": "pmf-call"},
                "outcome": {
                    "status": "completed",
                    "scores": {"overall": 5.0 + i * 0.3},
                    "findings": [
                        "very_disappointed" if i < 3 else "somewhat_disappointed",
                        f"{i} buying signals detected",
                    ],
                },
            }
            f.write(json.dumps(ep) + "\n")

    md = generate_pmf_dashboard(str(tmp_path))
    assert "PMF Dashboard" in md
    assert "10 customer calls" in md
    assert "Ellis Distribution" in md


def test_pmf_dashboard_empty(tmp_path):
    md = generate_pmf_dashboard(str(tmp_path))
    assert "No PMF call episodes found" in md


def test_working_memory(tmp_path):
    _create_episodes(tmp_path, "sales-call", 5)
    # Also generate playbook
    generate_playbook("sales-call", str(tmp_path))

    wm = assemble_working_memory("sales-call", str(tmp_path))
    assert wm["call_type"] == "sales-call"
    assert wm["playbook"] is not None
    assert len(wm["recent_patterns"]) > 0
    assert len(wm["prep_notes"]) > 0


def test_working_memory_empty(tmp_path):
    wm = assemble_working_memory("nonexistent", str(tmp_path))
    assert wm["playbook"] is None
    assert wm["recent_patterns"] == []


def test_format_prep_markdown():
    wm = {
        "call_type": "sales-call",
        "playbook": "# Sales Playbook\nKey patterns...",
        "recent_patterns": [{"findings": ["3 actions"], "overall_score": 7.5}],
        "dead_ends": ["Don't discuss pricing first"],
        "prep_notes": ["3 actions", "Strong MEDDIC compliance"],
    }
    md = format_prep_markdown(wm)
    assert "Call Prep: Sales Call" in md
    assert "Key Patterns" in md
    assert "Dead Ends" in md
    assert "Playbook Reference" in md
