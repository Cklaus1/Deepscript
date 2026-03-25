"""Tests for CMS bridge — episode creation and writing."""

import json
from pathlib import Path

from deepscript.analyzers.base import AnalysisResult
from deepscript.cms_bridge.episode import CallEpisode, build_episode
from deepscript.cms_bridge.writer import write_episode
from deepscript.core.classifier import Classification
from deepscript.core.communication import CommunicationMetrics, SpeakerStats


def test_call_episode_to_cms_dict():
    episode = CallEpisode(
        task_id="test_call",
        task_type="sales-call",
        model="claude-sonnet-4-6",
        scores={"overall": 8.5},
        findings=["Test finding"],
    )
    cms = episode.to_cms_dict()

    assert cms["context"]["task_type"] == "sales-call"
    assert cms["outcome"]["scores"]["overall"] == 8.5
    assert "Test finding" in cms["outcome"]["findings"]
    assert cms["episode_id"].startswith("ep_")


def test_build_episode_from_analysis():
    classification = Classification(call_type="sales-call", confidence=0.9, scores={})
    analysis = AnalysisResult(
        call_type="sales-call",
        sections={
            "action_items": [{"text": "Follow up"}],
            "decisions": [{"text": "Go with plan A"}],
            "buying_signals": [{"signal": "timeline"}, {"signal": "budget"}],
        },
    )
    communication = CommunicationMetrics(
        total_speakers=2, total_words=500, total_segments=20,
        total_questions=5, speaking_balance=0.85,
        speaker_switches_per_segment=0.6,
        speakers=[],
    )

    episode = build_episode(classification, analysis, communication, source_file="test.json")
    assert episode.task_type == "sales-call"
    assert len(episode.findings) > 0
    assert "overall" in episode.scores


def test_write_episode_creates_jsonl(tmp_path):
    episode = CallEpisode(
        task_id="test",
        task_type="sales-call",
        findings=["test"],
    )
    result_path = write_episode(episode, str(tmp_path))

    assert result_path.exists()
    assert result_path.name == "sales-call.jsonl"

    with open(result_path) as f:
        line = f.readline()
        data = json.loads(line)
    assert data["context"]["task_type"] == "sales-call"


def test_write_episode_appends(tmp_path):
    for i in range(3):
        episode = CallEpisode(task_id=f"call_{i}", task_type="discovery-call")
        write_episode(episode, str(tmp_path))

    jsonl_path = tmp_path / "episodes" / "coding" / "discovery-call.jsonl"
    with open(jsonl_path) as f:
        lines = f.readlines()
    assert len(lines) == 3
