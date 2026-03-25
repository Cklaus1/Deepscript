"""Tests for benchmark history — comparison, trends, and statistics."""

import json
from pathlib import Path
from unittest.mock import patch

from deepscript.benchmark.history import (
    compare_runs,
    format_comparison_markdown,
    format_history_markdown,
    format_trend_markdown,
    list_benchmark_runs,
    model_stats,
    model_trend,
)


def _create_benchmark_file(tmp_path, name, data):
    f = tmp_path / name
    with open(f, "w") as fh:
        json.dump(data, fh)
    return f


def _make_run(timestamp, results):
    return {
        "timestamp": timestamp,
        "models_tested": len(results),
        "results": results,
    }


def _make_model_result(model, quality, latency=500, tier=1, cost=0.01):
    return {
        "model": model,
        "provider": "nim",
        "tier": tier,
        "avg_quality": quality,
        "avg_latency_ms": latency,
        "total_cost_usd": cost,
        "success_rate": 1.0,
        "tasks": [],
    }


def test_list_benchmark_runs(tmp_path):
    _create_benchmark_file(tmp_path, "benchmark-20260325-100000.json", _make_run(
        "2026-03-25T10:00:00Z",
        [_make_model_result("meta/llama-3.1-70b", 8.5)],
    ))
    _create_benchmark_file(tmp_path, "benchmark-20260326-100000.json", _make_run(
        "2026-03-26T10:00:00Z",
        [_make_model_result("meta/llama-3.1-70b", 9.0)],
    ))

    with patch("deepscript.benchmark.history.BENCHMARK_DIR", tmp_path):
        runs = list_benchmark_runs()

    assert len(runs) == 2
    assert runs[0]["timestamp"] == "2026-03-26T10:00:00Z"  # Newest first


def test_compare_runs():
    run_a = _make_run("2026-03-25T10:00:00Z", [
        _make_model_result("model-a", 7.0, latency=800, tier=2),
        _make_model_result("model-b", 5.0, latency=400, tier=3),
    ])
    run_b = _make_run("2026-03-26T10:00:00Z", [
        _make_model_result("model-a", 8.5, latency=600, tier=1),
        _make_model_result("model-c", 6.0, latency=300, tier=2),
    ])

    comparison = compare_runs(run_a, run_b)
    assert comparison["models_compared"] == 1  # model-a in both
    assert comparison["improved"] == 1
    assert "model-b" in comparison["models_removed"]
    assert "model-c" in comparison["models_added"]

    model_a_comp = comparison["comparisons"][0]
    assert model_a_comp["model"] == "model-a"
    assert model_a_comp["quality_delta"] == 1.5
    assert model_a_comp["status"] == "improved"


def test_model_trend(tmp_path):
    for i, (ts, quality) in enumerate([
        ("2026-03-20T10:00:00Z", 6.0),
        ("2026-03-22T10:00:00Z", 7.0),
        ("2026-03-25T10:00:00Z", 8.0),
    ]):
        _create_benchmark_file(tmp_path, f"benchmark-2026032{i}-100000.json", _make_run(
            ts, [_make_model_result("meta/llama-3.1-70b", quality, latency=500 - i * 100)],
        ))

    with patch("deepscript.benchmark.history.BENCHMARK_DIR", tmp_path):
        points = model_trend("meta/llama-3.1-70b")

    assert len(points) == 3
    assert points[0]["quality"] == 6.0
    assert points[2]["quality"] == 8.0


def test_model_stats(tmp_path):
    for i, quality in enumerate([7.0, 8.0, 9.0, 7.5, 8.5]):
        _create_benchmark_file(tmp_path, f"benchmark-2026032{i}-100000.json", _make_run(
            f"2026-03-2{i}T10:00:00Z",
            [_make_model_result("test-model", quality, latency=400 + i * 50)],
        ))

    with patch("deepscript.benchmark.history.BENCHMARK_DIR", tmp_path):
        stats = model_stats("test-model")

    assert stats["runs"] == 5
    assert stats["quality"]["mean"] == 8.0
    assert stats["quality"]["stddev"] > 0  # Should have variance
    assert stats["quality"]["min"] == 7.0
    assert stats["quality"]["max"] == 9.0
    assert stats["quality"]["n"] == 5
    assert stats["latency_ms"]["n"] == 5


def test_model_stats_single_run(tmp_path):
    _create_benchmark_file(tmp_path, "benchmark-20260325-100000.json", _make_run(
        "2026-03-25T10:00:00Z",
        [_make_model_result("test-model", 7.5)],
    ))

    with patch("deepscript.benchmark.history.BENCHMARK_DIR", tmp_path):
        stats = model_stats("test-model")

    assert stats["runs"] == 1
    assert stats["quality"]["stddev"] == 0.0  # No variance with 1 run


def test_model_stats_not_found(tmp_path):
    with patch("deepscript.benchmark.history.BENCHMARK_DIR", tmp_path):
        stats = model_stats("nonexistent")

    assert stats["runs"] == 0


def test_format_history_markdown():
    runs = [
        {"timestamp": "2026-03-25T10:00:00Z", "models_tested": 5, "top_model": "model-a", "top_quality": 8.5},
    ]
    md = format_history_markdown(runs)
    assert "Benchmark History" in md
    assert "model-a" in md


def test_format_history_empty():
    md = format_history_markdown([])
    assert "No benchmark runs found" in md


def test_format_comparison_markdown():
    comparison = {
        "baseline": "2026-03-25T10:00:00Z",
        "current": "2026-03-26T10:00:00Z",
        "models_compared": 1,
        "improved": 1,
        "regressed": 0,
        "stable": 0,
        "models_added": [],
        "models_removed": [],
        "comparisons": [{
            "model": "model-a",
            "quality_before": 7.0,
            "quality_after": 8.5,
            "quality_delta": 1.5,
            "latency_before_ms": 800,
            "latency_after_ms": 600,
            "latency_delta_ms": -200,
            "tier_before": 2,
            "tier_after": 1,
            "tier_change": -1,
            "status": "improved",
        }],
    }
    md = format_comparison_markdown(comparison)
    assert "Benchmark Comparison" in md
    assert "+1.5" in md
    assert "improved" in md
