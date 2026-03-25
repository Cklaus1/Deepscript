"""Tests for benchmark runner and NIM catalog."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from deepscript.benchmark.nim_catalog import (
    NIMModel,
    categorize_models,
    filter_chat_models,
)
from deepscript.benchmark.ground_truth import compute_precision_recall, verify_against_transcript
from deepscript.benchmark.runner import (
    BenchmarkResult,
    ModelBenchmark,
    ScoreResult,
    _score_response,
    format_benchmark_markdown,
    save_benchmark_results,
)


def test_filter_chat_models():
    models = [
        NIMModel(id="meta/llama-3.1-70b-instruct", owned_by="meta", suitable_for_analysis=True),
        NIMModel(id="nvidia/nv-embedqa-e5-v5", owned_by="nvidia", suitable_for_analysis=False),
        NIMModel(id="nvidia/nemoguard-content-safety", owned_by="nvidia", suitable_for_analysis=False),
        NIMModel(id="mistralai/mistral-7b-instruct-v0.3", owned_by="mistralai", suitable_for_analysis=True),
    ]
    filtered = filter_chat_models(models)
    assert len(filtered) == 2
    assert all(m.suitable_for_analysis for m in filtered)
    ids = [m.id for m in filtered]
    assert "meta/llama-3.1-70b-instruct" in ids
    assert "mistralai/mistral-7b-instruct-v0.3" in ids


def test_categorize_models():
    models = [
        NIMModel(id="meta/llama-3.1-70b", owned_by="meta"),
        NIMModel(id="meta/llama-3.1-8b", owned_by="meta"),
        NIMModel(id="nvidia/nemotron-70b", owned_by="nvidia"),
    ]
    by_owner = categorize_models(models)
    assert "meta" in by_owner
    assert len(by_owner["meta"]) == 2
    assert "nvidia" in by_owner


def test_score_response_valid_json():
    task_config = {"expected_fields": ["call_type", "confidence", "reasoning"]}
    response = '{"call_type": "sales-call", "confidence": 0.9, "reasoning": "Contains sales language"}'
    scored = _score_response("classify", task_config, response, fixture_name="sales_transcript")
    assert scored.json_valid is True
    assert scored.schema_complete == 1.0
    assert scored.accuracy_f1 == 1.0  # Correct call_type matches ground truth
    assert scored.quality_score > 5.0


def test_score_response_wrong_classification():
    task_config = {"expected_fields": ["call_type", "confidence", "reasoning"]}
    response = '{"call_type": "voice-memo", "confidence": 0.9, "reasoning": "wrong"}'
    scored = _score_response("classify", task_config, response, fixture_name="sales_transcript")
    assert scored.json_valid is True
    assert scored.accuracy_f1 == 0.0  # Wrong call_type


def test_score_response_partial_schema():
    task_config = {"expected_fields": ["call_type", "confidence", "reasoning"]}
    response = '{"call_type": "sales-call"}'
    scored = _score_response("classify", task_config, response, fixture_name="sales_transcript")
    assert scored.json_valid is True
    assert scored.schema_complete < 1.0


def test_score_response_invalid_json():
    task_config = {"expected_fields": ["call_type"]}
    response = "This is not JSON at all"
    scored = _score_response("classify", task_config, response)
    assert scored.json_valid is False
    assert scored.quality_score <= 2.0


def test_score_response_none():
    task_config = {"expected_fields": ["call_type"]}
    scored = _score_response("classify", task_config, None)
    assert scored.json_valid is False
    assert scored.quality_score == 0.0


def test_score_response_summarize_with_keywords():
    task_config = {"expected_fields": []}
    response = "The quarterly review meeting covered Q1 revenue growth and mobile app roadmap priorities."
    scored = _score_response("summarize", task_config, response, fixture_name="sample_transcript")
    assert scored.json_valid is True
    # Ground truth keywords: quarterly, review, Q1, revenue, mobile, roadmap
    # Response contains 5 of 6 → accuracy_f1 should be >= 0.5
    assert scored.accuracy_f1 >= 0.5, f"Expected >= 0.5, got {scored.accuracy_f1}"
    assert scored.quality_score >= 3.0, f"Expected >= 3.0, got {scored.quality_score}"


def test_score_response_markdown_wrapped():
    task_config = {"expected_fields": ["call_type", "confidence"]}
    response = '```json\n{"call_type": "sales-call", "confidence": 0.9}\n```'
    scored = _score_response("classify", task_config, response, fixture_name="sales_transcript")
    assert scored.json_valid is True


def test_hallucination_detection():
    transcript = "Alice discussed the quarterly review and Bob mentioned the budget."
    task_config = {"expected_fields": ["text"]}
    # Response with a mix of grounded and hallucinated items
    response = '[{"text": "quarterly review"}, {"text": "completely fabricated meeting in Paris"}]'
    scored = _score_response("action_items", task_config, response, transcript_text=transcript)
    assert scored.grounding_rate < 1.0
    assert scored.hallucinated_count >= 1


def test_precision_recall():
    extracted = ["draft mobile app requirements", "set up meeting with design team", "buy groceries"]
    ground_truth = ["draft the mobile app requirements by Friday", "set up a meeting with the design team"]
    pr = compute_precision_recall(extracted, ground_truth)
    assert pr["recall"] >= 0.5  # Should match at least some
    assert pr["precision"] < 1.0  # "buy groceries" is a false positive


def test_verify_against_transcript():
    transcript = "We discussed the quarterly revenue and decided to prioritize the mobile app redesign."
    items = ["discussed the quarterly revenue", "prioritize the mobile app", "trip to Mars next week"]
    result = verify_against_transcript(items, transcript)
    assert result["grounded"] >= 2
    assert result["hallucinated"] >= 1
    assert "trip to Mars next week" in result["hallucinated_items"]


def test_model_benchmark_compute():
    bench = ModelBenchmark(model="test-model", provider="nim")
    bench.results = [
        BenchmarkResult(model="test", provider="nim", task="classify", quality_score=8.0, accuracy_f1=1.0, grounding_rate=1.0, latency_ms=500, cost_usd=0.001),
        BenchmarkResult(model="test", provider="nim", task="summarize", quality_score=7.0, accuracy_f1=0.8, grounding_rate=0.9, latency_ms=300, cost_usd=0.002),
    ]
    bench.compute_aggregates()
    assert bench.avg_quality == 7.5
    assert bench.avg_latency_ms == 400
    assert bench.avg_accuracy_f1 == 0.9
    assert bench.avg_grounding_rate == 0.95
    assert bench.quality_cost_ratio > 0
    assert bench.tier == 1
    assert bench.success_rate == 1.0


def test_model_benchmark_tier_assignment():
    bench = ModelBenchmark(model="test", provider="nim")
    bench.results = [
        BenchmarkResult(model="test", provider="nim", task="t1", quality_score=5.5),
    ]
    bench.compute_aggregates()
    assert bench.tier == 2  # >= 5.0 < 7.5

    bench2 = ModelBenchmark(model="test2", provider="nim")
    bench2.results = [
        BenchmarkResult(model="test2", provider="nim", task="t1", quality_score=3.0),
    ]
    bench2.compute_aggregates()
    assert bench2.tier == 3


def test_model_benchmark_with_errors():
    bench = ModelBenchmark(model="test", provider="nim")
    bench.results = [
        BenchmarkResult(model="test", provider="nim", task="t1", quality_score=8.0),
        BenchmarkResult(model="test", provider="nim", task="t2", error="timeout"),
    ]
    bench.compute_aggregates()
    assert bench.success_rate == 0.5
    assert bench.avg_quality == 8.0  # Only from successful results


def test_save_benchmark_results(tmp_path):
    benchmarks = [
        ModelBenchmark(model="model-a", provider="nim", avg_quality=8.5, tier=1),
        ModelBenchmark(model="model-b", provider="nim", avg_quality=5.0, tier=2),
    ]

    with patch("deepscript.benchmark.runner.BENCHMARK_DIR", tmp_path):
        output_path = save_benchmark_results(benchmarks)

    assert output_path.exists()
    with open(output_path) as f:
        data = json.load(f)
    assert data["models_tested"] == 2
    assert data["results"][0]["model"] == "model-a"


def test_format_benchmark_markdown():
    benchmarks = [
        ModelBenchmark(
            model="meta/llama-3.1-70b", provider="nim",
            avg_quality=8.5, avg_latency_ms=1200, total_cost_usd=0.005, success_rate=1.0,
            avg_accuracy_f1=0.9, avg_grounding_rate=0.95, quality_cost_ratio=1700, tier=1,
            results=[BenchmarkResult(model="m", provider="nim", task="classify", quality_score=8.5,
                                     accuracy_f1=0.9, grounding_rate=0.95, latency_ms=1200, json_valid=True, schema_complete=1.0)],
        ),
        ModelBenchmark(
            model="meta/llama-3.1-8b", provider="nim",
            avg_quality=5.0, avg_latency_ms=400, total_cost_usd=0.001, success_rate=1.0, tier=2,
            results=[],
        ),
    ]
    md = format_benchmark_markdown(benchmarks)
    assert "Tier 1" in md
    assert "Tier 2" in md
    assert "meta/llama-3.1-70b" in md
    assert "8.5/10" in md
    assert "Accuracy" in md
    assert "Grounding" in md
