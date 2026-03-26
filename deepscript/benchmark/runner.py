"""Benchmark runner — evaluate models on transcript analysis tasks."""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from deepscript.config.settings import LLMConfig
from deepscript.llm.provider import LLMProvider

logger = logging.getLogger(__name__)

BENCHMARK_DIR = Path.home() / ".deepscript" / "benchmarks"

# Default rate limit for NIM free tier
DEFAULT_RATE_LIMIT = 35  # requests per minute (stay under 40 with margin)


class RateLimiter:
    """Thread-safe token bucket rate limiter."""

    def __init__(self, requests_per_minute: int = DEFAULT_RATE_LIMIT) -> None:
        self.interval = 60.0 / requests_per_minute  # seconds between requests
        self._lock = threading.Lock()
        self._last_request = 0.0

    def wait(self) -> None:
        """Block until a request is allowed."""
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_request
            if elapsed < self.interval:
                sleep_time = self.interval - elapsed
                time.sleep(sleep_time)
            self._last_request = time.monotonic()


# Global rate limiter — shared across all benchmark threads
_rate_limiter: RateLimiter | None = None


def _get_rate_limiter(rpm: int = DEFAULT_RATE_LIMIT) -> RateLimiter:
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(rpm)
    return _rate_limiter

# Evaluation criteria weights (sum to 1.0)
QUALITY_WEIGHTS = {
    "format_valid": 0.10,      # Did it return valid JSON / follow format?
    "schema_complete": 0.15,   # Are all expected fields present?
    "accuracy": 0.30,          # Precision/recall vs ground truth
    "grounding": 0.25,         # Are extracted items actually in the transcript? (anti-hallucination)
    "evidence_quality": 0.20,  # Are findings specific with evidence?
}


@dataclass
class BenchmarkResult:
    """Result of benchmarking a single model on a single task."""

    model: str
    provider: str
    task: str  # e.g., "classify", "summarize", "sales_score"
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    json_valid: bool = False
    schema_complete: float = 0.0  # 0-1, fraction of expected fields present
    accuracy_f1: float = 0.0     # F1 vs ground truth
    grounding_rate: float = 0.0  # Fraction of items that appear in transcript
    hallucinated_count: int = 0
    response_length: int = 0
    error: str | None = None
    quality_score: float = 0.0  # 0-10 composite
    raw_response: str = ""


@dataclass
class ModelBenchmark:
    """Aggregate benchmark for a model across all tasks."""

    model: str
    provider: str
    results: list[BenchmarkResult] = field(default_factory=list)
    avg_quality: float = 0.0
    avg_latency_ms: int = 0
    total_cost_usd: float = 0.0
    success_rate: float = 0.0
    avg_accuracy_f1: float = 0.0
    avg_grounding_rate: float = 0.0
    total_hallucinated: int = 0
    quality_cost_ratio: float = 0.0  # quality per dollar
    tier: int = 0  # 1=top, 2=strong, 3=fast, 0=unranked

    def compute_aggregates(self) -> None:
        """Compute aggregate stats from individual results."""
        if not self.results:
            return
        successes = [r for r in self.results if not r.error]
        self.success_rate = len(successes) / len(self.results) if self.results else 0
        if successes:
            self.avg_quality = round(sum(r.quality_score for r in successes) / len(successes), 2)
            self.avg_latency_ms = round(sum(r.latency_ms for r in successes) / len(successes))
            self.avg_accuracy_f1 = round(sum(r.accuracy_f1 for r in successes) / len(successes), 3)
            self.avg_grounding_rate = round(sum(r.grounding_rate for r in successes) / len(successes), 3)
        self.total_cost_usd = round(sum(r.cost_usd for r in self.results), 6)
        self.total_hallucinated = sum(r.hallucinated_count for r in self.results)

        # Quality per dollar (higher is better)
        if self.total_cost_usd > 0:
            self.quality_cost_ratio = round(self.avg_quality / self.total_cost_usd, 1)

        # Auto-tier based on quality
        if self.avg_quality >= 7.5:
            self.tier = 1
        elif self.avg_quality >= 5.0:
            self.tier = 2
        elif self.avg_quality > 0:
            self.tier = 3


# --- Benchmark Tasks ---

BENCHMARK_TASKS = {
    "classify": {
        "prompt_template": "classify",
        "expected_fields": ["call_type", "confidence", "reasoning"],
        "description": "Classify transcript type",
    },
    "summarize": {
        "prompt_template": "summarize",
        "expected_fields": [],  # Free-form text
        "description": "Generate abstractive summary",
    },
    "action_items": {
        "prompt_template": "action_items",
        "expected_fields": ["text", "assignee", "speaker"],
        "description": "Extract action items",
    },
    "sales_score": {
        "prompt_template": "sales_score",
        "expected_fields": ["methodology", "scores", "total_score", "strengths", "gaps"],
        "description": "Score sales call methodology",
    },
    "discovery_score": {
        "prompt_template": "discovery_score",
        "expected_fields": ["framework", "framework_score", "pain_points", "jtbd"],
        "description": "Score discovery call",
    },
}


def _load_test_transcript() -> tuple[str, str]:
    """Load the primary test transcript (for backward compat).

    Returns: (transcript_text, fixture_name).
    """
    transcripts = _load_all_test_transcripts()
    if transcripts:
        return transcripts[0]
    return "This is a test transcript for benchmarking.", ""


def _load_all_test_transcripts(max_words: int = 500) -> list[tuple[str, str]]:
    """Load ALL test transcripts for multi-fixture benchmarking.

    Args:
        max_words: Max words per transcript (default 500 — enough for classify, saves tokens).

    Returns: list of (transcript_text, fixture_name).
    """
    fixture_dir = Path(__file__).parent.parent.parent / "tests" / "fixtures"
    fixtures = [
        ("sales_transcript", fixture_dir / "sales_transcript.json"),
        ("sample_transcript", fixture_dir / "sample_transcript.json"),
        ("discovery_transcript", fixture_dir / "discovery_transcript.json"),
        ("pmf_transcript", fixture_dir / "pmf_transcript.json"),
        ("interview_transcript", fixture_dir / "interview_transcript.json"),
        ("real_product_discussion", fixture_dir / "real_product_discussion.json"),
        ("real_meeting", fixture_dir / "real_meeting.json"),
    ]
    result = []
    for name, fp in fixtures:
        if fp.exists():
            with open(fp) as f:
                data = json.load(f)
            text = data.get("text", "")
            # Truncate for cost and context window control
            words = text.split()
            if len(words) > max_words:
                text = " ".join(words[:max_words]) + "\n[...truncated...]"
            result.append((text, name))
    return result


@dataclass
class ScoreResult:
    """Detailed scoring of a single response."""

    json_valid: bool = False
    schema_complete: float = 0.0
    accuracy_f1: float = 0.0
    grounding_rate: float = 1.0
    hallucinated_count: int = 0
    evidence_quality: float = 0.0
    quality_score: float = 0.0


def _score_response(
    task_name: str,
    task_config: dict[str, Any],
    response: str | None,
    transcript_text: str = "",
    fixture_name: str = "",
) -> ScoreResult:
    """Score a model response on quality using ground truth and hallucination detection.

    Returns ScoreResult with detailed metrics.
    """
    from deepscript.benchmark.ground_truth import (
        GROUND_TRUTHS,
        compute_precision_recall,
        verify_against_transcript,
    )

    if response is None:
        return ScoreResult()

    # For summarize task
    if task_name == "summarize":
        gt = GROUND_TRUTHS.get(fixture_name)
        keyword_score = 0.0
        if gt and gt.expected_summary_keywords:
            resp_lower = response.lower()
            hits = sum(1 for kw in gt.expected_summary_keywords if kw.lower() in resp_lower)
            keyword_score = hits / len(gt.expected_summary_keywords)

        grounding = verify_against_transcript([response], transcript_text) if transcript_text else {}
        quality = (
            1.0 * QUALITY_WEIGHTS["format_valid"]
            + 1.0 * QUALITY_WEIGHTS["schema_complete"]
            + keyword_score * QUALITY_WEIGHTS["accuracy"]
            + grounding.get("grounding_rate", 1.0) * QUALITY_WEIGHTS["grounding"]
            + min(1.0, len(response.split()) / 30) * QUALITY_WEIGHTS["evidence_quality"]
        ) * 10

        return ScoreResult(
            json_valid=True,
            schema_complete=1.0,
            accuracy_f1=round(keyword_score, 3),
            grounding_rate=grounding.get("grounding_rate", 1.0),
            quality_score=round(quality, 2),
        )

    # Parse JSON
    text = response.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return ScoreResult(quality_score=0.5)  # At least it responded

    # Schema completeness
    expected = task_config.get("expected_fields", [])
    if expected:
        if isinstance(parsed, list):
            if parsed and isinstance(parsed[0], dict):
                present = sum(1 for f in expected if f in parsed[0])
            else:
                present = 0
        elif isinstance(parsed, dict):
            present = sum(1 for f in expected if f in parsed)
        else:
            present = 0
        schema_complete = present / len(expected) if expected else 1.0
    else:
        schema_complete = 1.0

    # --- Ground truth accuracy ---
    accuracy_f1 = 0.0
    gt = GROUND_TRUTHS.get(fixture_name)
    if gt:
        if task_name == "classify" and isinstance(parsed, dict):
            # Match on call_type — accept primary or alternative classifications
            predicted_type = parsed.get("call_type", "")
            valid_types = {gt.call_type} | set(gt.accepted_types)
            accuracy_f1 = 1.0 if predicted_type in valid_types else 0.0

        elif task_name == "action_items" and isinstance(parsed, list):
            extracted = [item.get("text", str(item)) if isinstance(item, dict) else str(item) for item in parsed]
            pr = compute_precision_recall(extracted, gt.action_items)
            accuracy_f1 = pr["f1"]

        elif task_name == "sales_score" and isinstance(parsed, dict):
            # Check if key dimensions are present and scored
            scores = parsed.get("scores", {})
            if scores:
                accuracy_f1 = min(1.0, len(scores) / 6)  # MEDDIC has 6 dimensions

        elif task_name == "discovery_score" and isinstance(parsed, dict):
            # Check pain point extraction accuracy
            extracted_pains = [p.get("pain", str(p)) if isinstance(p, dict) else str(p)
                              for p in parsed.get("pain_points", [])]
            if gt.pain_points:
                pr = compute_precision_recall(extracted_pains, gt.pain_points)
                accuracy_f1 = pr["f1"]
            else:
                accuracy_f1 = 1.0 if not extracted_pains else 0.5

    # --- Hallucination detection ---
    grounding_rate = 1.0
    hallucinated_count = 0
    if transcript_text and isinstance(parsed, (dict, list)):
        # Extract text items to verify against transcript
        items_to_verify: list[str] = []
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    items_to_verify.append(item.get("text", item.get("quote", str(item))))
                else:
                    items_to_verify.append(str(item))
        elif isinstance(parsed, dict):
            # Check specific fields that should be grounded
            for key in ("reasoning", "quote", "evidence"):
                val = parsed.get(key)
                if isinstance(val, str) and val:
                    items_to_verify.append(val)
            # Check nested items
            for key in ("findings", "strengths", "gaps", "strongest_signals"):
                val = parsed.get(key, [])
                if isinstance(val, list):
                    items_to_verify.extend(str(v) for v in val if v)

        if items_to_verify:
            grounding = verify_against_transcript(items_to_verify, transcript_text)
            grounding_rate = grounding["grounding_rate"]
            hallucinated_count = grounding["hallucinated"]

    # --- Evidence quality ---
    evidence_quality = 0.0
    if isinstance(parsed, dict):
        non_empty = sum(1 for v in parsed.values()
                        if v and v != 0 and v != [] and v != {} and v != "null")
        evidence_quality = min(1.0, non_empty / max(len(parsed), 1))
    elif isinstance(parsed, list):
        evidence_quality = min(1.0, len(parsed) / 3)

    # Composite quality score (0-10)
    quality = (
        1.0 * QUALITY_WEIGHTS["format_valid"]
        + schema_complete * QUALITY_WEIGHTS["schema_complete"]
        + accuracy_f1 * QUALITY_WEIGHTS["accuracy"]
        + grounding_rate * QUALITY_WEIGHTS["grounding"]
        + evidence_quality * QUALITY_WEIGHTS["evidence_quality"]
    ) * 10

    return ScoreResult(
        json_valid=True,
        schema_complete=round(schema_complete, 3),
        accuracy_f1=round(accuracy_f1, 3),
        grounding_rate=round(grounding_rate, 3),
        hallucinated_count=hallucinated_count,
        evidence_quality=round(evidence_quality, 3),
        quality_score=round(quality, 2),
    )


def _benchmark_single_model(
    model_id: str,
    provider: str,
    task_names: list[str],
    transcripts: list[tuple[str, str]],
    base_url: str | None,
    api_key: str | None,
    max_tokens: int,
    rate_limiter: RateLimiter | None = None,
) -> ModelBenchmark:
    """Benchmark a single model across all tasks and all fixtures. Thread-safe."""
    config = LLMConfig(
        provider=provider, model=model_id, max_tokens=max_tokens,
        base_url=base_url, api_key=api_key, cost_tracking=True,
    )
    llm = LLMProvider(config)
    model_bench = ModelBenchmark(model=model_id, provider=provider)
    limiter = rate_limiter or _get_rate_limiter()

    for transcript_text, fixture_name in transcripts:
        for task_name in task_names:
            task_config = BENCHMARK_TASKS.get(task_name)
            if not task_config:
                continue

            # Build prompt for this task + transcript
            template = task_config["prompt_template"]
            try:
                extra_kwargs: dict[str, str] = {}
                if template == "sales_score":
                    extra_kwargs = {"methodology": "MEDDIC", "methodology_description": "MEDDIC Framework"}
                elif template == "discovery_score":
                    extra_kwargs = {"framework": "Mom Test", "framework_description": "The Mom Test framework"}
                prompt = llm.render_prompt(template, transcript=transcript_text, **extra_kwargs)
            except Exception as e:
                logger.warning("Failed to render prompt %s for %s: %s", template, fixture_name, e)
                continue

            limiter.wait()  # Respect API rate limits
            start = time.monotonic()
            try:
                response = llm.complete(prompt, max_tokens=max_tokens)
                elapsed_ms = int((time.monotonic() - start) * 1000)

                scored = _score_response(
                    task_name, task_config, response,
                    transcript_text=transcript_text, fixture_name=fixture_name,
                )

                input_tokens = 0
                output_tokens = 0
                cost = 0.0
                if llm.cost_tracker.entries:
                    last = llm.cost_tracker.entries[-1]
                    input_tokens = last.input_tokens
                    output_tokens = last.output_tokens
                    cost = last.cost_usd

                result = BenchmarkResult(
                    model=model_id, provider=provider,
                    task=f"{task_name}:{fixture_name}",
                    latency_ms=elapsed_ms, input_tokens=input_tokens,
                    output_tokens=output_tokens, cost_usd=cost,
                    json_valid=scored.json_valid, schema_complete=scored.schema_complete,
                    accuracy_f1=scored.accuracy_f1, grounding_rate=scored.grounding_rate,
                    hallucinated_count=scored.hallucinated_count,
                    response_length=len(response) if response else 0,
                    quality_score=scored.quality_score,
                    raw_response=(response or "")[:500],
                )
            except Exception as e:
                elapsed_ms = int((time.monotonic() - start) * 1000)
                result = BenchmarkResult(
                    model=model_id, provider=provider,
                    task=f"{task_name}:{fixture_name}",
                    latency_ms=elapsed_ms, error=str(e)[:200],
                )

            model_bench.results.append(result)
            logger.info("  [%s] %s/%s: quality=%.1f, latency=%dms",
                         model_id.split("/")[-1], task_name, fixture_name,
                         result.quality_score, result.latency_ms)

    model_bench.compute_aggregates()
    return model_bench


def run_benchmark(
    models: list[str],
    provider: str = "nim",
    tasks: list[str] | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    max_tokens: int = 2048,
    parallel: bool = True,
    max_parallel: int = 5,
    rate_limit_rpm: int = DEFAULT_RATE_LIMIT,
) -> list[ModelBenchmark]:
    """Run benchmarks across models and tasks.

    Rate-limited to stay under API quotas (default 35 req/min for NIM free tier).
    Models are benchmarked in parallel (each model's tasks run sequentially
    for clean latency measurements). Set parallel=False for sequential.
    """
    task_names = tasks or list(BENCHMARK_TASKS.keys())

    # Use shorter transcripts for classify-only benchmarks
    max_words = 500 if task_names == ["classify"] else 1500
    transcripts = _load_all_test_transcripts(max_words=max_words)
    if not transcripts:
        transcripts = [_load_test_transcript()]

    # Create shared rate limiter for all models
    global _rate_limiter
    _rate_limiter = RateLimiter(rate_limit_rpm)
    total_calls = len(models) * len(task_names) * len(transcripts)
    est_minutes = max(total_calls / rate_limit_rpm, total_calls * 2 / 60)  # account for response time
    logger.info("Benchmarking %d models × %d tasks × %d transcripts = %d calls (~%.0f min)",
                len(models), len(task_names), len(transcripts), total_calls, est_minutes)

    # Incremental results file — updated after each model completes
    incremental_path = BENCHMARK_DIR / "benchmark-latest.json"
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    benchmarks: list[ModelBenchmark] = []

    def _on_model_done(bench: ModelBenchmark) -> None:
        """Save results incrementally after each model."""
        benchmarks.append(bench)
        _save_incremental(benchmarks, incremental_path)
        status = f"Q={bench.avg_quality:.1f}" if bench.avg_quality > 0 else "FAIL"
        logger.info("  ✓ %s: %s (%d/%d models done)",
                     bench.model, status, len(benchmarks), len(models))

    if parallel and len(models) > 1:
        _run_parallel(
            models, provider, task_names, transcripts,
            base_url, api_key, max_tokens, max_parallel, _on_model_done,
        )
    else:
        for m in models:
            bench = _benchmark_single_model(
                m, provider, task_names, transcripts,
                base_url, api_key, max_tokens, _rate_limiter,
            )
            _on_model_done(bench)

    benchmarks.sort(key=lambda b: b.avg_quality, reverse=True)
    return benchmarks


def _save_incremental(benchmarks: list[ModelBenchmark], path: Path) -> None:
    """Save current results incrementally (called after each model)."""
    sorted_benchmarks = sorted(benchmarks, key=lambda b: b.avg_quality, reverse=True)
    data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "models_tested": len(sorted_benchmarks),
        "in_progress": True,
        "results": [
            {
                "model": b.model,
                "provider": b.provider,
                "tier": b.tier,
                "avg_quality": b.avg_quality,
                "avg_latency_ms": b.avg_latency_ms,
                "total_cost_usd": b.total_cost_usd,
                "success_rate": b.success_rate,
                "avg_accuracy_f1": b.avg_accuracy_f1,
                "avg_grounding_rate": b.avg_grounding_rate,
                "tasks": [asdict(r) for r in b.results],
            }
            for b in sorted_benchmarks
        ],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def _run_parallel(
    models: list[str],
    provider: str,
    task_names: list[str],
    transcripts: list[tuple[str, str]],
    base_url: str | None,
    api_key: str | None,
    max_tokens: int,
    max_parallel: int,
    on_done: Any = None,
) -> None:
    """Benchmark models in parallel using asyncio + thread pool."""
    import asyncio

    async def run_all() -> None:
        sem = asyncio.Semaphore(max_parallel)

        async def bench_one(model_id: str) -> None:
            async with sem:
                bench = await asyncio.to_thread(
                    _benchmark_single_model,
                    model_id, provider, task_names, transcripts,
                    base_url, api_key, max_tokens, _rate_limiter,
                )
            if on_done:
                on_done(bench)

        gather_tasks = [bench_one(m) for m in models]
        await asyncio.gather(*gather_tasks, return_exceptions=True)

    asyncio.run(run_all())


def save_benchmark_results(benchmarks: list[ModelBenchmark]) -> Path:
    """Save benchmark results to ~/.deepscript/benchmarks/."""
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output_path = BENCHMARK_DIR / f"benchmark-{timestamp}.json"

    data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "models_tested": len(benchmarks),
        "results": [
            {
                "model": b.model,
                "provider": b.provider,
                "tier": b.tier,
                "avg_quality": b.avg_quality,
                "avg_latency_ms": b.avg_latency_ms,
                "total_cost_usd": b.total_cost_usd,
                "success_rate": b.success_rate,
                "tasks": [asdict(r) for r in b.results],
            }
            for b in benchmarks
        ],
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    return output_path


def format_benchmark_markdown(benchmarks: list[ModelBenchmark]) -> str:
    """Format benchmark results as markdown."""
    lines = [
        "# DeepScript Model Benchmark Results",
        "",
        f"*{len(benchmarks)} models tested — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*",
        "",
    ]

    # Tier summary
    for tier, label in [(1, "Tier 1 — Top Quality"), (2, "Tier 2 — Strong"), (3, "Tier 3 — Fast/Budget")]:
        tier_models = [b for b in benchmarks if b.tier == tier]
        if not tier_models:
            continue
        lines.append(f"## {label}")
        lines.append("")
        lines.append("| Model | Quality | Accuracy | Grounding | Halluc. | Latency | Cost | $/Q |")
        lines.append("|-------|---------|----------|-----------|---------|---------|------|-----|")
        for b in tier_models:
            qcr = f"{b.quality_cost_ratio:.0f}" if b.quality_cost_ratio else "-"
            lines.append(
                f"| {b.model} | {b.avg_quality}/10 | {b.avg_accuracy_f1:.2f} | {b.avg_grounding_rate:.2f} | {b.total_hallucinated} | {b.avg_latency_ms}ms | ${b.total_cost_usd:.4f} | {qcr} |"
            )
        lines.append("")

    # Unranked
    unranked = [b for b in benchmarks if b.tier == 0]
    if unranked:
        lines.append("## Unranked (no successful responses)")
        lines.append("")
        for b in unranked:
            error = b.results[0].error if b.results else "unknown"
            lines.append(f"- {b.model}: {error}")
        lines.append("")

    # Detailed per-task breakdown for top 5
    top5 = [b for b in benchmarks if b.tier in (1, 2)][:5]
    if top5:
        lines.append("## Per-Task Breakdown (Top 5)")
        lines.append("")
        for b in top5:
            lines.append(f"### {b.model}")
            lines.append("")
            lines.append("| Task | Quality | Accuracy | Grounding | Halluc. | Latency | JSON |")
            lines.append("|------|---------|----------|-----------|---------|---------|------|")
            for r in b.results:
                lines.append(
                    f"| {r.task} | {r.quality_score}/10 | {r.accuracy_f1:.2f} | {r.grounding_rate:.2f} | {r.hallucinated_count} | {r.latency_ms}ms | {'yes' if r.json_valid else 'no'} |"
                )
            lines.append("")

    return "\n".join(lines)
