"""Benchmark history — load, compare, and trend past benchmark runs."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

BENCHMARK_DIR = Path.home() / ".deepscript" / "benchmarks"


def list_benchmark_runs() -> list[dict[str, Any]]:
    """List all saved benchmark runs, newest first."""
    if not BENCHMARK_DIR.exists():
        return []

    runs: list[dict[str, Any]] = []
    for f in sorted(BENCHMARK_DIR.glob("benchmark-*.json"), reverse=True):
        try:
            with open(f) as fh:
                data = json.load(fh)
            runs.append({
                "file": f.name,
                "path": str(f),
                "timestamp": data.get("timestamp", ""),
                "models_tested": data.get("models_tested", 0),
                "top_model": data["results"][0]["model"] if data.get("results") else "",
                "top_quality": data["results"][0]["avg_quality"] if data.get("results") else 0,
            })
        except (json.JSONDecodeError, KeyError, IndexError):
            continue

    return runs


def load_benchmark_run(path: str) -> dict[str, Any] | None:
    """Load a specific benchmark run by file path."""
    p = Path(path)
    if not p.exists():
        # Try as filename in benchmark dir
        p = BENCHMARK_DIR / path
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


def compare_runs(
    run_a: dict[str, Any],
    run_b: dict[str, Any],
) -> dict[str, Any]:
    """Compare two benchmark runs and show improvements/regressions.

    Args:
        run_a: Older run (baseline).
        run_b: Newer run (current).

    Returns:
        Comparison dict with per-model deltas.
    """
    # Build model lookup for each run
    models_a: dict[str, dict[str, Any]] = {}
    for r in run_a.get("results", []):
        models_a[r["model"]] = r

    models_b: dict[str, dict[str, Any]] = {}
    for r in run_b.get("results", []):
        models_b[r["model"]] = r

    # Find models in both runs
    common = set(models_a.keys()) & set(models_b.keys())
    only_a = set(models_a.keys()) - set(models_b.keys())
    only_b = set(models_b.keys()) - set(models_a.keys())

    comparisons: list[dict[str, Any]] = []
    for model in sorted(common):
        a = models_a[model]
        b = models_b[model]
        quality_delta = round(b["avg_quality"] - a["avg_quality"], 2)
        latency_delta = b.get("avg_latency_ms", 0) - a.get("avg_latency_ms", 0)
        tier_change = b.get("tier", 0) - a.get("tier", 0)

        status = "improved" if quality_delta > 0.5 else "regressed" if quality_delta < -0.5 else "stable"

        comparisons.append({
            "model": model,
            "quality_before": a["avg_quality"],
            "quality_after": b["avg_quality"],
            "quality_delta": quality_delta,
            "latency_before_ms": a.get("avg_latency_ms", 0),
            "latency_after_ms": b.get("avg_latency_ms", 0),
            "latency_delta_ms": latency_delta,
            "tier_before": a.get("tier", 0),
            "tier_after": b.get("tier", 0),
            "tier_change": tier_change,
            "status": status,
        })

    comparisons.sort(key=lambda c: c["quality_delta"], reverse=True)

    return {
        "baseline": run_a.get("timestamp", ""),
        "current": run_b.get("timestamp", ""),
        "models_compared": len(common),
        "models_added": sorted(only_b),
        "models_removed": sorted(only_a),
        "comparisons": comparisons,
        "improved": sum(1 for c in comparisons if c["status"] == "improved"),
        "regressed": sum(1 for c in comparisons if c["status"] == "regressed"),
        "stable": sum(1 for c in comparisons if c["status"] == "stable"),
    }


def model_trend(model_id: str) -> list[dict[str, Any]]:
    """Get quality/latency trend for a specific model across all runs."""
    if not BENCHMARK_DIR.exists():
        return []

    points: list[dict[str, Any]] = []
    for f in sorted(BENCHMARK_DIR.glob("benchmark-*.json")):
        try:
            with open(f) as fh:
                data = json.load(fh)
            for r in data.get("results", []):
                if r["model"] == model_id:
                    points.append({
                        "timestamp": data.get("timestamp", ""),
                        "quality": r["avg_quality"],
                        "latency_ms": r.get("avg_latency_ms", 0),
                        "tier": r.get("tier", 0),
                        "cost_usd": r.get("total_cost_usd", 0),
                        "success_rate": r.get("success_rate", 0),
                    })
                    break
        except (json.JSONDecodeError, KeyError):
            continue

    return points


def format_history_markdown(runs: list[dict[str, Any]]) -> str:
    """Format benchmark history as markdown."""
    if not runs:
        return "# Benchmark History\n\nNo benchmark runs found. Run `deepscript benchmark` first.\n"

    lines = [
        "# Benchmark History",
        "",
        f"*{len(runs)} runs on record*",
        "",
        "| # | Date | Models | Top Model | Best Quality |",
        "|---|------|--------|-----------|-------------|",
    ]
    for i, run in enumerate(runs, 1):
        ts = run["timestamp"][:19] if run.get("timestamp") else "?"
        lines.append(
            f"| {i} | {ts} | {run['models_tested']} | {run['top_model']} | {run['top_quality']}/10 |"
        )

    return "\n".join(lines)


def format_comparison_markdown(comparison: dict[str, Any]) -> str:
    """Format run comparison as markdown."""
    lines = [
        "# Benchmark Comparison",
        "",
        f"**Baseline:** {comparison['baseline'][:19]}",
        f"**Current:** {comparison['current'][:19]}",
        "",
        f"**Models compared:** {comparison['models_compared']} | "
        f"Improved: {comparison['improved']} | "
        f"Regressed: {comparison['regressed']} | "
        f"Stable: {comparison['stable']}",
        "",
    ]

    comps = comparison.get("comparisons", [])
    if comps:
        lines.append("| Model | Quality | Delta | Latency | Tier | Status |")
        lines.append("|-------|---------|-------|---------|------|--------|")
        for c in comps:
            delta_str = f"+{c['quality_delta']}" if c["quality_delta"] > 0 else str(c["quality_delta"])
            tier_str = f"{c['tier_before']}→{c['tier_after']}" if c["tier_change"] != 0 else str(c["tier_after"])
            lines.append(
                f"| {c['model']} | {c['quality_after']}/10 | {delta_str} | {c['latency_after_ms']}ms | {tier_str} | {c['status']} |"
            )
        lines.append("")

    added = comparison.get("models_added", [])
    if added:
        lines.append(f"**New models:** {', '.join(added)}")
    removed = comparison.get("models_removed", [])
    if removed:
        lines.append(f"**Removed models:** {', '.join(removed)}")

    return "\n".join(lines)


def model_stats(model_id: str) -> dict[str, Any]:
    """Compute aggregate statistics for a model across all benchmark runs.

    Includes mean, stddev, min, max for quality and latency.
    """
    points = model_trend(model_id)
    if not points:
        return {"model": model_id, "runs": 0}

    import math

    qualities = [p["quality"] for p in points]
    latencies = [p["latency_ms"] for p in points if p["latency_ms"] > 0]

    def _stats(values: list[float]) -> dict[str, float]:
        n = len(values)
        if n == 0:
            return {"mean": 0, "stddev": 0, "min": 0, "max": 0, "n": 0}
        mean = sum(values) / n
        if n > 1:
            variance = sum((x - mean) ** 2 for x in values) / (n - 1)
            stddev = math.sqrt(variance)
        else:
            stddev = 0.0
        return {
            "mean": round(mean, 2),
            "stddev": round(stddev, 2),
            "min": round(min(values), 2),
            "max": round(max(values), 2),
            "n": n,
        }

    return {
        "model": model_id,
        "runs": len(points),
        "quality": _stats(qualities),
        "latency_ms": _stats([float(x) for x in latencies]),
        "tiers": [p["tier"] for p in points],
        "current_tier": points[-1]["tier"] if points else 0,
    }


def format_trend_markdown(model_id: str, points: list[dict[str, Any]]) -> str:
    """Format model trend as markdown."""
    if not points:
        return f"# Trend: {model_id}\n\nNo benchmark data found for this model.\n"

    lines = [
        f"# Trend: {model_id}",
        "",
        f"*{len(points)} data points*",
        "",
        "| Date | Quality | Latency | Tier | Cost |",
        "|------|---------|---------|------|------|",
    ]
    for p in points:
        ts = p["timestamp"][:19] if p.get("timestamp") else "?"
        lines.append(
            f"| {ts} | {p['quality']}/10 | {p['latency_ms']}ms | {p['tier']} | ${p['cost_usd']:.4f} |"
        )

    # Trend + statistics
    if len(points) >= 2:
        stats = model_stats(model_id)
        qs = stats["quality"]
        ls = stats["latency_ms"]

        first = points[0]["quality"]
        last = points[-1]["quality"]
        delta = last - first
        trend = "improving" if delta > 0.5 else "declining" if delta < -0.5 else "stable"

        lines.append("")
        lines.append(f"**Trend:** {trend} ({first} → {last}, delta: {delta:+.1f})")
        lines.append("")
        lines.append("### Statistics")
        lines.append("")
        lines.append("| Metric | Mean | StdDev | Min | Max | Runs |")
        lines.append("|--------|------|--------|-----|-----|------|")
        lines.append(
            f"| Quality | {qs['mean']}/10 | ±{qs['stddev']} | {qs['min']} | {qs['max']} | {qs['n']} |"
        )
        if ls["n"] > 0:
            lines.append(
                f"| Latency | {ls['mean']}ms | ±{ls['stddev']}ms | {ls['min']}ms | {ls['max']}ms | {ls['n']} |"
            )

    return "\n".join(lines)
