"""Token usage and cost tracking for LLM calls — in-memory + persistent."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Approximate pricing per 1M tokens (USD) — updated as needed
MODEL_PRICING: dict[str, dict[str, float]] = {
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.0},
    "claude-opus-4-6": {"input": 15.0, "output": 75.0},
}

DEFAULT_PRICING = {"input": 3.0, "output": 15.0}

USAGE_DIR = Path.home() / ".deepscript"
USAGE_FILE = USAGE_DIR / "usage.jsonl"


@dataclass
class UsageEntry:
    timestamp: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    source_file: str = ""
    call_type: str = ""
    latency_ms: int = 0  # Response time in milliseconds
    provider: str = ""  # claude, openai, ollama, vllm, sglang, nim


@dataclass
class CostTracker:
    """Tracks cumulative LLM token usage and estimated cost."""

    budget_limit: float = 50.0
    entries: list[UsageEntry] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    budget_exceeded: bool = False

    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: int = 0,
        provider: str = "",
    ) -> None:
        """Record a single LLM call's token usage and performance."""
        pricing = MODEL_PRICING.get(model, DEFAULT_PRICING)
        cost = (
            input_tokens * pricing["input"] / 1_000_000
            + output_tokens * pricing["output"] / 1_000_000
        )

        entry = UsageEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=round(cost, 6),
            latency_ms=latency_ms,
            provider=provider,
        )
        self.entries.append(entry)
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost_usd += cost

        if self.total_cost_usd > self.budget_limit:
            logger.warning(
                "LLM cost ($%.4f) exceeds monthly budget ($%.2f)",
                self.total_cost_usd,
                self.budget_limit,
            )
            self.budget_exceeded = True

    def summary(self) -> dict[str, Any]:
        """Return a summary of this session's usage."""
        return {
            "calls": len(self.entries),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "budget_limit_usd": self.budget_limit,
        }

    def persist(self, source_file: str = "", call_type: str = "") -> None:
        """Append this session's entries to persistent usage log."""
        if not self.entries:
            return

        USAGE_DIR.mkdir(parents=True, exist_ok=True)
        with open(USAGE_FILE, "a", encoding="utf-8") as f:
            for entry in self.entries:
                entry.source_file = source_file
                entry.call_type = call_type
                f.write(json.dumps(asdict(entry), default=str) + "\n")


def load_usage_history(
    days: int | None = None,
    month: str | None = None,
) -> list[UsageEntry]:
    """Load usage entries from persistent log.

    Args:
        days: Filter to last N days. None = all.
        month: Filter to specific month (YYYY-MM). None = all.
    """
    if not USAGE_FILE.exists():
        return []

    entries: list[UsageEntry] = []
    now = datetime.now(timezone.utc)

    with open(USAGE_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                entry = UsageEntry(**data)

                # Filter by days
                if days is not None:
                    ts = datetime.fromisoformat(entry.timestamp.replace("Z", "+00:00"))
                    delta = (now - ts).days
                    if delta > days:
                        continue

                # Filter by month
                if month is not None:
                    if not entry.timestamp.startswith(month):
                        continue

                entries.append(entry)
            except (json.JSONDecodeError, TypeError):
                continue

    return entries


def usage_summary(
    entries: list[UsageEntry],
    budget_limit: float = 50.0,
) -> dict[str, Any]:
    """Compute aggregate usage stats from entries."""
    if not entries:
        return {
            "calls": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cost_usd": 0.0,
            "budget_limit_usd": budget_limit,
            "budget_remaining_usd": budget_limit,
            "models": {},
            "by_call_type": {},
        }

    total_input = sum(e.input_tokens for e in entries)
    total_output = sum(e.output_tokens for e in entries)
    total_cost = sum(e.cost_usd for e in entries)

    # Per-model breakdown with performance stats
    models: dict[str, dict[str, Any]] = {}
    for e in entries:
        if e.model not in models:
            models[e.model] = {
                "calls": 0, "input_tokens": 0, "output_tokens": 0,
                "cost_usd": 0.0, "provider": e.provider,
                "latencies_ms": [],
            }
        models[e.model]["calls"] += 1
        models[e.model]["input_tokens"] += e.input_tokens
        models[e.model]["output_tokens"] += e.output_tokens
        models[e.model]["cost_usd"] = round(models[e.model]["cost_usd"] + e.cost_usd, 6)
        if e.latency_ms > 0:
            models[e.model]["latencies_ms"].append(e.latency_ms)

    # Compute latency stats per model
    for model_data in models.values():
        lats = model_data.pop("latencies_ms")
        if lats:
            model_data["avg_latency_ms"] = round(sum(lats) / len(lats))
            model_data["min_latency_ms"] = min(lats)
            model_data["max_latency_ms"] = max(lats)
            model_data["p50_latency_ms"] = sorted(lats)[len(lats) // 2]

    # Per-call-type breakdown
    by_type: dict[str, dict[str, Any]] = {}
    for e in entries:
        ct = e.call_type or "unknown"
        if ct not in by_type:
            by_type[ct] = {"calls": 0, "cost_usd": 0.0}
        by_type[ct]["calls"] += 1
        by_type[ct]["cost_usd"] = round(by_type[ct]["cost_usd"] + e.cost_usd, 6)

    return {
        "calls": len(entries),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_cost_usd": round(total_cost, 6),
        "budget_limit_usd": budget_limit,
        "budget_remaining_usd": round(budget_limit - total_cost, 6),
        "models": models,
        "by_call_type": by_type,
        "first_entry": entries[0].timestamp if entries else None,
        "last_entry": entries[-1].timestamp if entries else None,
    }


def clear_usage() -> int:
    """Clear the usage log. Returns number of entries cleared."""
    if not USAGE_FILE.exists():
        return 0
    with open(USAGE_FILE, "r") as f:
        count = sum(1 for line in f if line.strip())
    USAGE_FILE.unlink()
    return count
