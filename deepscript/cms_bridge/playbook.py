"""CMS playbook generation — distills patterns from episodes into actionable playbooks."""

from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_episodes(call_type: str, store_path: str) -> list[dict[str, Any]]:
    """Load all episodes for a call type from CMS store."""
    jsonl_path = Path(store_path) / "episodes" / "coding" / f"{call_type}.jsonl"
    if not jsonl_path.exists():
        return []

    episodes = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                episodes.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return episodes


def generate_playbook(call_type: str, store_path: str) -> str:
    """Generate a playbook from accumulated episodes.

    Reads episodes, extracts patterns, and produces versioned markdown.

    Args:
        call_type: The call type to generate a playbook for.
        store_path: Path to CMS store root.

    Returns:
        Playbook markdown string.
    """
    episodes = load_episodes(call_type, store_path)

    if not episodes:
        return f"# {call_type} Playbook\n\nNo episodes found. Analyze more calls with `--cms` to build this playbook.\n"

    # Extract patterns
    all_findings: list[str] = []
    all_scores: dict[str, list[float]] = {}
    score_totals: list[float] = []

    for ep in episodes:
        outcome = ep.get("outcome", {})
        findings = outcome.get("findings", [])
        all_findings.extend(findings)

        scores = outcome.get("scores", {})
        for key, val in scores.items():
            if isinstance(val, (int, float)):
                all_scores.setdefault(key, []).append(val)

        overall = scores.get("overall")
        if isinstance(overall, (int, float)):
            score_totals.append(overall)

    # Rank findings by frequency
    finding_counts = Counter(all_findings)
    top_patterns = finding_counts.most_common(20)

    # Split into "what works" vs "what fails" (heuristic: positive/negative language)
    works = []
    fails = []
    for finding, count in top_patterns:
        finding_lower = finding.lower()
        if any(w in finding_lower for w in ["risk", "missing", "weak", "gap", "concern", "anti"]):
            fails.append((finding, count))
        else:
            works.append((finding, count))

    # Compute benchmarks
    benchmarks: dict[str, dict[str, float]] = {}
    for key, values in all_scores.items():
        if values:
            benchmarks[key] = {
                "avg": round(sum(values) / len(values), 2),
                "min": round(min(values), 2),
                "max": round(max(values), 2),
                "count": len(values),
            }

    # Build markdown
    lines = [
        f"# {call_type.replace('-', ' ').title()} Playbook",
        "",
        f"*Auto-generated from {len(episodes)} analyzed calls*",
        f"*Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*",
        "",
    ]

    if works:
        lines.append("## What Works")
        lines.append("")
        for finding, count in works:
            lines.append(f"- {finding} *(seen {count}x)*")
        lines.append("")

    if fails:
        lines.append("## What to Watch")
        lines.append("")
        for finding, count in fails:
            lines.append(f"- {finding} *(seen {count}x)*")
        lines.append("")

    if benchmarks:
        lines.append("## Benchmarks")
        lines.append("")
        lines.append("| Metric | Avg | Min | Max | Calls |")
        lines.append("|--------|-----|-----|-----|-------|")
        for key, vals in benchmarks.items():
            lines.append(f"| {key} | {vals['avg']} | {vals['min']} | {vals['max']} | {vals['count']} |")
        lines.append("")

    if score_totals:
        avg_overall = round(sum(score_totals) / len(score_totals), 2)
        lines.append(f"**Average Overall Score:** {avg_overall}/10 across {len(score_totals)} calls")
        lines.append("")

    md = "\n".join(lines)

    # Save to CMS store
    playbook_dir = Path(store_path) / "semantic" / "playbooks"
    playbook_dir.mkdir(parents=True, exist_ok=True)
    playbook_path = playbook_dir / f"{call_type}.md"
    with open(playbook_path, "w", encoding="utf-8") as f:
        f.write(md)

    logger.info("Playbook written to %s", playbook_path)
    return md
