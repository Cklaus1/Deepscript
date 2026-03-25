"""Working memory assembly — prep context for upcoming calls."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def assemble_working_memory(
    call_type: str,
    store_path: str,
    token_budget: int = 4000,
) -> dict[str, Any]:
    """Assemble working memory for call prep.

    Combines:
    - Relevant playbook sections
    - Recent episode patterns
    - Dead-end warnings

    Args:
        call_type: The call type to prep for.
        store_path: Path to CMS store root.
        token_budget: Approximate max tokens for the assembled context.

    Returns:
        Dict with playbook, recent_patterns, dead_ends, and prep_notes.
    """
    store = Path(store_path)
    result: dict[str, Any] = {
        "call_type": call_type,
        "playbook": None,
        "recent_patterns": [],
        "dead_ends": [],
        "prep_notes": [],
    }

    # Load playbook
    playbook_path = store / "semantic" / "playbooks" / f"{call_type}.md"
    if playbook_path.exists():
        playbook_text = playbook_path.read_text(encoding="utf-8")
        # Truncate to budget
        if len(playbook_text.split()) > token_budget // 2:
            words = playbook_text.split()
            playbook_text = " ".join(words[: token_budget // 2]) + "\n[...truncated...]"
        result["playbook"] = playbook_text

    # Load recent episodes (last 10)
    episodes_path = store / "episodes" / "coding" / f"{call_type}.jsonl"
    if episodes_path.exists():
        recent = []
        with open(episodes_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines[-10:]:
            try:
                ep = json.loads(line.strip())
                findings = ep.get("outcome", {}).get("findings", [])
                scores = ep.get("outcome", {}).get("scores", {})
                recent.append({
                    "timestamp": ep.get("timestamp", ""),
                    "findings": findings[:5],
                    "overall_score": scores.get("overall"),
                })
            except json.JSONDecodeError:
                continue
        result["recent_patterns"] = recent

    # Load dead-ends
    dead_ends_path = store / "gating" / "dead-ends.jsonl"
    if dead_ends_path.exists():
        dead_ends = []
        with open(dead_ends_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    de = json.loads(line.strip())
                    if de.get("task_type") == call_type or de.get("call_type") == call_type:
                        dead_ends.append(de.get("pattern", de.get("finding", str(de))))
                except json.JSONDecodeError:
                    continue
        result["dead_ends"] = dead_ends[:10]

    # Generate prep notes from patterns
    if result["recent_patterns"]:
        all_findings = []
        for rp in result["recent_patterns"]:
            all_findings.extend(rp.get("findings", []))
        if all_findings:
            # Deduplicate and take top 5
            seen = set()
            for f in all_findings:
                if f not in seen:
                    result["prep_notes"].append(f)
                    seen.add(f)
                if len(result["prep_notes"]) >= 5:
                    break

    return result


def format_prep_markdown(working_memory: dict[str, Any]) -> str:
    """Format working memory as markdown for display."""
    call_type = working_memory.get("call_type", "unknown")
    lines = [
        f"# Call Prep: {call_type.replace('-', ' ').title()}",
        "",
    ]

    prep_notes = working_memory.get("prep_notes", [])
    if prep_notes:
        lines.append("## Key Patterns from Previous Calls")
        lines.append("")
        for note in prep_notes:
            lines.append(f"- {note}")
        lines.append("")

    dead_ends = working_memory.get("dead_ends", [])
    if dead_ends:
        lines.append("## Avoid (Dead Ends)")
        lines.append("")
        for de in dead_ends:
            lines.append(f"- {de}")
        lines.append("")

    playbook = working_memory.get("playbook")
    if playbook:
        lines.append("## Playbook Reference")
        lines.append("")
        lines.append(playbook)

    recent = working_memory.get("recent_patterns", [])
    if recent:
        avg_scores = [r["overall_score"] for r in recent if r.get("overall_score")]
        if avg_scores:
            avg = round(sum(avg_scores) / len(avg_scores), 1)
            lines.append(f"\n*Recent avg score: {avg}/10 across {len(avg_scores)} calls*")

    if not prep_notes and not dead_ends and not playbook:
        lines.append("No previous call data available. Analyze calls with `--cms` to build prep context.")

    return "\n".join(lines)
