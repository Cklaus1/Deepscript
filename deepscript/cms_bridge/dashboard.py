"""PMF Dashboard — Cross-call Vohra engine for Product-Market Fit tracking."""

from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def generate_pmf_dashboard(store_path: str, ellis_threshold: float = 0.40) -> str:
    """Generate a PMF dashboard from accumulated pmf-call episodes.

    Implements the Sean Ellis / Rahul Vohra framework:
    - Ellis distribution (very/somewhat/not disappointed)
    - PMF score trend over time
    - Signal analysis

    Args:
        store_path: Path to CMS store root.
        ellis_threshold: Target % of "very disappointed" for PMF (default 40%).

    Returns:
        Dashboard markdown string.
    """
    jsonl_path = Path(store_path) / "episodes" / "coding" / "pmf-call.jsonl"
    if not jsonl_path.exists():
        return "# PMF Dashboard\n\nNo PMF call episodes found. Analyze customer calls with `--type pmf-call --cms` to build this dashboard.\n"

    episodes = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    episodes.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not episodes:
        return "# PMF Dashboard\n\nNo PMF episodes found.\n"

    # Extract PMF data from findings
    ellis_counts: Counter[str] = Counter()
    pmf_scores: list[float] = []
    all_findings: list[str] = []
    strongest_signals: list[str] = []
    anti_pmf: list[str] = []

    for ep in episodes:
        outcome = ep.get("outcome", {})
        scores = outcome.get("scores", {})
        findings = outcome.get("findings", [])
        all_findings.extend(findings)

        # Extract PMF score
        pmf = scores.get("overall")
        if isinstance(pmf, (int, float)):
            pmf_scores.append(pmf)

        # Try to extract Ellis classification from findings
        for f in findings:
            f_lower = f.lower()
            if "very disappointed" in f_lower or "very_disappointed" in f_lower:
                ellis_counts["very_disappointed"] += 1
            elif "somewhat disappointed" in f_lower or "somewhat_disappointed" in f_lower:
                ellis_counts["somewhat_disappointed"] += 1
            elif "not disappointed" in f_lower or "not_disappointed" in f_lower:
                ellis_counts["not_disappointed"] += 1

            if "anti-pmf" in f_lower or "anti_pmf" in f_lower:
                anti_pmf.append(f)
            elif any(w in f_lower for w in ["signal", "buying", "workflow", "dependency"]):
                strongest_signals.append(f)

    total_calls = len(episodes)

    # Build dashboard markdown
    lines = [
        f"# PMF Dashboard — Generated from {total_calls} customer calls",
        "",
        f"*Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*",
        "",
    ]

    # Ellis Distribution
    if ellis_counts:
        total_classified = sum(ellis_counts.values())
        lines.append("## Ellis Distribution")
        lines.append("")
        for tier, label in [
            ("very_disappointed", "Very Disappointed (strong PMF)"),
            ("somewhat_disappointed", "Somewhat Disappointed"),
            ("not_disappointed", "Not Disappointed"),
        ]:
            count = ellis_counts.get(tier, 0)
            pct = round(count / total_classified * 100) if total_classified else 0
            marker = ""
            if tier == "very_disappointed":
                marker = f" — target: {int(ellis_threshold * 100)}%+"
                if pct >= ellis_threshold * 100:
                    marker += " ✅"
            lines.append(f"- {label}: {pct}% ({count}/{total_classified}){marker}")
        lines.append("")

    # PMF Score Trend
    if pmf_scores:
        avg_score = round(sum(pmf_scores) / len(pmf_scores), 1)
        lines.append("## PMF Score Trend")
        lines.append("")
        lines.append(f"- Average: {avg_score}/10 across {len(pmf_scores)} calls")

        # Simple trend: compare first half vs second half
        if len(pmf_scores) >= 4:
            mid = len(pmf_scores) // 2
            first_half = round(sum(pmf_scores[:mid]) / mid, 1)
            second_half = round(sum(pmf_scores[mid:]) / (len(pmf_scores) - mid), 1)
            trend = "improving" if second_half > first_half else "declining" if second_half < first_half else "stable"
            lines.append(f"- First half avg: {first_half}, second half avg: {second_half} → **{trend}**")
        lines.append("")

    # Top Findings
    finding_counts = Counter(all_findings)
    top = finding_counts.most_common(10)
    if top:
        lines.append("## Top Patterns")
        lines.append("")
        for finding, count in top:
            lines.append(f"- {finding} *(seen {count}x)*")
        lines.append("")

    # Anti-PMF patterns
    if anti_pmf:
        anti_counts = Counter(anti_pmf)
        lines.append("## Anti-PMF Flags")
        lines.append("")
        for flag, count in anti_counts.most_common(10):
            lines.append(f"- {flag} *(seen {count}x)*")
        lines.append("")

    md = "\n".join(lines)

    # Save dashboard
    dashboard_dir = Path(store_path) / "semantic" / "playbooks"
    dashboard_dir.mkdir(parents=True, exist_ok=True)
    dashboard_path = dashboard_dir / "pmf-dashboard.md"
    with open(dashboard_path, "w", encoding="utf-8") as f:
        f.write(md)

    logger.info("PMF dashboard written to %s", dashboard_path)
    return md
