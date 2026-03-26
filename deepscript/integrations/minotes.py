"""MiNotes contact page generator — creates speaker profile pages from DeepScript intelligence.

Each identified speaker gets a MiNotes page with:
- YAML frontmatter (cluster_id, name, email, phone, company, role, calls, confidence)
- Call history with dates, types, summaries
- Topics discussed across all calls
- Action items assigned to this person
- Relationship map (co-speakers)
- Name history (how identification evolved)
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from deepscript.core.speaker_intelligence import SpeakerProfile

logger = logging.getLogger(__name__)


def generate_contact_pages(
    profiles: dict[str, SpeakerProfile],
    transcript_dir: str | Path,
    analysis_dir: str | Path | None = None,
    output_dir: str | Path = "CRM/Contacts",
    speaker_db_path: str | Path | None = None,
    min_calls: int = 2,
) -> list[Path]:
    """Generate MiNotes contact pages for identified speakers.

    Args:
        profiles: Speaker profiles from identify_speakers().
        transcript_dir: Directory with AudioScript transcripts.
        analysis_dir: Directory with DeepScript analysis JSON files.
        output_dir: Where to write MiNotes pages.
        speaker_db_path: Path to speaker_identities.json for name history.
        min_calls: Minimum calls to generate a page (skip one-off speakers).

    Returns: List of written file paths.
    """
    transcript_dir = Path(transcript_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load speaker DB for name history
    speaker_db = {}
    if speaker_db_path:
        try:
            with open(speaker_db_path) as f:
                speaker_db = json.load(f).get("identities", {})
        except Exception:
            pass

    # Load transcripts for call details
    call_details = _collect_call_details(transcript_dir, analysis_dir, profiles)

    written: list[Path] = []

    for cid, profile in profiles.items():
        if not profile.likely_name:
            continue
        if profile.total_calls < min_calls:
            continue

        page = _render_contact_page(profile, call_details.get(cid, []), speaker_db.get(cid, {}), profiles)

        # Filename: sanitize name
        import re
        safe_name = re.sub(r"[^a-zA-Z0-9 _\-]", "", profile.likely_name).strip()
        if not safe_name:
            safe_name = cid
        file_path = output_path / f"{safe_name}.md"

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(page)
        written.append(file_path)

    logger.info("Generated %d contact pages in %s", len(written), output_path)
    return written


def _collect_call_details(
    transcript_dir: Path,
    analysis_dir: Path | None,
    profiles: dict[str, SpeakerProfile],
) -> dict[str, list[dict[str, Any]]]:
    """Collect per-speaker call details from transcripts and analyses."""

    speaker_calls: dict[str, list[dict[str, Any]]] = defaultdict(list)

    # Load analyses if available
    analyses: dict[str, dict] = {}
    if analysis_dir:
        analysis_path = Path(analysis_dir)
        for f in analysis_path.glob("*.analysis.json"):
            try:
                with open(f) as fh:
                    analyses[f.stem.replace(".analysis", "")] = json.load(fh)
            except Exception:
                continue

    # Walk transcripts
    for fp in sorted(transcript_dir.glob("**/*.json")):
        if any(skip in fp.name for skip in ["embeddings", "manifest", "speaker_identities", ".audioscript"]):
            continue
        try:
            with open(fp) as f:
                transcript = json.load(f)
        except Exception:
            continue

        call_name = fp.stem
        diar = transcript.get("diarization", {})
        resolved = diar.get("speakers_resolved", [])
        llm = transcript.get("llm_analysis", {})
        metadata = transcript.get("metadata", {})
        analysis = analyses.get(call_name, {})

        # Get call metadata
        creation_time = metadata.get("audio", {}).get("creation_time", "")
        duration = metadata.get("audio", {}).get("duration_seconds", 0)
        title = llm.get("title", "")
        call_type = analysis.get("classification", {}).get("call_type", "unknown")
        summary_data = analysis.get("analysis", {}).get("summary", {})
        if isinstance(summary_data, dict):
            summary = summary_data.get("text", "")
        elif isinstance(summary_data, str):
            summary = summary_data
        else:
            summary = ""

        # Action items
        action_items = analysis.get("analysis", {}).get("action_items", [])

        # Map speakers to this call
        for sr in resolved:
            cid = sr.get("speaker_cluster_id", "")
            if cid not in profiles:
                continue

            # Find action items assigned to this speaker
            speaker_name = (profiles[cid].likely_name or "").lower()
            my_actions = []
            for ai in action_items:
                if isinstance(ai, dict):
                    assignee = (ai.get("assignee") or ai.get("speaker") or "").lower()
                    if speaker_name and speaker_name in assignee:
                        my_actions.append(ai.get("text", ai.get("action", str(ai))))

            speaker_calls[cid].append({
                "file": call_name,
                "date": creation_time[:10] if creation_time else "",
                "title": title,
                "type": call_type,
                "duration_min": round(duration / 60, 1) if duration else 0,
                "summary": summary[:200],
                "action_items": my_actions,
            })

    return dict(speaker_calls)


def _render_contact_page(
    profile: SpeakerProfile,
    calls: list[dict[str, Any]],
    db_entry: dict[str, Any],
    all_profiles: dict[str, SpeakerProfile],
) -> str:
    """Render a MiNotes contact page as markdown with YAML frontmatter."""
    lines: list[str] = []

    # --- YAML Frontmatter ---
    lines.append("---")
    lines.append("type: contact")
    lines.append(f"cluster_id: {profile.cluster_id}")
    lines.append(f"name: \"{profile.likely_name}\"")
    lines.append(f"display_name: \"{profile.display_name}\"")

    full_name = profile.best_full_name
    if full_name and full_name != profile.likely_name:
        lines.append(f"full_name: \"{full_name}\"")

    lines.append(f"confidence: {profile.name_confidence:.2f}")

    if profile.role:
        lines.append(f"role: \"{profile.role}\"")

    lines.append(f"total_calls: {profile.total_calls}")

    if profile.total_speaking_seconds:
        hours = profile.total_speaking_seconds / 3600
        lines.append(f"speaking_hours: {hours:.1f}")

    # Contact info placeholders (populated when ms365 contacts connected)
    lines.append("email: \"\"")
    lines.append("phone: \"\"")
    lines.append("company: \"\"")
    lines.append("title: \"\"")
    lines.append("linkedin: \"\"")

    if calls:
        dates = [c["date"] for c in calls if c["date"]]
        if dates:
            lines.append(f"first_contact: \"{min(dates)}\"")
            lines.append(f"last_contact: \"{max(dates)}\"")

    # Aliases
    aliases = db_entry.get("aliases", [])
    if aliases:
        lines.append(f"aliases: {json.dumps(aliases)}")

    # Tags
    tags = [f"speaker/{profile.likely_name.lower().replace(' ', '-')}"]
    if profile.role:
        tags.append(f"role/{profile.role.lower().replace(' ', '-')[:30]}")
    lines.append(f"tags: {json.dumps(tags)}")

    lines.append("---")
    lines.append("")

    # --- Page Content ---
    lines.append(f"# {profile.display_name}")
    lines.append("")

    if profile.role:
        lines.append(f"**Role:** {profile.role}")
    lines.append(f"**Calls:** {profile.total_calls} | **Confidence:** {profile.name_confidence:.0%}")
    if full_name and full_name != profile.likely_name:
        lines.append(f"**Full name:** {full_name}")
    lines.append("")

    # --- Identification Evidence ---
    if profile.evidence:
        lines.append("## Identification Evidence")
        lines.append("")
        seen = set()
        for e in sorted(profile.evidence, key=lambda x: -x.confidence):
            key = f"{e.source}:{e.name}"
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"- [{e.source}] **{e.name}** ({e.confidence:.0%}): {e.detail[:80]}")
        lines.append("")

    # --- Call History ---
    if calls:
        lines.append("## Call History")
        lines.append("")
        lines.append("| Date | Type | Title | Duration |")
        lines.append("|------|------|-------|----------|")
        for c in sorted(calls, key=lambda x: x.get("date", ""), reverse=True):
            date = c.get("date", "?")
            ctype = c.get("type", "?")
            title = (c.get("title") or "")[:50]
            dur = f"{c.get('duration_min', 0):.0f} min"
            lines.append(f"| {date} | {ctype} | {title} | {dur} |")
        lines.append("")

    # --- Topics Discussed ---
    if profile.topics:
        lines.append("## Topics Discussed")
        lines.append("")
        for t in profile.topics[:15]:
            lines.append(f"- {t}")
        lines.append("")

    # --- Action Items ---
    all_actions = []
    for c in calls:
        for ai in c.get("action_items", []):
            all_actions.append({"text": ai, "date": c.get("date", ""), "call": c.get("title", "")})

    if all_actions:
        lines.append(f"## Action Items ({len(all_actions)})")
        lines.append("")
        for ai in all_actions:
            date_str = f" ({ai['date']})" if ai["date"] else ""
            lines.append(f"- [ ] {ai['text']}{date_str}")
        lines.append("")

    # --- Relationship Map ---
    if profile.co_speakers:
        lines.append("## Usually With")
        lines.append("")
        for co_cid, count in sorted(profile.co_speakers.items(), key=lambda x: -x[1])[:8]:
            co_p = all_profiles.get(co_cid)
            co_name = co_p.display_name if co_p else co_cid[:12]
            co_role = f" ({co_p.role})" if co_p and co_p.role else ""
            lines.append(f"- [[{co_name}]]{co_role} — {count} calls together")
        lines.append("")

    # --- Name History ---
    name_history = db_entry.get("name_history", [])
    if name_history:
        lines.append("## Name History")
        lines.append("")
        lines.append("| Date | Name | Confidence | Sources |")
        lines.append("|------|------|-----------|---------|")
        for h in name_history:
            date = h.get("timestamp", "")[:10]
            name = h.get("name", "")
            conf = h.get("confidence", 0)
            sources = ", ".join(h.get("sources", []))
            lines.append(f"| {date} | {name} | {conf:.0%} | {sources} |")
        lines.append("")

    return "\n".join(lines)


def generate_contacts_summary(
    profiles: dict[str, SpeakerProfile],
    output_dir: str | Path = "CRM/Contacts",
) -> str:
    """Generate a summary index page for all contacts."""
    named = sorted(
        [p for p in profiles.values() if p.likely_name],
        key=lambda p: -p.total_calls,
    )

    lines = [
        "---",
        "type: index",
        "title: Contacts",
        "---",
        "",
        f"# Contacts — {len(named)} People Identified",
        "",
        "| Name | Calls | Role | Confidence | Last Contact |",
        "|------|-------|------|-----------|-------------|",
    ]

    for p in named:
        role = p.role or ""
        lines.append(f"| [[{p.display_name}]] | {p.total_calls} | {role} | {p.name_confidence:.0%} | |")

    return "\n".join(lines)
