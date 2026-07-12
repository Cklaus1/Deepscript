"""MiNotes CRM page generator — creates person/company/interaction/task pages from DeepScript intelligence.

Each identified speaker gets a person page in CRM/people/.
Each unique company gets a company page in CRM/companies/.
Each call generates an interaction page in CRM/interactions/.
Action items are embedded as checkboxes on person pages and indexed in CRM/_Tasks.md.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from deepscript.core.speaker_intelligence import SpeakerProfile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_crm_pages(
    profiles: dict[str, SpeakerProfile],
    transcript_dir: str | Path,
    analysis_dir: str | Path | None = None,
    output_dir: str | Path = "CRM",
    speaker_db_path: str | Path | None = None,
    min_calls: int = 1,
    cms_store_path: str | Path | None = None,
    analysis_contexts: dict[str, Any] | None = None,
) -> dict[str, list[Path]]:
    """Generate the full CRM structure: people, companies, interactions, tasks.

    Returns dict mapping folder name to list of written paths.
    """
    transcript_dir = Path(transcript_dir)
    output_path = Path(output_dir)
    cms_path = Path(cms_store_path) if cms_store_path else None

    # Load speaker DB for name history
    speaker_db: dict[str, Any] = {}
    if speaker_db_path:
        try:
            with open(speaker_db_path) as f:
                speaker_db = json.load(f).get("identities", {})
        except Exception:
            pass

    # Collect call details
    call_details = _collect_call_details(transcript_dir, analysis_dir, cms_path, profiles, analysis_contexts)

    # Build company index
    company_people: dict[str, list[str]] = defaultdict(list)
    for cid, profile in profiles.items():
        if profile.company:
            company_people[profile.company].append(cid)

    written: dict[str, list[Path]] = {
        "people": [],
        "companies": [],
        "interactions": [],
    }

    # Write person pages — disambiguate collisions via cluster_id suffix
    person_sanitized = Counter()
    for cid, profile in profiles.items():
        if not profile.likely_name:
            continue
        if profile.total_calls < min_calls:
            continue
        person_sanitized[_sanitize(profile.likely_name)] += 1
    for cid, profile in profiles.items():
        if not profile.likely_name:
            continue
        if profile.total_calls < min_calls:
            continue
        calls = call_details.get(cid, [])
        page = _render_person_page(profile, calls, speaker_db.get(cid, {}), profiles)
        safe_name = _sanitize(profile.likely_name)
        if person_sanitized[safe_name] > 1:
            safe_name = f"{safe_name}-{cid[:6]}"
        fp = output_path / "people" / f"{safe_name}.md"
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(page, encoding="utf-8")
        written["people"].append(fp)

    # Write company pages — disambiguate collisions via company string hash
    company_sanitized = Counter()
    for company in company_people:
        company_sanitized[_sanitize(company)] += 1
    for company, cids in company_people.items():
        people_pages = []
        for cid in cids:
            p = profiles.get(cid)
            if p and p.likely_name:
                people_pages.append(_sanitize(p.likely_name))
        page = _render_company_page(company, people_pages)
        safe_company = _sanitize(company)
        if company_sanitized[safe_company] > 1:
            safe_company = f"{safe_company}-{hashlib.sha1(company.encode('utf-8')).hexdigest()[:6]}"
        fp = output_path / "companies" / f"{safe_company}.md"
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(page, encoding="utf-8")
        written["companies"].append(fp)

    # Write interaction pages — one per unique call, not per (speaker, call).
    # call_details is keyed by speaker cluster_id, so the same call appears once
    # per participant; dedup by the call's source file to avoid redundant writes
    # (and an inflated count) since interaction pages are call-level, not per-speaker.
    seen_calls: set[str] = set()
    unique_calls: list[dict[str, Any]] = []
    for call in call_details.values():
        if isinstance(call, list):
            for c in call:
                call_key = c.get("file") or f"{c.get('date', '')}-{c.get('title', '')}"
                if call_key in seen_calls:
                    continue
                seen_calls.add(call_key)
                unique_calls.append(c)

    # Detect distinct calls whose sanitized {date}-{title} collapse to the same
    # filename (e.g. two "Fusen UX" calls on the same day). Without this they
    # silently overwrite each other on disk. Disambiguate only the colliders.
    base_counts = Counter(_interaction_basename(c) for c in unique_calls)
    for c in unique_calls:
        disambiguate = base_counts[_interaction_basename(c)] > 1
        fp = _write_interaction_page(
            c, output_path, analysis_contexts, disambiguate=disambiguate
        )
        if fp:
            written["interactions"].append(fp)

    # Write global task index
    _write_task_index(call_details, output_path, profiles)

    total = sum(len(v) for v in written.values())
    logger.info("Generated %d CRM pages (%d people, %d companies, %d interactions, tasks indexed)",
                total, len(written["people"]), len(written["companies"]), len(written["interactions"]))
    return written


# ---------------------------------------------------------------------------
# Call detail collection
# ---------------------------------------------------------------------------

def _collect_call_details(
    transcript_dir: Path,
    analysis_dir: Path | None,
    cms_path: Path | None,
    profiles: dict[str, SpeakerProfile],
    analysis_contexts: dict[str, Any] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Collect per-speaker call details from transcripts, analyses, and CMS episodes."""

    speaker_calls: dict[str, list[dict[str, Any]]] = defaultdict(list)

    # Load analyses
    analyses: dict[str, dict] = {}
    if analysis_dir:
        for f in Path(analysis_dir).glob("*.analysis.json"):
            try:
                with open(f) as fh:
                    analyses[f.stem.replace(".analysis", "")] = json.load(fh)
            except Exception:
                continue

    # Load CMS episodes
    cms_episodes: dict[str, dict] = {}
    if cms_path and cms_path.exists():
        for f in cms_path.glob("*.jsonl"):
            try:
                with open(f) as fh:
                    for line in fh:
                        ep = json.loads(line)
                        cms_episodes[ep.get("context", {}).get("task_id", "")] = ep
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

        creation_time = (
            metadata.get("audio", {}).get("format_tags", {}).get("creation_time", "")
            or metadata.get("audio", {}).get("creation_time", "")
        )
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

        action_items = analysis.get("analysis", {}).get("action_items", [])

        # CMS episode data
        ep = cms_episodes.get(call_name, {})
        cms_data = {
            "episode_id": ep.get("episode_id", ""),
            "scores": ep.get("outcome", {}).get("scores", {}),
            "findings": ep.get("outcome", {}).get("findings", []),
        }

        for sr in resolved:
            cid = sr.get("speaker_cluster_id", "") or sr.get("local_label", "")
            if cid not in profiles:
                continue

            speaker_name = (profiles[cid].likely_name or "").lower()
            my_actions = []
            for ai in action_items:
                if isinstance(ai, dict):
                    assignee = (ai.get("assignee") or ai.get("speaker") or "").lower()
                    if not speaker_name:
                        continue
                    # Exact or prefix-of-tokens match to avoid "Chris" claiming
                    # "Chris Klaus" AND "Chris Erven"
                    if assignee == speaker_name:
                        my_actions.append(ai.get("text", ai.get("action", str(ai))))
                        continue
                    s_tokens = speaker_name.split()
                    a_tokens = assignee.split()
                    if s_tokens and a_tokens[:len(s_tokens)] == s_tokens:
                        my_actions.append(ai.get("text", ai.get("action", str(ai))))

            speaker_calls[cid].append({
                "file": call_name,
                "date": creation_time[:10] if creation_time else "",
                "title": title,
                "type": call_type,
                "duration_min": round(duration / 60, 1) if duration else 0,
                "summary": summary[:500],
                "action_items": my_actions,
                "cms": cms_data,
            })

    return dict(speaker_calls)


# ---------------------------------------------------------------------------
# Person page
# ---------------------------------------------------------------------------

def _render_person_page(
    profile: SpeakerProfile,
    calls: list[dict[str, Any]],
    db_entry: dict[str, Any],
    all_profiles: dict[str, SpeakerProfile],
) -> str:
    lines: list[str] = []

    # Frontmatter
    lines.append("---")
    lines.append("type: person")
    lines.append(f"cluster_id: {profile.cluster_id}")
    lines.append(f"name: {json.dumps(profile.likely_name)}")
    lines.append(f"display_name: {json.dumps(profile.display_name)}")

    full_name = profile.best_full_name
    if full_name and full_name != profile.likely_name:
        lines.append(f"full_name: {json.dumps(full_name)}")

    lines.append(f"confidence: {profile.name_confidence:.2f}")

    # Contact info — parse early so company/email/phone/title are available
    # for frontmatter below
    contact_email = ""
    contact_phone = ""
    contact_title = ""
    contact_company = ""
    for ev in profile.evidence:
        if ev.source == "contacts" and ev.detail:
            detail = ev.detail
            if "(" in detail and "@" in detail:
                contact_email = detail.split("(")[-1].rstrip(")")
            if " at " in detail:
                parts = detail.split(" at ")
                if len(parts) >= 2:
                    contact_company = parts[-1].split("(")[0].strip().rstrip(",")
                after_name = detail.split(",", 1)
                if len(after_name) >= 2:
                    contact_title = after_name[1].split(" at ")[0].strip().rstrip(",")
            elif ", " in detail:
                after_name = detail.split(",", 1)
                if len(after_name) >= 2:
                    contact_title = after_name[1].strip().rstrip(",")
            break

    if profile.role:
        lines.append(f"role: {json.dumps(profile.role)}")
    if profile.company:
        lines.append(f"company: {json.dumps(profile.company)}")
    elif contact_company:
        lines.append(f"company: {json.dumps(contact_company)}")

    lines.append(f"total_calls: {profile.total_calls}")

    if profile.total_speaking_seconds:
        hours = profile.total_speaking_seconds / 3600
        lines.append(f"speaking_hours: {hours:.1f}")

    lines.append(f"email: {json.dumps(contact_email)}")
    lines.append(f"phone: {json.dumps(contact_phone)}")
    lines.append(f"title: {json.dumps(contact_title)}")
    lines.append(f"linkedin: \"\"")

    if calls:
        dates = [c["date"] for c in calls if c["date"]]
        if dates:
            lines.append(f"first_contact: \"{min(dates)}\"")
            lines.append(f"last_contact: \"{max(dates)}\"")

    aliases = db_entry.get("aliases", [])
    if aliases:
        lines.append(f"aliases: {json.dumps(aliases)}")

    tags = [f"speaker/{profile.likely_name.lower().replace(' ', '-')}"]
    if profile.role:
        tags.append(f"role/{profile.role.lower().replace(' ', '-')[:30]}")
    lines.append(f"tags: {json.dumps(tags)}")
    lines.append("---")
    lines.append("")

    # Page content
    lines.append(f"# {profile.display_name}")
    lines.append("")
    if profile.role:
        lines.append(f"**Role:** {profile.role}")
    if profile.company:
        lines.append(f"**Company:** {profile.company}")
    lines.append(f"**Calls:** {profile.total_calls} | **Confidence:** {profile.name_confidence:.0%}")
    if full_name and full_name != profile.likely_name:
        lines.append(f"**Full name:** {full_name}")
    lines.append("")

    # Identification Evidence
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

    # Call History
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

    # Topics
    if profile.topics:
        lines.append("## Topics Discussed")
        lines.append("")
        for t in profile.topics[:15]:
            lines.append(f"- {t}")
        lines.append("")

    # Action Items (embedded checkboxes)
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

    # Relationship Map
    if profile.co_speakers:
        lines.append("## Usually With")
        lines.append("")
        for co_cid, count in sorted(profile.co_speakers.items(), key=lambda x: -x[1])[:8]:
            co_p = all_profiles.get(co_cid)
            co_name = co_p.display_name if co_p else co_cid[:12]
            co_role = f" ({co_p.role})" if co_p and co_p.role else ""
            lines.append(f"- [[{co_name}]]{co_role} — {count} calls together")
        lines.append("")

    # Name History
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


# ---------------------------------------------------------------------------
# Company page
# ---------------------------------------------------------------------------

def _render_company_page(company: str, people: list[str]) -> str:
    lines = [
        "---",
        "type: company",
        f"name: \"{company}\"",
        "---",
        "",
        f"# {company}",
        "",
        f"**People:** {len(people)}",
        "",
        "## People",
        "",
    ]
    for p in people:
        lines.append(f"- [[{p}]]")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Interaction page
# ---------------------------------------------------------------------------

def _interaction_basename(call: dict[str, Any]) -> str:
    """Sanitized {date}-{title} stem for a call's interaction page (no suffix)."""
    date = call.get("date", "unknown")
    safe_date = date.replace("-", "") if date else "unknown"
    safe_title = _sanitize(call.get("title", "call"))[:40]
    return f"{safe_date}-{safe_title}"


def _write_interaction_page(
    call: dict[str, Any],
    output_path: Path,
    analysis_contexts: dict[str, Any] | None = None,
    disambiguate: bool = False,
) -> Path | None:
    basename = _interaction_basename(call)
    # When multiple distinct calls share a basename (same day + title prefix),
    # append a short stable hash of the source file so neither overwrites the other.
    if disambiguate:
        seed = call.get("file") or f"{call.get('summary', '')[:80]}"
        suffix = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:6]
        basename = f"{basename}-{suffix}"
    filename = f"{basename}.md"
    fp = output_path / "interactions" / filename
    fp.parent.mkdir(parents=True, exist_ok=True)

    cms = call.get("cms", {})
    lines = [
        "---",
        "type: interaction",
        f"title: {json.dumps(call.get('title', ''))}",
        f"date: {json.dumps(call.get('date', ''))}",
        f"call_type: {json.dumps(call.get('type', ''))}",
        f"duration_min: {call.get('duration_min', 0)}",
        f"file: {json.dumps(call.get('file', ''))}",
    ]
    if cms.get("episode_id"):
        lines.append(f"cms_episode_id: {json.dumps(cms['episode_id'])}")
    lines.append("---")
    lines.append("")
    lines.append(f"# {call.get('title', 'Call')}")
    lines.append("")
    lines.append(f"**Date:** {call.get('date', '')}  ")
    lines.append(f"**Type:** {call.get('type', '')}  ")
    lines.append(f"**Duration:** {call.get('duration_min', 0):.0f} min")
    lines.append("")

    summary = call.get("summary", "")
    if summary:
        lines.append("## Summary")
        lines.append("")
        lines.append(summary)
        lines.append("")

    if cms.get("findings"):
        lines.append("## CMS Findings")
        lines.append("")
        for f in cms["findings"]:
            lines.append(f"- {f}")
        lines.append("")

    if cms.get("scores"):
        lines.append("## Scores")
        lines.append("")
        for k, v in cms["scores"].items():
            lines.append(f"- **{k}:** {v}")
        lines.append("")

    # CMS episode references
    if analysis_contexts and cms.get("episode_id"):
        ep_id = cms["episode_id"]
        episodes = analysis_contexts.get("cms_episodes", {})
        for ep in episodes.get("episodes", []):
            if ep.get("episode_id") == ep_id:
                lines.append("## CMS Episode")
                lines.append("")
                lines.append(f"- **Episode:** [{ep.get('title', ep_id)}]({ep.get('url', '#')})")
                lines.append(f"- **Season/Episode:** S{ep.get('season', '?')}:E{ep.get('episode_num', '?')}")
                lines.append(f"- **Air Date:** {ep.get('air_date', 'TBD')}")
                lines.append(f"- **Duration:** {ep.get('duration_sec', 0) // 60} min")
                lines.append("")
                if ep.get("description"):
                    lines.append(ep["description"])
                    lines.append("")
                break

    fp.write_text("\n".join(lines), encoding="utf-8")
    return fp


# ---------------------------------------------------------------------------
# Task index
# ---------------------------------------------------------------------------

def _write_task_index(
    call_details: dict[str, list[dict[str, Any]]],
    output_path: Path,
    profiles: dict[str, SpeakerProfile],
) -> None:
    all_tasks: list[dict[str, str]] = []

    for cid, calls in call_details.items():
        for c in calls:
            for ai in c.get("action_items", []):
                all_tasks.append({
                    "text": ai,
                    "assignee": cid,
                    "date": c.get("date", ""),
                    "call": c.get("title", ""),
                })

    lines = [
        "---",
        "type: task_index",
        "---",
        "",
        "# Tasks",
        "",
        f"**Total:** {len(all_tasks)}",
        "",
        "| Task | Assignee | Date | Call |",
        "|------|----------|------|------|",
    ]

    for t in sorted(all_tasks, key=lambda x: x.get("date", "")):
        assignee_cid = t["assignee"]
        profile = profiles.get(assignee_cid)
        assignee_display = profile.display_name if profile else assignee_cid
        # Escape pipes in free-form text and call titles
        text = str(t["text"]).replace("|", "\\|")
        call = str(t["call"]).replace("|", "\\|")
        lines.append(f"| {text} | {assignee_display} | {t['date']} | {call} |")

    lines.append("")
    fp = output_path / "_Tasks.md"
    fp.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize(name: str) -> str:
    return re.sub(r"[^\w \-]", "", name, flags=re.UNICODE).strip() or "untitled"


# ---------------------------------------------------------------------------
# Legacy compatibility — keep generate_contact_pages for existing callers
# ---------------------------------------------------------------------------

def generate_contact_pages(
    profiles: dict[str, SpeakerProfile],
    transcript_dir: str | Path,
    analysis_dir: str | Path | None = None,
    output_dir: str | Path = "CRM/Contacts",
    speaker_db_path: str | Path | None = None,
    min_calls: int = 2,
) -> list[Path]:
    """Legacy wrapper — delegates to generate_crm_pages for compatibility."""
    written = generate_crm_pages(
        profiles=profiles,
        transcript_dir=transcript_dir,
        analysis_dir=analysis_dir,
        output_dir=output_dir,
        speaker_db_path=speaker_db_path,
        min_calls=min_calls,
    )
    return [p for v in written.values() for p in v]


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