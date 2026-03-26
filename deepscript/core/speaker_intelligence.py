"""Cross-call speaker intelligence — identify who speakers are across multiple calls.

Combines evidence from:
1. AudioScript voice embeddings (same voice = same person)
2. AudioScript LLM name extraction ("Chris addresses as 'Kim'")
3. Calendar event attendees
4. Contact list matching
5. Email thread participants
6. Cross-call topic analysis
7. Co-speaker patterns
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class SpeakerEvidence:
    """A piece of evidence for speaker identification."""

    source: str  # llm_extraction, calendar, contacts, email, topic_analysis, co_speaker
    name: str
    confidence: float
    detail: str = ""
    call_id: str = ""


@dataclass
class SpeakerProfile:
    """Cross-call profile for a voice cluster."""

    cluster_id: str
    likely_name: str | None = None
    name_confidence: float = 0.0
    evidence: list[SpeakerEvidence] = field(default_factory=list)
    role: str | None = None
    total_calls: int = 0
    total_speaking_seconds: float = 0.0
    topics: list[str] = field(default_factory=list)
    co_speakers: dict[str, int] = field(default_factory=dict)  # cluster_id → call count

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "likely_name": self.likely_name,
            "name_confidence": round(self.name_confidence, 3),
            "evidence": [
                {"source": e.source, "name": e.name, "confidence": e.confidence, "detail": e.detail}
                for e in self.evidence
            ],
            "role": self.role,
            "total_calls": self.total_calls,
            "total_speaking_seconds": round(self.total_speaking_seconds, 1),
            "topics": self.topics[:10],
            "co_speakers": dict(sorted(self.co_speakers.items(), key=lambda x: -x[1])[:5]),
        }


def identify_speakers(
    transcript_dir: str | Path,
    speaker_db_path: str | Path | None = None,
    calendar_provider: str = "none",
    contacts_provider: str = "none",
    llm: Any = None,
) -> dict[str, SpeakerProfile]:
    """Identify speakers across all transcripts in a directory.

    Returns: {cluster_id: SpeakerProfile}
    """
    transcript_dir = Path(transcript_dir)
    profiles: dict[str, SpeakerProfile] = {}

    # Load speaker identity DB
    speaker_db = _load_speaker_db(speaker_db_path) if speaker_db_path else {}

    # Load all transcripts and collect evidence
    transcripts = _load_transcripts(transcript_dir)
    logger.info("Loaded %d transcripts for speaker identification", len(transcripts))

    for file_path, transcript in transcripts:
        call_id = file_path.stem
        diar = transcript.get("diarization", {})
        resolved = diar.get("speakers_resolved", [])
        llm_analysis = transcript.get("llm_analysis", {})
        llm_speakers = llm_analysis.get("speakers", [])
        metadata = transcript.get("metadata", {})
        segments = transcript.get("segments", [])

        # Get all cluster IDs in this call
        call_clusters = set()
        for sr in resolved:
            cid = sr.get("speaker_cluster_id", "")
            if not cid:
                continue
            call_clusters.add(cid)

            if cid not in profiles:
                profiles[cid] = SpeakerProfile(cluster_id=cid)
            profiles[cid].total_calls += 1

            # Check if AudioScript already confirmed a name
            if sr.get("display_name"):
                profiles[cid].evidence.append(SpeakerEvidence(
                    source="audioscript_confirmed",
                    name=sr["display_name"],
                    confidence=0.95,
                    detail=f"Confirmed by AudioScript (status={sr.get('status')})",
                    call_id=call_id,
                ))

        # Evidence from AudioScript LLM name extraction
        for ls in llm_speakers:
            label = ls.get("label", "")
            likely_name = ls.get("likely_name")
            evidence_text = ls.get("evidence", "")
            role = ls.get("role", "")

            if not likely_name:
                continue

            # Match label to cluster_id
            cid = _match_label_to_cluster(label, resolved)
            if cid and cid in profiles:
                profiles[cid].evidence.append(SpeakerEvidence(
                    source="llm_extraction",
                    name=likely_name,
                    confidence=0.85,
                    detail=evidence_text[:200],
                    call_id=call_id,
                ))
                if role and not profiles[cid].role:
                    profiles[cid].role = role

        # Co-speaker patterns
        for cid in call_clusters:
            for other_cid in call_clusters:
                if cid != other_cid and cid in profiles:
                    profiles[cid].co_speakers[other_cid] = profiles[cid].co_speakers.get(other_cid, 0) + 1

        # Topic extraction from LLM analysis
        topics = llm_analysis.get("topics", [])
        title = llm_analysis.get("title", "")
        for cid in call_clusters:
            if cid in profiles:
                if title:
                    profiles[cid].topics.append(title)
                for t in topics:
                    if isinstance(t, str):
                        profiles[cid].topics.append(t)
                    elif isinstance(t, dict):
                        profiles[cid].topics.append(t.get("name", t.get("topic", str(t))))

    # Evidence from speaker DB (voice embedding history)
    for cid, profile in profiles.items():
        db_entry = speaker_db.get(cid, {})
        if db_entry:
            if db_entry.get("canonical_name"):
                profile.evidence.append(SpeakerEvidence(
                    source="speaker_db",
                    name=db_entry["canonical_name"],
                    confidence=0.90,
                    detail=f"From speaker identity DB (calls={db_entry.get('total_calls', 0)})",
                ))
            profile.total_speaking_seconds = db_entry.get("total_speaking_seconds", 0)

    # Calendar evidence
    if calendar_provider != "none":
        _add_calendar_evidence(profiles, transcripts, calendar_provider)

    # Contact list evidence
    if contacts_provider != "none":
        _add_contacts_evidence(profiles, contacts_provider)

    # Resolve best name for each profile
    for profile in profiles.values():
        _resolve_name(profile)

    # Deduplicate topics
    for profile in profiles.values():
        seen = set()
        unique = []
        for t in profile.topics:
            normalized = t.lower().strip()[:50]
            if normalized not in seen:
                seen.add(normalized)
                unique.append(t)
        profile.topics = unique[:20]

    return profiles


def _load_speaker_db(path: str | Path | None) -> dict[str, Any]:
    """Load AudioScript speaker identity database."""
    if not path:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            db = json.load(f)
        return db.get("identities", {})
    except Exception as e:
        logger.warning("Failed to load speaker DB: %s", e)
        return {}


def _load_transcripts(directory: Path) -> list[tuple[Path, dict]]:
    """Load all transcript JSON files from a directory."""
    results = []
    for fp in sorted(directory.glob("**/*.json")):
        if any(skip in fp.name for skip in ["embeddings", "manifest", "speaker_identities", ".audioscript"]):
            continue
        try:
            with open(fp) as f:
                data = json.load(f)
            if "text" in data or "segments" in data:
                results.append((fp, data))
        except (json.JSONDecodeError, OSError):
            continue
    return results


def _match_label_to_cluster(label: str, resolved: list[dict]) -> str | None:
    """Match an LLM speaker label to a cluster_id from diarization resolution."""
    # Direct match on local_label
    for sr in resolved:
        if sr.get("local_label") == label:
            return sr.get("speaker_cluster_id")
    # Match on cluster_id (label might already be a cluster_id)
    for sr in resolved:
        if sr.get("speaker_cluster_id") == label:
            return label
    # Match on display_name
    for sr in resolved:
        if sr.get("display_name") == label:
            return sr.get("speaker_cluster_id")
    return None


def _add_calendar_evidence(
    profiles: dict[str, SpeakerProfile],
    transcripts: list[tuple[Path, dict]],
    provider: str,
) -> None:
    """Add calendar attendee evidence to speaker profiles."""
    for file_path, transcript in transcripts:
        metadata = transcript.get("metadata", {})
        creation_time = metadata.get("audio", {}).get("creation_time")
        if not creation_time:
            continue

        # Look up calendar event
        try:
            if provider == "ms365":
                result = subprocess.run(
                    ["ms365", "calendar", "view", "--start", creation_time[:19], "--end", creation_time[:19], "-o", "json"],
                    capture_output=True, text=True, timeout=10,
                )
            elif provider == "google":
                result = subprocess.run(
                    ["gws", "calendar", "events", "list", "--params",
                     json.dumps({"timeMin": creation_time, "timeMax": creation_time, "maxResults": 3}),
                     "--format", "json"],
                    capture_output=True, text=True, timeout=10,
                )
            else:
                continue

            if result.returncode != 0:
                continue

            events = json.loads(result.stdout)
            if isinstance(events, dict):
                events = events.get("items", events.get("value", []))
            if not events:
                continue

            # Get attendee names from calendar event
            event = events[0] if isinstance(events, list) else events
            attendees = []
            for att in event.get("attendees", []):
                name = att.get("emailAddress", {}).get("name") or att.get("displayName") or att.get("email", "")
                if name:
                    attendees.append(name)

            # Match attendees to profiles by name similarity
            diar = transcript.get("diarization", {})
            resolved = diar.get("speakers_resolved", [])
            for cid, profile in profiles.items():
                # Check if any existing evidence name matches a calendar attendee
                for ev in profile.evidence:
                    for att_name in attendees:
                        if _names_match(ev.name, att_name):
                            profile.evidence.append(SpeakerEvidence(
                                source="calendar",
                                name=att_name,
                                confidence=0.80,
                                detail=f"Calendar event attendee: {event.get('subject', '')}",
                                call_id=file_path.stem,
                            ))
                            break

        except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
            continue


def _add_contacts_evidence(
    profiles: dict[str, SpeakerProfile],
    provider: str,
) -> None:
    """Match speaker names against Outlook/Google contacts."""
    try:
        if provider == "ms365":
            result = subprocess.run(
                ["ms365", "contacts", "list", "--top", "500", "-o", "json"],
                capture_output=True, text=True, timeout=15,
            )
        elif provider == "google":
            result = subprocess.run(
                ["gws", "people", "connections", "list", "--format", "json"],
                capture_output=True, text=True, timeout=15,
            )
        else:
            return

        if result.returncode != 0:
            return

        contacts = json.loads(result.stdout)
        if isinstance(contacts, dict):
            contacts = contacts.get("value", contacts.get("connections", []))

        # Build name lookup from contacts
        contact_names: dict[str, dict] = {}  # normalized_name → contact info
        for c in contacts:
            name = c.get("displayName") or ""
            if not name:
                # Try givenName + surname
                given = c.get("givenName", "")
                surname = c.get("surname", "")
                name = f"{given} {surname}".strip()
            if name:
                contact_names[name.lower()] = {
                    "name": name,
                    "company": c.get("companyName", ""),
                    "title": c.get("jobTitle", ""),
                    "email": "",
                }
                # Also index by first name only
                first = name.split()[0].lower()
                if first not in contact_names:
                    contact_names[first] = contact_names[name.lower()]

        # Match profiles against contacts
        for profile in profiles.values():
            for ev in profile.evidence:
                name_lower = ev.name.lower()
                first_name = name_lower.split()[0] if name_lower else ""

                # Full name match
                if name_lower in contact_names:
                    contact = contact_names[name_lower]
                    profile.evidence.append(SpeakerEvidence(
                        source="contacts",
                        name=contact["name"],
                        confidence=0.75,
                        detail=f"Contact: {contact['name']}, {contact.get('title', '')} at {contact.get('company', '')}",
                    ))
                    if contact.get("title") and not profile.role:
                        profile.role = contact["title"]
                # First name match
                elif first_name in contact_names:
                    contact = contact_names[first_name]
                    profile.evidence.append(SpeakerEvidence(
                        source="contacts_partial",
                        name=contact["name"],
                        confidence=0.60,
                        detail=f"Contact first-name match: {contact['name']}",
                    ))

    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        logger.warning("Contact list lookup failed: %s", e)


def _names_match(name1: str, name2: str) -> bool:
    """Check if two names likely refer to the same person."""
    n1 = name1.lower().strip()
    n2 = name2.lower().strip()
    if n1 == n2:
        return True
    # First name match
    if n1.split()[0] == n2.split()[0]:
        return True
    # One contains the other
    if n1 in n2 or n2 in n1:
        return True
    return False


def _resolve_name(profile: SpeakerProfile) -> None:
    """Pick the best name from all evidence, weighted by confidence."""
    if not profile.evidence:
        return

    # Group evidence by name
    name_scores: dict[str, float] = {}
    name_best_source: dict[str, str] = {}

    for ev in profile.evidence:
        if not ev.name:
            continue
        normalized = ev.name.lower().strip()
        name_scores[normalized] = name_scores.get(normalized, 0) + ev.confidence
        if normalized not in name_best_source or ev.confidence > 0.8:
            name_best_source[normalized] = ev.name  # Keep original casing

    if not name_scores:
        return

    # Pick highest scoring name
    best_normalized = max(name_scores, key=name_scores.get)
    best_name = name_best_source.get(best_normalized, best_normalized)
    best_score = name_scores[best_normalized]

    # Normalize confidence to 0-1 range (cap at 1.0)
    total_evidence = len([e for e in profile.evidence if e.name and e.name.lower().strip() == best_normalized])
    confidence = min(1.0, best_score / max(total_evidence, 1))

    profile.likely_name = best_name
    profile.name_confidence = round(confidence, 3)


def format_speaker_profiles(profiles: dict[str, SpeakerProfile]) -> str:
    """Format speaker profiles as markdown."""
    lines = [
        f"# Speaker Intelligence — {len(profiles)} Voice Clusters",
        "",
    ]

    named = sorted(
        [p for p in profiles.values() if p.likely_name],
        key=lambda p: -p.name_confidence,
    )
    unnamed = [p for p in profiles.values() if not p.likely_name]

    if named:
        lines.append(f"## Identified ({len(named)})")
        lines.append("")
        for p in named:
            lines.append(f"### {p.likely_name} ({p.cluster_id})")
            lines.append(f"**Confidence:** {p.name_confidence:.0%} | **Calls:** {p.total_calls} | **Role:** {p.role or 'unknown'}")
            lines.append("")
            if p.evidence:
                lines.append("Evidence:")
                seen = set()
                for e in sorted(p.evidence, key=lambda x: -x.confidence):
                    key = f"{e.source}:{e.name}"
                    if key in seen:
                        continue
                    seen.add(key)
                    lines.append(f"  - [{e.source}] {e.name} ({e.confidence:.0%}): {e.detail[:80]}")
                lines.append("")
            if p.topics:
                lines.append(f"Topics: {', '.join(p.topics[:5])}")
            if p.co_speakers:
                co = [f"{cid} ({count}x)" for cid, count in list(p.co_speakers.items())[:3]]
                lines.append(f"Co-speakers: {', '.join(co)}")
            lines.append("")

    if unnamed:
        lines.append(f"## Unidentified ({len(unnamed)})")
        lines.append("")
        for p in sorted(unnamed, key=lambda x: -x.total_calls):
            topics_str = f" — topics: {', '.join(p.topics[:3])}" if p.topics else ""
            lines.append(f"- {p.cluster_id}: {p.total_calls} calls{topics_str}")

    return "\n".join(lines)
