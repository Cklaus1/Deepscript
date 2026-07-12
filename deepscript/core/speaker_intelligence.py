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
    company: str | None = None
    total_calls: int = 0
    total_speaking_seconds: float = 0.0
    topics: list[str] = field(default_factory=list)
    co_speakers: dict[str, int] = field(default_factory=dict)  # cluster_id → call count

    @property
    def display_name(self) -> str:
        """Progressive display name based on confidence.

        ≥0.90 — "Adam Schwartz" (full name, no qualifier)
        ≥0.80 — "Adam Schwartz" (confident enough, no qualifier)
        ≥0.60 — "Adam (likely)"
        ≥0.40 — "Adam (possible)"
        <0.40 — cluster_id
        """
        if not self.likely_name:
            return self.cluster_id

        if self.name_confidence >= 0.80:
            return self.likely_name
        elif self.name_confidence >= 0.60:
            return f"{self.likely_name} (likely)"
        elif self.name_confidence >= 0.40:
            return f"{self.likely_name} (possible)"
        else:
            return self.cluster_id

    @property
    def best_full_name(self) -> str | None:
        """Return the most complete name available (prefer full names from contacts/calendar)."""
        # Prefer names with spaces (full names) from high-confidence sources
        full_names = [
            e.name for e in self.evidence
            if " " in e.name and e.confidence >= 0.70
            and e.source in ("contacts", "calendar", "audioscript_confirmed", "speaker_db")
        ]
        if full_names:
            # Pick the longest (most complete) name
            return max(full_names, key=len)

        # Fall back to any full name
        full_names = [e.name for e in self.evidence if " " in e.name and e.confidence >= 0.60]
        if full_names:
            return max(full_names, key=len)

        return self.likely_name

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "likely_name": self.likely_name,
            "display_name": self.display_name,
            "best_full_name": self.best_full_name,
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
    cached_calendar_events: list[dict] | None = None,
    cached_contacts: list[dict] | None = None,
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

    # Apply name corrections (fixes Whisper mistranscriptions like "a news" → "Anuj")
    try:
        from deepscript.core.name_corrections import load_corrections, apply_corrections_to_segments
        corrections = load_corrections(
            transcript_dir=transcript_dir,
            speaker_db_path=speaker_db_path,
            contacts=cached_contacts,
        )
        if corrections:
            total_fixed = 0
            for _, transcript in transcripts:
                total_fixed += apply_corrections_to_segments(
                    transcript.get("segments", []), corrections
                )
            if total_fixed:
                logger.info("Name corrections: %d segments fixed (%d correction rules)",
                            total_fixed, len(corrections))
    except ImportError:
        pass

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
            cid = sr.get("speaker_cluster_id") or f"{call_id}:{sr.get('local_label', '')}"
            if not cid:
                continue
            if cid in call_clusters:
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
        # With word-level probability validation (#9)
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
                # Check word-level probability for the name if segments available
                name_prob = _check_name_word_probability(likely_name, segments)
                # Down-weight names with very low Whisper confidence
                conf = 0.85
                if name_prob is not None and name_prob < 0.30:
                    conf = 0.65  # Low ASR confidence — might be hallucinated
                elif name_prob is not None and name_prob > 0.80:
                    conf = 0.90  # High ASR confidence — strong signal

                profiles[cid].evidence.append(SpeakerEvidence(
                    source="llm_extraction",
                    name=likely_name,
                    confidence=conf,
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

    # Calendar evidence (includes RSVP status when available)
    if calendar_provider != "none" or cached_calendar_events is not None:
        _add_calendar_evidence(profiles, transcripts, calendar_provider, cached_events=cached_calendar_events)

    # Contact list evidence
    if contacts_provider != "none" or cached_contacts is not None:
        _add_contacts_evidence(profiles, contacts_provider, cached_contacts=cached_contacts)

    # Account owner detection (used by multiple signals below)
    _owner_first = ""
    _owner_full = ""

    # --- Advanced signals (speaker_signals module) ---
    try:
        from deepscript.core.speaker_signals import (
            extract_cross_speaker_references,
            compute_speaker_role_scores,
            match_roles_to_contacts,
            filter_low_confidence_segments,
            compute_name_frequencies,
            extract_company_mentions,
        )

        # 6. Segment confidence filtering — apply to all transcripts
        for file_path, transcript in transcripts:
            segments = transcript.get("segments", [])
            if segments:
                transcript["_filtered_segments"] = filter_low_confidence_segments(segments)

        # 1. Cross-speaker name references
        # Detect account owner to reduce noise (they're addressed in every call)
        if cached_calendar_events and not _owner_first:
            _owner_first, _ = _detect_account_owner(cached_calendar_events)

        cross_ref_count = 0
        for file_path, transcript in transcripts:
            segments = transcript.get("_filtered_segments", transcript.get("segments", []))
            refs = extract_cross_speaker_references(segments)
            for ref in refs:
                cid = ref["addressed_cluster"]
                # Skip if this is the account owner's first name (too noisy)
                if _owner_first and ref["name"].lower() == _owner_first:
                    continue
                if cid in profiles:
                    already = any(
                        e.source == "cross_speaker_ref" and e.name == ref["name"]
                        and e.call_id == file_path.stem
                        for e in profiles[cid].evidence
                    )
                    if not already:
                        profiles[cid].evidence.append(SpeakerEvidence(
                            source="cross_speaker_ref",
                            name=ref["name"],
                            confidence=ref["confidence"],
                            detail=f"Addressed by {ref['speaker_cluster'][:12]}: \"{ref['text'][:60]}\"",
                            call_id=file_path.stem,
                        ))
                        cross_ref_count += 1
        if cross_ref_count:
            logger.info("Cross-speaker references: %d evidence entries added", cross_ref_count)

        # 8. Name frequency boosting
        # Skip account owner's name — mentioned in every call, adds noise
        freq_boost_count = 0
        for file_path, transcript in transcripts:
            segments = transcript.get("_filtered_segments", transcript.get("segments", []))
            name_freqs = compute_name_frequencies(segments)
            for cid_speaker, freq_map in name_freqs.items():
                for name, count in freq_map.items():
                    if count < 3:
                        continue
                    # Skip account owner's first name
                    if _owner_first and name.lower() == _owner_first:
                        continue
                    for cid_target, profile in profiles.items():
                        if cid_target == cid_speaker:
                            continue
                        if any(e.name.lower().startswith(name.lower()) for e in profile.evidence):
                            already = any(
                                e.source == "name_frequency" and e.name == name
                                for e in profile.evidence
                            )
                            if not already:
                                boost = min(0.15, count * 0.02)
                                profile.evidence.append(SpeakerEvidence(
                                    source="name_frequency",
                                    name=name,
                                    confidence=0.60 + boost,
                                    detail=f"Name mentioned {count}x across calls",
                                ))
                                freq_boost_count += 1
        if freq_boost_count:
            logger.info("Name frequency boosting: %d entries", freq_boost_count)

        # 4. Role-to-speaker correlation (needs contacts)
        # Only add one role match per cluster (strongest) to avoid flooding
        if cached_contacts:
            all_role_evidence: dict[str, dict] = {}  # cid → best evidence
            for file_path, transcript in transcripts:
                segments = transcript.get("_filtered_segments", transcript.get("segments", []))
                role_scores = compute_speaker_role_scores(segments)
                if role_scores:
                    role_evidence = match_roles_to_contacts(role_scores, cached_contacts)
                    for ev in role_evidence:
                        cid = ev["cluster_id"]
                        if cid not in all_role_evidence or ev["confidence"] > all_role_evidence[cid]["confidence"]:
                            all_role_evidence[cid] = ev

            role_count = 0
            for cid, ev in all_role_evidence.items():
                if cid in profiles:
                    profiles[cid].evidence.append(SpeakerEvidence(
                        source="role_correlation",
                        name=ev["name"],
                        confidence=ev["confidence"],
                        detail=ev["detail"][:100],
                    ))
                    role_count += 1
            if role_count:
                logger.info("Role-to-speaker correlation: %d entries", role_count)

        # 7. Company/org mentions (needs contacts)
        # Multiple companies per cluster OK (family office meetings discuss many entities)
        # but deduplicate same company-contact pair per cluster
        if cached_contacts:
            cluster_company_pairs: set[tuple[str, str]] = set()  # (cid, contact_name)
            company_count = 0
            for file_path, transcript in transcripts:
                segments = transcript.get("_filtered_segments", transcript.get("segments", []))
                company_map = extract_company_mentions(segments, cached_contacts)
                for cid, companies_found in company_map.items():
                    if cid not in profiles:
                        continue
                    for company in companies_found:
                        for c in cached_contacts:
                            if (c.get("companyName") or "").lower() == company.lower():
                                name = c.get("displayName", "")
                                if not name:
                                    continue
                                pair = (cid, name)
                                if pair not in cluster_company_pairs:
                                    cluster_company_pairs.add(pair)
                                    profiles[cid].evidence.append(SpeakerEvidence(
                                        source="company_mention",
                                        name=name,
                                        confidence=0.45,
                                        detail=f"Mentions {company} — contact {name} works there",
                                        call_id=file_path.stem,
                                    ))
                                    company_count += 1
                                break  # First contact per company
            if company_count:
                logger.info("Company mention matches: %d entries", company_count)

        # --- NEW HIGH-VALUE SIGNALS ---
        from deepscript.core.speaker_signals import (
            find_elimination_candidates,
            match_title_names_to_speakers,
            detect_self_identification,
        )
        from datetime import datetime, timedelta, timezone as _tz

        # Build transcript → calendar event mapping (reused by multiple signals)
        transcript_events: dict[str, dict] = {}  # file_stem → event
        if cached_calendar_events:
            for file_path, transcript in transcripts:
                ct = transcript.get("metadata", {}).get("audio", {}).get("format_tags", {}).get("creation_time", "")
                if not ct:
                    ct = transcript.get("metadata", {}).get("audio", {}).get("creation_time", "")
                if not ct:
                    continue
                try:
                    rec_dt = datetime.fromisoformat(ct.replace("Z", "+00:00"))
                    if rec_dt.tzinfo is None:
                        rec_dt = rec_dt.replace(tzinfo=_tz.utc)
                except (ValueError, TypeError):
                    continue
                for event in cached_calendar_events:
                    evt_start = event.get("start", {}).get("dateTime", "")
                    if not evt_start:
                        continue
                    try:
                        evt_dt = datetime.fromisoformat(evt_start.replace("Z", "+00:00"))
                        if evt_dt.tzinfo is None:
                            evt_dt = evt_dt.replace(tzinfo=_tz.utc)
                        if abs((rec_dt - evt_dt).total_seconds()) < 3600:
                            transcript_events[file_path.stem] = event
                            break
                    except (ValueError, TypeError):
                        continue

        # Detect account owner full name for filtering
        if cached_calendar_events and not _owner_full:
            _, _owner_full = _detect_account_owner(cached_calendar_events)

        # 6. Self-identification in opening segments
        self_id_count = 0
        for file_path, transcript in transcripts:
            segments = transcript.get("_filtered_segments", transcript.get("segments", []))
            self_ids = detect_self_identification(segments)
            for ev in self_ids:
                cid = ev["cluster_id"]
                if _owner_first and ev["name"].lower() == _owner_first:
                    continue
                if cid in profiles:
                    already = any(e.source == "self_identification" for e in profiles[cid].evidence)
                    if not already:
                        profiles[cid].evidence.append(SpeakerEvidence(
                            source="self_identification",
                            name=ev["name"],
                            confidence=ev["confidence"],
                            detail=ev["detail"][:100],
                            call_id=file_path.stem,
                        ))
                        self_id_count += 1
        if self_id_count:
            logger.info("Self-identification: %d speakers found", self_id_count)

        # 7. Meeting title name parsing
        title_count = 0
        for file_path, transcript in transcripts:
            event = transcript_events.get(file_path.stem)
            if not event:
                continue
            title_ev = match_title_names_to_speakers(
                transcript, event, profiles, account_owner_name=_owner_full
            )
            for ev in title_ev:
                cid = ev["cluster_id"]
                if cid in profiles:
                    already = any(
                        e.source == "title_parsing" and e.name == ev["name"]
                        for e in profiles[cid].evidence
                    )
                    if not already:
                        profiles[cid].evidence.append(SpeakerEvidence(
                            source="title_parsing",
                            name=ev["name"],
                            confidence=ev["confidence"],
                            detail=ev["detail"][:100],
                            call_id=file_path.stem,
                        ))
                        title_count += 1
        if title_count:
            logger.info("Meeting title parsing: %d names extracted", title_count)

        # 8. Process of elimination (run LAST — benefits from all prior evidence)
        elim_count = 0
        for file_path, transcript in transcripts:
            event = transcript_events.get(file_path.stem)
            if not event:
                continue
            elim_ev = find_elimination_candidates(
                transcript, profiles, event, account_owner_name=_owner_full
            )
            for ev in elim_ev:
                cid = ev["cluster_id"]
                if cid in profiles:
                    already = any(e.source == "elimination" for e in profiles[cid].evidence)
                    if not already:
                        profiles[cid].evidence.append(SpeakerEvidence(
                            source="elimination",
                            name=ev["name"],
                            confidence=ev["confidence"],
                            detail=ev["detail"][:100],
                            call_id=file_path.stem,
                        ))
                        elim_count += 1
        if elim_count:
            logger.info("Process of elimination: %d speakers identified", elim_count)

    except ImportError as e:
        logger.warning("Advanced speaker signals unavailable: %s", e)

    # Resolve best name for each profile (first pass — before LLM panel)
    for profile in profiles.values():
        _resolve_name(profile)

    # --- LLM Expert Panel (last resort for unknowns) ---
    # Runs after all algorithmic signals are resolved. Sends unknown speakers'
    # speech samples + calendar candidates to a 3-expert LLM panel.
    if llm is not None:
        try:
            from deepscript.core.speaker_panel import run_speaker_panel

            panel_results = run_speaker_panel(
                profiles=profiles,
                transcripts=transcripts,
                calendar_events=cached_calendar_events,
                contacts=cached_contacts,
                llm=llm,
                account_owner_name=_owner_full if cached_calendar_events else "",
            )

            for result in panel_results:
                cid = result.cluster_id
                if cid in profiles:
                    # Map panel confidence (0-100) to evidence confidence (0.0-1.0)
                    conf = min(0.90, result.confidence / 100)
                    profiles[cid].evidence.append(SpeakerEvidence(
                        source="llm_panel",
                        name=result.name,
                        confidence=conf,
                        detail=f"3-expert panel ({result.confidence}%): {result.reasoning[:100]}",
                    ))
                    # Re-resolve this profile with the new evidence
                    _resolve_name(profiles[cid])

            if panel_results:
                logger.info("LLM expert panel: %d speakers identified", len(panel_results))
        except Exception as e:
            logger.warning("LLM panel failed: %s", e, exc_info=True)

        # Label remaining unknowns with descriptive tags (separate try block)
        try:
            from deepscript.core.speaker_panel import label_unknown_speakers
            labels = label_unknown_speakers(profiles, transcripts, llm)
            for label in labels:
                cid = label.cluster_id
                if cid in profiles and not profiles[cid].likely_name:
                    profiles[cid].evidence.append(SpeakerEvidence(
                        source="llm_label",
                        name=label.label,
                        confidence=0.40,  # Low — it's a description, not a name
                        detail=f"LLM label ({label.category}): {label.reasoning[:80]}",
                    ))
                    _resolve_name(profiles[cid])
            if labels:
                logger.info("LLM labeling: %d speakers labeled", len(labels))
        except Exception as e:
            logger.warning("LLM labeling failed: %s", e, exc_info=True)

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
    """Load AudioScript speaker identity database.

    Resolves soft-merged clusters: if a cluster has `linked_to`, its
    canonical_name is inherited from the primary cluster so downstream
    code treats them as the same person.
    """
    if not path:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            db = json.load(f)
        identities = db.get("identities", {})

        # Resolve soft-linked clusters: inherit primary's canonical name
        for cid, data in identities.items():
            linked_to = data.get("linked_to")
            if linked_to and linked_to in identities:
                primary = identities[linked_to]
                if primary.get("canonical_name") and not data.get("canonical_name"):
                    data["canonical_name"] = primary["canonical_name"]

        return identities
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


def _check_name_word_probability(
    name: str,
    segments: list[dict],
) -> float | None:
    """Check the average Whisper word probability for a name in segments.

    Returns the average probability of the name's words, or None if not found.
    Low probability (<0.30) suggests ASR hallucination.
    """
    name_lower = name.lower().split()[0]  # Check first name only
    probs = []

    for seg in segments:
        for word_data in seg.get("words", []):
            word = (word_data.get("word") or "").strip().lower().rstrip(".,!?;:")
            if word == name_lower:
                prob = word_data.get("probability", 0)
                if prob > 0:
                    probs.append(prob)

    if not probs:
        return None
    return sum(probs) / len(probs)


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
    cached_events: list[dict] | None = None,
) -> None:
    """Add calendar attendee evidence to speaker profiles.

    If cached_events is provided, matches transcripts against the pre-fetched
    events by timestamp (±30 min window). Otherwise falls back to per-transcript
    CLI lookups (slow).
    """
    from datetime import datetime, timedelta, timezone

    if cached_events is not None:
        _add_calendar_evidence_cached(profiles, transcripts, cached_events)
        return

    for file_path, transcript in transcripts:
        metadata = transcript.get("metadata", {})
        creation_time = metadata.get("audio", {}).get("format_tags", {}).get("creation_time")
        if not creation_time:
            creation_time = metadata.get("audio", {}).get("creation_time")
        if not creation_time:
            continue

        # Look up calendar event via CLI
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

            _match_event_to_profiles(profiles, transcript, events[0], file_path)

        except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
            continue


def _add_calendar_evidence_cached(
    profiles: dict[str, SpeakerProfile],
    transcripts: list[tuple[Path, dict]],
    cached_events: list[dict],
) -> None:
    """Match transcripts to pre-fetched calendar events by timestamp."""
    from datetime import datetime, timedelta, timezone

    # Detect account owner: the most frequent organizer across events.
    # This person appears in most events and should be excluded from
    # attendee matching to avoid false positive speaker identification.
    _owner_first, _owner_full = _detect_account_owner(cached_events)
    account_owner = _owner_full
    if account_owner:
        logger.info("Calendar account owner detected: %s (excluded from attendee matching)", account_owner)

    # Parse event times for fast lookup
    event_times: list[tuple[datetime, datetime, dict]] = []
    for event in cached_events:
        try:
            start_str = event.get("start", {}).get("dateTime", "")
            end_str = event.get("end", {}).get("dateTime", "")
            if not start_str:
                continue
            # Parse — Graph API returns times without Z but in UTC
            start_dt = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
            if start_dt.tzinfo is None:
                start_dt = start_dt.replace(tzinfo=timezone.utc)
            end_dt = datetime.fromisoformat(end_str.replace("Z", "+00:00")) if end_str else start_dt + timedelta(hours=1)
            if end_dt.tzinfo is None:
                end_dt = end_dt.replace(tzinfo=timezone.utc)
            event_times.append((start_dt, end_dt, event))
        except (ValueError, TypeError):
            continue

    event_times.sort(key=lambda x: x[0])
    matched = 0

    for file_path, transcript in transcripts:
        metadata = transcript.get("metadata", {})
        creation_time = metadata.get("audio", {}).get("format_tags", {}).get("creation_time")
        if not creation_time:
            creation_time = metadata.get("audio", {}).get("creation_time")
        if not creation_time:
            continue

        try:
            rec_dt = datetime.fromisoformat(creation_time.replace("Z", "+00:00"))
            if rec_dt.tzinfo is None:
                rec_dt = rec_dt.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            continue

        duration = metadata.get("audio", {}).get("duration_seconds", 3600)
        window = timedelta(minutes=30)

        # Find events overlapping [rec_dt - 30min, rec_dt + duration + 30min]
        rec_start = rec_dt - window
        rec_end = rec_dt + timedelta(seconds=duration) + window

        best_event = None
        best_overlap = timedelta(0)
        for evt_start, evt_end, event in event_times:
            if evt_end < rec_start:
                continue
            if evt_start > rec_end:
                break  # sorted, no more matches
            # Calculate overlap
            overlap_start = max(evt_start, rec_start)
            overlap_end = min(evt_end, rec_end)
            overlap = overlap_end - overlap_start
            if overlap > best_overlap:
                best_overlap = overlap
                best_event = event

        if best_event:
            _match_event_to_profiles(profiles, transcript, best_event, file_path, skip_name=account_owner)
            matched += 1

    logger.info("Calendar matching: %d/%d transcripts matched to events", matched, len(transcripts))


def _match_event_to_profiles(
    profiles: dict[str, SpeakerProfile],
    transcript: dict,
    event: dict,
    file_path: Path,
    skip_name: str = "",
) -> None:
    """Match calendar event attendees to speaker profiles."""
    # Skip the account owner from attendee matching — they appear in every
    # event and would pollute all speaker profiles with false matches.
    organizer_name = (event.get("organizer", {}).get("emailAddress", {}).get("name") or "").lower()

    attendees = []
    for att in event.get("attendees", []):
        name = att.get("emailAddress", {}).get("name") or att.get("displayName") or ""
        email = att.get("emailAddress", {}).get("address", "")
        # RSVP status: accepted attendees are near-certain to be present
        response = att.get("status", {}).get("response", "none")
        accepted = response in ("accepted", "organizer", "tentativelyAccepted")
        if name and name.lower() != organizer_name and name.lower() != skip_name:
            attendees.append({"name": name, "email": email, "accepted": accepted})

    if not attendees:
        return

    subject = event.get("subject", "")

    # Get cluster IDs in this transcript
    diar = transcript.get("diarization", {})
    resolved = diar.get("speakers_resolved", [])
    call_clusters = set()
    for sr in resolved:
        cid = sr.get("speaker_cluster_id", "")
        if cid:
            call_clusters.add(cid)

    for cid in call_clusters:
        if cid not in profiles:
            continue
        profile = profiles[cid]
        # Check if any existing evidence name matches an attendee
        existing_names = [ev.name for ev in profile.evidence]
        for att in attendees:
            att_name = att["name"]
            for ev_name in existing_names:
                if _names_match(ev_name, att_name):
                    # Avoid duplicate calendar evidence for same event
                    already_has = any(
                        e.source == "calendar" and e.name == att_name and e.call_id == file_path.stem
                        for e in profile.evidence
                    )
                    if not already_has:
                        # RSVP-accepted attendees get higher confidence
                        conf = 0.85 if att.get("accepted") else 0.75
                        source = "calendar_rsvp" if att.get("accepted") else "calendar"
                        profile.evidence.append(SpeakerEvidence(
                            source=source,
                            name=att_name,
                            confidence=conf,
                            detail=f"Calendar attendee ({('accepted' if att.get('accepted') else 'invited')}): {subject}",
                            call_id=file_path.stem,
                        ))
                    break


def _add_contacts_evidence(
    profiles: dict[str, SpeakerProfile],
    provider: str,
    cached_contacts: list[dict] | None = None,
) -> None:
    """Match speaker names against Outlook/Google contacts.

    If cached_contacts is provided, uses the pre-fetched contact list.
    Otherwise falls back to CLI lookups.
    """
    try:
        contacts = cached_contacts
        if contacts is None:
            if provider == "ms365":
                result = subprocess.run(
                    ["ms365", "contacts", "list", "--all", "-o", "json"],
                    capture_output=True, text=True, timeout=30,
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
        contact_names: dict[str, dict] = {}  # normalized_full_name → contact info
        contact_first_names: dict[str, dict] = {}  # first_name → contact info
        for c in contacts:
            name = c.get("displayName") or ""
            if not name:
                given = c.get("givenName", "")
                surname = c.get("surname", "")
                name = f"{given} {surname}".strip()
            if not name:
                continue

            # Extract primary email
            email = ""
            email_addresses = c.get("emailAddresses") or []
            if email_addresses:
                email = email_addresses[0].get("address", "")
            if not email:
                primary = c.get("primaryEmailAddress") or {}
                email = primary.get("address", "")

            contact_info = {
                "name": name,
                "company": c.get("companyName") or "",
                "title": c.get("jobTitle") or "",
                "email": email,
                "phone": c.get("mobilePhone") or "",
            }

            contact_names[name.lower()] = contact_info
            # Also index by first name — stored separately to apply lower confidence
            first = name.split()[0].lower()
            if first not in contact_first_names:
                contact_first_names[first] = contact_info

        logger.info("Contacts loaded: %d entries", len(contact_names))

        # Match profiles against contacts
        matched = 0
        for profile in profiles.values():
            # Snapshot current evidence to avoid iterating over additions
            current_evidence = list(profile.evidence)
            profile_matched = False

            # Skip first-name matching if profile already has strong identity from speaker DB
            has_db_name = any(e.source == "speaker_db" and e.confidence >= 0.85 for e in current_evidence)

            for ev in current_evidence:
                name_lower = ev.name.lower().strip()
                if not name_lower:
                    continue

                # Full name match (high confidence)
                if name_lower in contact_names:
                    contact = contact_names[name_lower]
                    already_has = any(e.source == "contacts" and e.name == contact["name"] for e in profile.evidence)
                    if not already_has:
                        detail_parts = [f"Contact: {contact['name']}"]
                        if contact["title"]:
                            detail_parts.append(contact["title"])
                        if contact["company"]:
                            detail_parts.append(f"at {contact['company']}")
                        if contact["email"]:
                            detail_parts.append(f"({contact['email']})")
                        profile.evidence.append(SpeakerEvidence(
                            source="contacts",
                            name=contact["name"],
                            confidence=0.75,
                            detail=", ".join(detail_parts),
                        ))
                        if contact.get("title") and not profile.role:
                            profile.role = f"{contact['title']}" + (f" at {contact['company']}" if contact.get("company") else "")
                        if contact.get("company") and not profile.company:
                            profile.company = contact["company"]
                        profile_matched = True

                # First-name-only match (lower confidence, only if evidence is a single word
                # and profile doesn't already have a confirmed identity from speaker DB)
                elif not has_db_name and " " not in name_lower and name_lower in contact_first_names:
                    contact = contact_first_names[name_lower]
                    already_has = any(e.source in ("contacts", "contacts_partial") and e.name == contact["name"] for e in profile.evidence)
                    if not already_has:
                        profile.evidence.append(SpeakerEvidence(
                            source="contacts_partial",
                            name=contact["name"],
                            confidence=0.50,
                            detail=f"Contact first-name match: {contact['name']}",
                        ))
                        profile_matched = True

            if profile_matched:
                matched += 1

        logger.info("Contacts matching: %d profiles matched", matched)

    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        logger.warning("Contact list lookup failed: %s", e)


def _detect_account_owner(events: list[dict]) -> tuple[str, str]:
    """Detect the calendar account owner from organizer frequency.

    Returns (first_name_lower, full_name_lower). Empty strings if no events.
    """
    from collections import Counter
    org_counts = Counter()
    for e in events:
        org = (e.get("organizer", {}).get("emailAddress", {}).get("name") or "").strip()
        if org:
            org_counts[org] += 1
    if not org_counts:
        return ("", "")
    full = org_counts.most_common(1)[0][0]
    return (full.split()[0].lower(), full.lower())


def _names_match(name1: str, name2: str) -> bool:
    """Check if two names likely refer to the same person."""
    n1 = name1.lower().strip()
    n2 = name2.lower().strip()
    if not n1 or not n2:
        return False
    if n1 == n2:
        return True
    # First name match
    if n1.split() and n2.split() and n1.split()[0] == n2.split()[0]:
        return True
    # Token subset (one is a subset of the other's words)
    s1, s2 = set(n1.split()), set(n2.split())
    if s1 and s2 and (s1 <= s2 or s2 <= s1):
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


# --- Speaker DB Writeback ---


def _is_upgrade(existing_name: str | None, new_name: str, new_confidence: float) -> bool:
    """Check if the new name is an upgrade over the existing one.

    Upgrades:
    - No existing name → any name is an upgrade
    - "Adam" → "Adam Fuller" (more specific)
    - "Adam (possible)" → "Adam (likely)" (higher confidence)
    - "Adam (likely)" → "Adam Fuller" (qualifier removed + full name)
    - Unqualified existing name requires higher confidence to replace

    NOT upgrades:
    - "Adam Fuller" → "Adam" (less specific)
    - "Adam Fuller" → "Tom" (different person — flag as conflict)
    """
    if not existing_name:
        return True

    # Strip qualifiers for comparison
    clean_existing = existing_name.split(" (")[0].strip()
    clean_new = new_name.split(" (")[0].strip()

    # Same base name, new is longer (more specific) → upgrade
    if clean_existing.lower() in clean_new.lower() and len(clean_new) > len(clean_existing):
        return True

    # Same base name, existing had qualifier, new doesn't → upgrade
    if clean_existing.lower() == clean_new.lower() and "(" in existing_name and "(" not in new_name:
        return True

    # Existing had qualifier, new has better qualifier → upgrade
    qual_rank = {"(possible)": 1, "(likely)": 2}
    existing_qual = 0
    new_qual = 3  # No qualifier = best
    for q, rank in qual_rank.items():
        if q in existing_name:
            existing_qual = rank
        if q in new_name:
            new_qual = rank
    if clean_existing.lower() == clean_new.lower() and new_qual > existing_qual:
        return True

    # Require higher confidence to replace an unqualified existing name
    if "(" not in existing_name and new_confidence < 0.80:
        return False

    return False


def writeback_to_speaker_db(
    profiles: dict[str, SpeakerProfile],
    speaker_db_path: str | Path,
    min_confidence: float = 0.40,
) -> dict[str, Any]:
    """Write identified speaker names back to AudioScript's speaker_identities.json.

    Progressive naming with versioned history:
    - Upgrades existing names when we learn more (Adam → Adam Fuller)
    - Tracks name history with timestamps, sources, and confidence
    - Maintains aliases (goes-by names, professional vs personal names)
    - Never downgrades — only replaces with more specific or higher confidence names
    - Flags conflicts when evidence points to different people

    Returns: summary of changes made.
    """
    from datetime import datetime, timezone
    import os
    import uuid

    path = Path(speaker_db_path)
    if not path.exists():
        logger.warning("Speaker DB not found: %s", path)
        return {"error": "file not found"}

    with open(path) as f:
        db = json.load(f)

    now = datetime.now(timezone.utc).isoformat()
    identities = db.get("identities", {})
    changes = {"updated": [], "upgraded": [], "conflicts": [],
               "skipped_low_confidence": [], "not_in_db": []}

    for cid, profile in profiles.items():
        if not profile.likely_name or profile.name_confidence < min_confidence:
            continue

        if cid not in identities:
            # Create a new DB entry for speakers found in transcripts but not in the DB.
            # These are typically brief speakers or diarization artifacts without embeddings.
            full_name = profile.best_full_name
            candidate_name = profile.likely_name
            if profile.name_confidence >= 0.80:
                candidate_name = full_name or candidate_name
                candidate_status = "deepscript_confident"
            elif profile.name_confidence >= 0.60:
                candidate_status = "deepscript_likely"
            else:
                candidate_status = "deepscript_possible"

            new_identity = {
                "speaker_cluster_id": cid,
                "canonical_name": candidate_name,
                "aliases": [],
                "status": "unknown",
                "embedding_centroid": [],
                "sample_count": 0,
                "first_seen": now,
                "last_seen": now,
                "total_calls": profile.total_calls,
                "total_speaking_seconds": profile.total_speaking_seconds,
                "typical_co_speakers": list(profile.co_speakers.keys())[:10],
                "created_from": "deepscript_identification",
                "updated_at": now,
                "deepscript_status": candidate_status,
                "deepscript_confidence": round(profile.name_confidence, 3),
                "deepscript_role": profile.role,
                "deepscript_evidence_count": len(profile.evidence),
                "deepscript_full_name": full_name,
                "deepscript_updated_at": now,
            }
            _add_name_history(new_identity, candidate_name, profile.name_confidence,
                              candidate_status, profile, now)
            identities[cid] = new_identity
            changes["not_in_db"].append({
                "cluster_id": cid,
                "name": candidate_name,
                "confidence": profile.name_confidence,
                "calls": profile.total_calls,
                "action": "created",
            })
            continue

        identity = identities[cid]
        existing_name = identity.get("canonical_name")
        full_name = profile.best_full_name

        # Build the candidate name
        if profile.name_confidence >= 0.80:
            candidate_name = full_name or profile.likely_name
            candidate_status = "deepscript_confident"
        elif profile.name_confidence >= 0.60:
            candidate_name = f"{profile.likely_name} (likely)"
            candidate_status = "deepscript_likely"
        else:
            candidate_name = f"{profile.likely_name} (possible)"
            candidate_status = "deepscript_possible"

        # Check for conflict — different base name
        clean_existing = (existing_name or "").split(" (")[0].strip().lower()
        clean_candidate = candidate_name.split(" (")[0].strip().lower()
        if existing_name and clean_existing and clean_candidate and clean_existing != clean_candidate:
            # Check if one contains the other (Adam vs Adam Fuller = not a conflict)
            if clean_existing not in clean_candidate and clean_candidate not in clean_existing:
                # Check not_same_as — if this conflict was already reviewed and confirmed,
                # silently skip instead of re-flagging every run
                not_same = set(identity.get("not_same_as", []))
                # Find clusters whose name matches the candidate — if they're in not_same_as, skip
                already_resolved = False
                for other_cid, other_data in identities.items():
                    if other_cid == cid:
                        continue
                    other_name = (other_data.get("canonical_name") or "").split(" (")[0].strip().lower()
                    if other_name and other_name == clean_candidate and other_cid in not_same:
                        already_resolved = True
                        break

                if not already_resolved:
                    changes["conflicts"].append({
                        "cluster_id": cid,
                        "existing_name": existing_name,
                        "proposed_name": candidate_name,
                        "confidence": profile.name_confidence,
                    })
                # Don't overwrite — flag for human review
                # But still record in name_history
                _add_name_history(identity, candidate_name, profile.name_confidence,
                                  candidate_status, profile, now, conflict=True)
                continue

        # Check if this is an upgrade
        if existing_name and not _is_upgrade(existing_name, candidate_name, profile.name_confidence):
            # Not an upgrade — but still record the evidence in history
            _add_name_history(identity, candidate_name, profile.name_confidence,
                              candidate_status, profile, now)
            continue

        # Apply the update
        change_type = "upgraded" if existing_name else "updated"

        # Record in name history BEFORE overwriting
        _add_name_history(identity, candidate_name, profile.name_confidence,
                          candidate_status, profile, now)

        # Update canonical name
        identity["canonical_name"] = candidate_name
        identity["deepscript_status"] = candidate_status
        identity["deepscript_confidence"] = round(profile.name_confidence, 3)
        identity["deepscript_role"] = profile.role
        identity["deepscript_evidence_count"] = len(profile.evidence)
        identity["deepscript_full_name"] = full_name
        identity["deepscript_updated_at"] = now

        # Manage aliases — all known names for this person
        aliases = identity.get("aliases", [])
        # Add previous canonical name as alias
        if existing_name and existing_name not in aliases:
            clean = existing_name.split(" (")[0].strip()
            if clean and clean not in aliases:
                aliases.append(clean)
        # Add first-name-only as alias if we have full name
        if full_name and " " in full_name:
            first = full_name.split()[0]
            if first not in aliases:
                aliases.append(first)
        # Add likely_name if different from canonical
        if profile.likely_name and profile.likely_name != candidate_name:
            clean = profile.likely_name.split(" (")[0].strip()
            if clean not in aliases:
                aliases.append(clean)
        # Deduplicate
        identity["aliases"] = sorted(set(aliases))

        changes[change_type].append({
            "cluster_id": cid,
            "old_name": existing_name,
            "new_name": candidate_name,
            "confidence": profile.name_confidence,
            "status": candidate_status,
            "full_name": full_name,
            "role": profile.role,
            "calls": profile.total_calls,
            "aliases": identity["aliases"],
        })

    # Save updated DB
    created_count = sum(1 for x in changes["not_in_db"] if isinstance(x, dict) and x.get("action") == "created")
    total_changes = len(changes["updated"]) + len(changes["upgraded"]) + created_count
    if total_changes > 0 or changes["conflicts"]:
        db["identities"] = identities

        tmp = path.with_suffix(f".tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}")
        try:
            with open(tmp, "w") as f:
                json.dump(db, f, indent=2, default=str)
            tmp.replace(path)
        finally:
            if tmp.exists():
                tmp.unlink()

        logger.info("Speaker DB: %d new, %d upgraded, %d created, %d conflicts in %s",
                     len(changes["updated"]), len(changes["upgraded"]),
                     created_count, len(changes["conflicts"]), path)

    return {
        "new_names": len(changes["updated"]),
        "upgraded": len(changes["upgraded"]),
        "created": created_count,
        "conflicts": len(changes["conflicts"]),
        "skipped_low_confidence": len(changes["skipped_low_confidence"]),
        "not_in_db": len(changes["not_in_db"]),
        "details": changes,
    }


@dataclass
class MergeCandidate:
    """A pair of speaker clusters that may be the same person."""

    cluster_a: str
    cluster_b: str
    name_a: str
    name_b: str
    total_score: float  # Combined confidence score
    embedding_similarity: float
    name_similarity: float
    co_speaker_overlap: float
    calls_a: int
    calls_b: int
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_a": self.cluster_a,
            "cluster_b": self.cluster_b,
            "name_a": self.name_a,
            "name_b": self.name_b,
            "total_score": round(self.total_score, 3),
            "embedding_similarity": round(self.embedding_similarity, 3),
            "name_similarity": round(self.name_similarity, 3),
            "co_speaker_overlap": round(self.co_speaker_overlap, 3),
            "calls_a": self.calls_a,
            "calls_b": self.calls_b,
            "reasons": self.reasons,
        }


def find_duplicate_speakers(
    speaker_db_path: str | Path,
    embedding_threshold: float = 0.75,
    name_threshold: float = 0.70,
    min_score: float = 0.50,
) -> list[MergeCandidate]:
    """Scan all speaker clusters for potential duplicates.

    Compares clusters pairwise using three signals:
    1. Embedding cosine similarity (voice match)
    2. Name similarity (same/similar canonical names)
    3. Co-speaker overlap (appear with same people)

    Args:
        speaker_db_path: Path to speaker_identities.json.
        embedding_threshold: Min embedding similarity to flag (default 0.75).
        name_threshold: Min name similarity to flag (default 0.70).
        min_score: Min combined score to include in results (default 0.50).

    Returns: List of MergeCandidate sorted by total_score descending.
    """
    import math

    path = Path(speaker_db_path)
    if not path.exists():
        logger.warning("Speaker DB not found: %s", path)
        return []

    with open(path) as f:
        db = json.load(f)

    identities = db.get("identities", {})
    if len(identities) < 2:
        return []

    # Pre-compute data for all clusters
    clusters: list[dict[str, Any]] = []
    for cid, data in identities.items():
        clusters.append({
            "id": cid,
            "name": data.get("canonical_name") or "",
            "aliases": set(a.lower() for a in data.get("aliases", []) if a),
            "embedding": data.get("embedding_centroid") or [],
            "co_speakers": set(data.get("typical_co_speakers", [])),
            "calls": data.get("total_calls", 0),
            "seconds": data.get("total_speaking_seconds", 0),
            "not_same_as": set(data.get("not_same_as", [])),
        })

    candidates: list[MergeCandidate] = []

    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            a, b = clusters[i], clusters[j]

            # Skip pairs confirmed as different people
            if b["id"] in a["not_same_as"] or a["id"] in b["not_same_as"]:
                continue

            reasons = []

            # 1. Embedding similarity
            emb_sim = 0.0
            if a["embedding"] and b["embedding"] and len(a["embedding"]) == len(b["embedding"]):
                dot = sum(x * y for x, y in zip(a["embedding"], b["embedding"]))
                norm_a = math.sqrt(sum(x * x for x in a["embedding"]))
                norm_b = math.sqrt(sum(x * x for x in b["embedding"]))
                if norm_a > 0 and norm_b > 0:
                    emb_sim = dot / (norm_a * norm_b)
                if emb_sim >= embedding_threshold:
                    reasons.append(f"voice similarity {emb_sim:.2f}")

            # 2. Name similarity
            name_sim = _compute_name_similarity(
                a["name"], b["name"], a["aliases"], b["aliases"]
            )
            if name_sim >= name_threshold:
                reasons.append(f"name match ({a['name']} ↔ {b['name']})")

            # 3. Co-speaker overlap (Jaccard)
            co_overlap = 0.0
            if a["co_speakers"] and b["co_speakers"]:
                # Remove each other from co-speaker sets
                a_co = a["co_speakers"] - {b["id"]}
                b_co = b["co_speakers"] - {a["id"]}
                if a_co or b_co:
                    intersection = a_co & b_co
                    union = a_co | b_co
                    co_overlap = len(intersection) / len(union) if union else 0.0
                    if co_overlap > 0.3:
                        reasons.append(f"shared co-speakers ({len(intersection)}/{len(union)})")

            # Combined score: weighted sum
            # Voice is strongest signal, name second, co-speakers corroborating
            total_score = (
                emb_sim * 0.50 +      # Voice match is king
                name_sim * 0.35 +      # Name match is strong
                co_overlap * 0.15      # Co-speakers are supporting evidence
            )

            if total_score >= min_score and reasons:
                candidates.append(MergeCandidate(
                    cluster_a=a["id"],
                    cluster_b=b["id"],
                    name_a=a["name"] or a["id"],
                    name_b=b["name"] or b["id"],
                    total_score=total_score,
                    embedding_similarity=emb_sim,
                    name_similarity=name_sim,
                    co_speaker_overlap=co_overlap,
                    calls_a=a["calls"],
                    calls_b=b["calls"],
                    reasons=reasons,
                ))

    candidates.sort(key=lambda c: -c.total_score)
    logger.info("Dedup scan: %d pairs checked, %d merge candidates found",
                len(clusters) * (len(clusters) - 1) // 2, len(candidates))
    return candidates


def _compute_name_similarity(
    name_a: str, name_b: str,
    aliases_a: set[str], aliases_b: set[str],
) -> float:
    """Compute name similarity between two speakers.

    Returns 0.0-1.0 based on exact match, containment, and alias overlap.
    """
    if not name_a or not name_b:
        return 0.0

    na = name_a.lower().split(" (")[0].strip()
    nb = name_b.lower().split(" (")[0].strip()

    # Exact match
    if na == nb:
        return 1.0

    # One contains the other (Adam vs Adam Fuller)
    if na in nb or nb in na:
        return 0.85

    # First name match
    fa = na.split()[0] if na else ""
    fb = nb.split()[0] if nb else ""
    if fa and fb and fa == fb:
        return 0.60

    # Alias match: does name_a appear in aliases_b or vice versa?
    all_a = {na} | aliases_a
    all_b = {nb} | aliases_b
    if all_a & all_b:
        return 0.75

    # Check if any alias first name matches
    first_a = {n.split()[0] for n in all_a if n}
    first_b = {n.split()[0] for n in all_b if n}
    if first_a & first_b:
        return 0.50

    return 0.0


def _backup_speaker_db(speaker_db_path: str | Path) -> Path:
    """Create a timestamped backup of the speaker DB before destructive operations.

    Stored alongside the DB: speaker_identities.backup.2026-03-26T18-00-00.json
    Returns the backup path.
    """
    from datetime import datetime, timezone
    import shutil

    path = Path(speaker_db_path)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    backup = path.with_suffix(f".backup.{ts}.json")
    shutil.copy2(path, backup)
    logger.info("Speaker DB backed up to %s", backup)
    return backup


def soft_merge_speakers(
    speaker_db_path: str | Path,
    keep_id: str,
    merge_id: str,
) -> bool:
    """Soft-merge: link two clusters without destroying either.

    Adds a `linked_to` field on the secondary cluster pointing to the primary.
    Both clusters remain in the DB. Downstream queries should resolve linked
    clusters to the primary. Reversible via `unmerge_speakers()`.
    """
    import os
    import uuid
    from datetime import datetime, timezone

    path = Path(speaker_db_path)
    with open(path) as f:
        db = json.load(f)

    identities = db.get("identities", {})
    a = identities.get(keep_id)
    b = identities.get(merge_id)
    if not a or not b:
        return False

    # Don't double-link
    if b.get("linked_to") == keep_id:
        logger.info("Already linked: %s → %s", merge_id, keep_id)
        return True

    # Refuse re-linking an already-linked cluster
    if b.get("linked_to"):
        logger.warning("Refusing re-link: %s is already linked to %s", merge_id, b["linked_to"])
        return False

    # Prevent circular links
    if a.get("linked_to"):
        logger.warning("Cannot link: %s is already linked to %s", keep_id, a["linked_to"])
        return False

    now = datetime.now(timezone.utc).isoformat()

    # Mark secondary as linked
    b["linked_to"] = keep_id
    b["linked_at"] = now

    # Add the secondary's name as alias on primary
    aliases = set(a.get("aliases", []))
    if b.get("canonical_name"):
        aliases.add(b["canonical_name"])
    aliases.update(b.get("aliases", []))
    a["aliases"] = sorted(aliases)

    # Aggregate stats on primary (but keep originals on secondary for undo)
    b["_pre_link_stats"] = {
        "total_calls": b.get("total_calls", 0),
        "total_speaking_seconds": b.get("total_speaking_seconds", 0),
        "sample_count": b.get("sample_count", 0),
    }
    a["total_calls"] = a.get("total_calls", 0) + b.get("total_calls", 0)
    a["total_speaking_seconds"] = (
        a.get("total_speaking_seconds", 0) + b.get("total_speaking_seconds", 0)
    )

    # Merge co-speakers onto primary
    co = set(a.get("typical_co_speakers", []) + b.get("typical_co_speakers", []))
    co.discard(keep_id)
    co.discard(merge_id)
    a["typical_co_speakers"] = list(co)

    # Merge embedding centroids (weighted average) — keep original on secondary for undo
    emb_a = a.get("embedding_centroid", [])
    emb_b = b.get("embedding_centroid", [])
    n_a = a.get("sample_count", 1)
    n_b = b.get("sample_count", 1)
    if emb_a and emb_b and len(emb_a) == len(emb_b):
        b["_pre_link_embedding"] = list(emb_a)  # Store primary's original embedding
        b["_pre_link_sample_count"] = n_a
        a["embedding_centroid"] = [
            (ea * n_a + eb * n_b) / (n_a + n_b)
            for ea, eb in zip(emb_a, emb_b)
        ]
        a["sample_count"] = n_a + n_b

    a["updated_at"] = now

    # Save
    tmp = path.with_suffix(f".tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}")
    try:
        with open(tmp, "w") as f:
            json.dump(db, f, indent=2, default=str)
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()

    logger.info("Soft-merged: %s → %s (linked, reversible)", merge_id, keep_id)
    return True


def unmerge_speakers(
    speaker_db_path: str | Path,
    cluster_id: str,
) -> bool:
    """Undo a soft-merge: unlink a cluster from its primary.

    Restores the secondary cluster's independence and reverses stat aggregation.
    Only works on soft-merged clusters (those with `linked_to` field).
    """
    import os
    import uuid

    path = Path(speaker_db_path)
    with open(path) as f:
        db = json.load(f)

    identities = db.get("identities", {})
    b = identities.get(cluster_id)
    if not b:
        logger.warning("Cluster not found: %s", cluster_id)
        return False

    keep_id = b.get("linked_to")
    if not keep_id:
        logger.warning("Cluster %s is not linked (not a soft-merge)", cluster_id)
        return False

    a = identities.get(keep_id)
    if not a:
        logger.warning("Primary cluster %s not found", keep_id)
        # Still remove the link from the secondary
        del b["linked_to"]
        del b["linked_at"]
        b.pop("_pre_link_stats", None)
        b.pop("_pre_link_embedding", None)
        b.pop("_pre_link_sample_count", None)
    else:
        # Restore stats on primary
        pre_stats = b.get("_pre_link_stats", {})
        if pre_stats:
            a["total_calls"] = max(0, a.get("total_calls", 0) - pre_stats.get("total_calls", 0))
            a["total_speaking_seconds"] = max(0, (
                a.get("total_speaking_seconds", 0) - pre_stats.get("total_speaking_seconds", 0)
            ))

        # Restore primary's original embedding
        pre_emb = b.get("_pre_link_embedding")
        pre_sc = b.get("_pre_link_sample_count")
        if pre_emb:
            a["embedding_centroid"] = pre_emb
            a["sample_count"] = pre_sc or a.get("sample_count", 1)

        # Remove secondary's name from primary aliases
        if b.get("canonical_name") and b["canonical_name"] in a.get("aliases", []):
            a["aliases"].remove(b["canonical_name"])

        a["updated_at"] = b.get("linked_at", "")

    # Remove link fields from secondary
    del b["linked_to"]
    b.pop("linked_at", None)
    b.pop("_pre_link_stats", None)
    b.pop("_pre_link_embedding", None)
    b.pop("_pre_link_sample_count", None)

    # Save
    tmp = path.with_suffix(f".tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}")
    try:
        with open(tmp, "w") as f:
            json.dump(db, f, indent=2, default=str)
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()

    logger.info("Unmerged: %s unlinked from %s", cluster_id, keep_id)
    return True


def merge_speakers(
    speaker_db_path: str | Path,
    keep_id: str,
    merge_id: str,
    hard: bool = False,
) -> bool:
    """Merge two speaker clusters.

    Default: soft-merge (reversible via unmerge_speakers).
    With hard=True: physical merge (destructive, uses AudioScript if available).
    Always creates a backup before hard merge.
    """
    if not hard:
        return soft_merge_speakers(speaker_db_path, keep_id, merge_id)

    # Hard merge — backup first
    _backup_speaker_db(speaker_db_path)

    try:
        import sys
        audioscript_path = Path(speaker_db_path).parent.parent
        if str(audioscript_path) not in sys.path:
            sys.path.insert(0, str(audioscript_path))

        from audioscript.speakers.identity_db import SpeakerIdentityDB
        db = SpeakerIdentityDB(str(speaker_db_path))
        result = db.merge_clusters(keep_id, merge_id)
        if result:
            logger.info("Hard-merged %s into %s", merge_id, keep_id)
        return result
    except ImportError:
        return _hard_merge(speaker_db_path, keep_id, merge_id)


def _hard_merge(
    speaker_db_path: str | Path,
    keep_id: str,
    merge_id: str,
) -> bool:
    """Physical merge — clusters are combined, secondary is deleted."""
    import os
    import uuid
    from datetime import datetime, timezone

    path = Path(speaker_db_path)
    with open(path) as f:
        db = json.load(f)

    identities = db.get("identities", {})
    a = identities.get(keep_id)
    b = identities.get(merge_id)
    if not a or not b:
        return False

    now = datetime.now(timezone.utc).isoformat()

    # Merge embedding centroids (weighted average)
    n_a = a.get("sample_count", 1)
    n_b = b.get("sample_count", 1)
    emb_a = a.get("embedding_centroid", [])
    emb_b = b.get("embedding_centroid", [])
    if emb_a and emb_b and len(emb_a) == len(emb_b):
        a["embedding_centroid"] = [
            (ea * n_a + eb * n_b) / (n_a + n_b)
            for ea, eb in zip(emb_a, emb_b)
        ]

    a["sample_count"] = n_a + n_b
    a["total_calls"] = a.get("total_calls", 0) + b.get("total_calls", 0)
    a["total_speaking_seconds"] = (
        a.get("total_speaking_seconds", 0) + b.get("total_speaking_seconds", 0)
    )

    b_fs = b.get("first_seen", "�")
    if b_fs < a.get("first_seen", "z"):
        a["first_seen"] = b_fs
    if b.get("last_seen", "") > a.get("last_seen", ""):
        a["last_seen"] = b["last_seen"]

    co = set(a.get("typical_co_speakers", []) + b.get("typical_co_speakers", []))
    co.discard(keep_id)
    co.discard(merge_id)
    a["typical_co_speakers"] = list(co)

    aliases = set(a.get("aliases", []))
    if b.get("canonical_name"):
        aliases.add(b["canonical_name"])
    aliases.update(b.get("aliases", []))
    a["aliases"] = sorted(aliases)
    a["updated_at"] = now

    hist_a = a.get("name_history", [])
    hist_b = b.get("name_history", [])
    merged_hist = sorted(hist_a + hist_b, key=lambda h: h.get("timestamp", ""))
    a["name_history"] = merged_hist[-20:]

    for occ in db.get("occurrences", []):
        if occ.get("speaker_cluster_id") == merge_id:
            occ["speaker_cluster_id"] = keep_id
    for ev in db.get("evidence", []):
        if ev.get("speaker_cluster_id") == merge_id:
            ev["speaker_cluster_id"] = keep_id

    del identities[merge_id]

    tmp = path.with_suffix(f".tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}")
    try:
        with open(tmp, "w") as f:
            json.dump(db, f, indent=2, default=str)
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()

    logger.info("Hard merge: %s into %s", merge_id, keep_id)
    return True


def mark_not_same(
    speaker_db_path: str | Path,
    cluster_a: str,
    cluster_b: str,
) -> bool:
    """Mark two clusters as confirmed different people.

    Adds each cluster's ID to the other's `not_same_as` list.
    Future conflict detection and dedup scans will skip this pair.
    """
    import os
    import uuid

    path = Path(speaker_db_path)
    if not path.exists():
        return False

    with open(path) as f:
        db = json.load(f)

    identities = db.get("identities", {})
    a = identities.get(cluster_a)
    b = identities.get(cluster_b)
    if not a or not b:
        logger.warning("Cannot mark not-same: cluster(s) not found")
        return False

    # Add to each other's not_same_as list
    a_list = set(a.get("not_same_as", []))
    b_list = set(b.get("not_same_as", []))
    a_list.add(cluster_b)
    b_list.add(cluster_a)
    a["not_same_as"] = sorted(a_list)
    b["not_same_as"] = sorted(b_list)

    # Save
    tmp = path.with_suffix(f".tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}")
    try:
        with open(tmp, "w") as f:
            json.dump(db, f, indent=2, default=str)
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()

    name_a = a.get("canonical_name") or cluster_a
    name_b = b.get("canonical_name") or cluster_b
    logger.info("Marked as different people: %s (%s) ≠ %s (%s)", cluster_a, name_a, cluster_b, name_b)
    return True


def _add_name_history(
    identity: dict[str, Any],
    name: str,
    confidence: float,
    status: str,
    profile: "SpeakerProfile",
    timestamp: str,
    conflict: bool = False,
) -> None:
    """Add an entry to the speaker's name history.

    Tracks every identification attempt with timestamp, source,
    confidence, and evidence. Enables trend analysis and audit trail.
    """
    history = identity.setdefault("name_history", [])

    # Collect evidence sources
    sources = sorted(set(e.source for e in profile.evidence))
    top_evidence = sorted(profile.evidence, key=lambda e: -e.confidence)[:3]

    entry = {
        "timestamp": timestamp,
        "name": name,
        "confidence": round(confidence, 3),
        "status": status,
        "conflict": conflict,
        "sources": sources,
        "evidence_count": len(profile.evidence),
        "top_evidence": [
            {"source": e.source, "name": e.name, "detail": e.detail[:100]}
            for e in top_evidence
        ],
        "role": profile.role,
        "calls_at_time": profile.total_calls,
    }

    # Don't add duplicate entries (same name + confidence within 1 hour)
    if history:
        last = history[-1]
        if last.get("name") == name and last.get("confidence") == round(confidence, 3):
            return

    history.append(entry)

    # Keep last 20 entries max
    if len(history) > 20:
        identity["name_history"] = history[-20:]
