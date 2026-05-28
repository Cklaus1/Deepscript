"""LLM multi-expert panel for unknown speaker identification.

Uses a 3-expert panel (Conversation Analyst, Professional Profiler, Network Analyst)
to analyze unknown speakers' segments across calls and determine their identity
from calendar attendee candidates.

This runs AFTER all algorithmic signals — it's the last resort for speakers
that the rules couldn't identify. Requires an LLM provider.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Max segments to include per call (keep prompt under token limits)
MAX_SEGMENTS_PER_CALL = 15
# Max calls to sample from
MAX_CALLS = 8
# Max words per segment excerpt
MAX_WORDS_PER_SEGMENT = 60


@dataclass
class PanelResult:
    """Result from the LLM expert panel."""
    cluster_id: str
    name: str
    confidence: int  # 0-100
    reasoning: str
    key_evidence: list[str]
    analyst_picks: dict[str, str] = field(default_factory=dict)  # analyst → name

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "name": self.name,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "key_evidence": self.key_evidence,
            "analyst_picks": self.analyst_picks,
        }


def run_speaker_panel(
    profiles: dict[str, Any],
    transcripts: list[tuple[Path, dict]],
    calendar_events: list[dict] | None,
    contacts: list[dict] | None,
    llm: Any,
    account_owner_name: str = "",
    min_calls: int = 1,
    min_segments: int = 5,
    max_speakers: int = 0,  # 0 = no limit
) -> list[PanelResult]:
    """Run the LLM expert panel on unidentified speakers.

    Args:
        profiles: Speaker profiles from identify_speakers().
        transcripts: List of (file_path, transcript_dict).
        calendar_events: Pre-fetched calendar events.
        contacts: Pre-fetched contacts list.
        llm: LLMProvider instance.
        account_owner_name: Name to exclude from candidates (lowercase).
        min_calls: Min calls for a speaker to be worth paneling.
        min_segments: Min speech segments across calls (quality gate).
        max_speakers: Max unknown speakers to analyze (cost control).

    Returns: List of PanelResult for successfully identified speakers.
    """
    if not llm:
        return []

    # Find unidentified speakers worth analyzing
    unknowns = []
    for cid, profile in profiles.items():
        if profile.likely_name:
            continue
        if profile.total_calls < min_calls:
            continue
        # Check segment count — need enough speech content for meaningful analysis
        samples = _gather_speech_samples(cid, transcripts)
        total_segs = sum(len(s["segments"]) for s in samples)
        if total_segs < min_segments:
            continue
        unknowns.append((cid, profile))

    unknowns.sort(key=lambda x: -x[1].total_calls)
    if max_speakers > 0:
        unknowns = unknowns[:max_speakers]

    if not unknowns:
        logger.info("Speaker panel: no unidentified speakers with >= %d calls", min_calls)
        return []

    logger.info("Speaker panel: analyzing %d unidentified speakers via LLM", len(unknowns))

    # Build candidate pool from calendar + contacts
    candidate_pool = _build_candidate_pool(
        profiles, calendar_events, contacts, account_owner_name
    )

    # Load the prompt template
    prompt_template = _load_prompt_template()

    results: list[PanelResult] = []
    for cid, profile in unknowns:
        # Gather this speaker's segments across calls
        speech_samples = _gather_speech_samples(cid, transcripts)
        if not speech_samples:
            continue

        # Find candidates specific to this speaker's calls
        candidates = _find_candidates_for_speaker(
            cid, profile, transcripts, calendar_events, candidate_pool, account_owner_name
        )
        if not candidates:
            continue

        # Build co-speaker names
        co_names = []
        for co_cid in sorted(profile.co_speakers, key=lambda c: -profile.co_speakers[c])[:5]:
            if co_cid in profiles and profiles[co_cid].likely_name:
                co_names.append(profiles[co_cid].likely_name.split(" (")[0])

        # Format prompt
        prompt = prompt_template.format(
            cluster_id=cid,
            call_count=profile.total_calls,
            speaking_minutes=profile.total_speaking_seconds / 60,
            co_speaker_names=", ".join(co_names) if co_names else "(unknown)",
            sample_count=len(speech_samples),
            speech_samples=_format_speech_samples(speech_samples),
            candidates=_format_candidates(candidates),
        )

        # Call LLM
        response = llm.complete(
            prompt=prompt,
            system="You are an expert identity analyst. Respond only with valid JSON.",
            max_tokens=1500,
        )

        if not response:
            continue

        # Parse response
        result = _parse_panel_response(cid, response)
        if result and result.confidence >= 40:
            results.append(result)
            logger.info(
                "Panel identified %s as %s (confidence %d%%): %s",
                cid, result.name, result.confidence, result.reasoning[:80]
            )

    return results


def _load_prompt_template() -> str:
    """Load the speaker panel prompt template."""
    template_path = Path(__file__).parent.parent / "llm" / "prompts" / "speaker_panel.txt"
    if template_path.exists():
        return template_path.read_text()
    raise FileNotFoundError(f"Speaker panel prompt not found: {template_path}")


def _build_candidate_pool(
    profiles: dict[str, Any],
    calendar_events: list[dict] | None,
    contacts: list[dict] | None,
    account_owner_name: str,
) -> dict[str, dict]:
    """Build a pool of candidate identities from calendar + contacts.

    Returns: {name_lower: {"name": str, "title": str, "company": str, "email": str}}
    """
    pool: dict[str, dict] = {}

    # Already-identified names (exclude from candidates)
    identified = set()
    for p in profiles.values():
        if p.likely_name:
            identified.add(p.likely_name.lower().split(" (")[0])

    # Calendar attendees
    if calendar_events:
        for event in calendar_events:
            for att in event.get("attendees", []):
                name = att.get("emailAddress", {}).get("name", "")
                email = att.get("emailAddress", {}).get("address", "")
                if not name:
                    continue
                name_lower = name.lower()
                if name_lower == account_owner_name:
                    continue
                if any(name_lower.startswith(idn) or idn.startswith(name_lower) for idn in identified):
                    continue
                if name_lower not in pool:
                    pool[name_lower] = {"name": name, "title": "", "company": "", "email": email}

    # Enrich with contact details
    if contacts:
        for c in contacts:
            cname = c.get("displayName", "")
            if not cname:
                continue
            cname_lower = cname.lower()
            if cname_lower in pool:
                pool[cname_lower]["title"] = c.get("jobTitle") or ""
                pool[cname_lower]["company"] = c.get("companyName") or ""
                if not pool[cname_lower]["email"]:
                    emails = c.get("emailAddresses") or []
                    if emails:
                        pool[cname_lower]["email"] = emails[0].get("address", "")

    return pool


def _find_candidates_for_speaker(
    cid: str,
    profile: Any,
    transcripts: list[tuple[Path, dict]],
    calendar_events: list[dict] | None,
    candidate_pool: dict[str, dict],
    account_owner_name: str,
) -> list[dict]:
    """Find candidate identities relevant to this specific speaker's calls."""
    from datetime import datetime, timedelta, timezone

    if not calendar_events:
        # Return top candidates from pool
        return list(candidate_pool.values())[:15]

    # Find calendar events matching this speaker's calls
    speaker_attendees: Counter = Counter()

    for file_path, transcript in transcripts:
        diar = transcript.get("diarization", {})
        resolved = diar.get("speakers_resolved", [])
        speaker_in_call = any(
            sr.get("speaker_cluster_id") == cid for sr in resolved
        )
        if not speaker_in_call:
            continue

        ct = transcript.get("metadata", {}).get("audio", {}).get("format_tags", {}).get("creation_time", "")
        if not ct:
            continue

        try:
            rec_dt = datetime.fromisoformat(ct.replace("Z", "+00:00"))
            if rec_dt.tzinfo is None:
                rec_dt = rec_dt.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            continue

        for event in calendar_events:
            evt_start = event.get("start", {}).get("dateTime", "")
            if not evt_start:
                continue
            try:
                evt_dt = datetime.fromisoformat(evt_start.replace("Z", "+00:00"))
                if evt_dt.tzinfo is None:
                    evt_dt = evt_dt.replace(tzinfo=timezone.utc)
                if abs((rec_dt - evt_dt).total_seconds()) < 7200:
                    for att in event.get("attendees", []):
                        name = att.get("emailAddress", {}).get("name", "")
                        if name and name.lower() != account_owner_name:
                            speaker_attendees[name] += 1
                    break
            except (ValueError, TypeError):
                continue

    # Return candidates sorted by frequency (most common attendees across this speaker's calls)
    candidates = []
    seen_names = set()
    for name, count in speaker_attendees.most_common(15):
        name_lower = name.lower()
        seen_names.add(name_lower)
        if name_lower in candidate_pool:
            entry = dict(candidate_pool[name_lower])
            entry["meeting_overlap"] = count
            candidates.append(entry)
        else:
            candidates.append({"name": name, "title": "", "company": "", "email": "", "meeting_overlap": count})

    # If no direct calendar matches, expand via co-speaker associations:
    # Find calendar attendees from meetings where this speaker's CO-SPEAKERS appear
    if not candidates and profile.co_speakers and calendar_events:
        co_attendees: Counter = Counter()
        for co_cid in list(profile.co_speakers.keys())[:5]:
            co_profile = profiles.get(co_cid)
            if not co_profile or not co_profile.likely_name:
                continue
            co_name = co_profile.likely_name.lower().split(" (")[0]

            for event in calendar_events:
                event_names = [
                    (a.get("emailAddress", {}).get("name") or "").lower()
                    for a in event.get("attendees", [])
                ]
                if any(co_name in en or en in co_name for en in event_names):
                    for att in event.get("attendees", []):
                        att_name = att.get("emailAddress", {}).get("name", "")
                        if att_name and att_name.lower() != account_owner_name:
                            att_lower = att_name.lower()
                            if att_lower not in seen_names:
                                # Skip already-identified speakers
                                already_id = any(
                                    att_lower.startswith(p.likely_name.lower().split(" (")[0])
                                    or p.likely_name.lower().split(" (")[0].startswith(att_lower)
                                    for p in profiles.values() if p.likely_name
                                )
                                if not already_id:
                                    co_attendees[att_name] += 1

        for name, count in co_attendees.most_common(10):
            name_lower = name.lower()
            if name_lower in candidate_pool:
                entry = dict(candidate_pool[name_lower])
                entry["meeting_overlap"] = count
                entry["source"] = "co-speaker association"
                candidates.append(entry)
            else:
                candidates.append({"name": name, "title": "", "company": "", "email": "", "meeting_overlap": count, "source": "co-speaker"})

        if candidates:
            logger.info("Expanded candidates for %s via co-speaker associations: %d", cid, len(candidates))

    return candidates


def _gather_speech_samples(
    cid: str,
    transcripts: list[tuple[Path, dict]],
) -> list[dict]:
    """Gather representative speech samples for an unknown speaker across calls.

    Returns: [{"call": str, "segments": [{"text": str, "start": float}]}]
    """
    samples = []

    for file_path, transcript in transcripts:
        diar = transcript.get("diarization", {})
        resolved = diar.get("speakers_resolved", [])
        if not any(sr.get("speaker_cluster_id") == cid for sr in resolved):
            continue

        segments = transcript.get("segments", [])
        speaker_segs = []
        for seg in segments:
            if seg.get("speaker_cluster_id") == cid:
                text = seg.get("text", "").strip()
                if text and len(text) > 10:
                    # Truncate long segments
                    words = text.split()
                    if len(words) > MAX_WORDS_PER_SEGMENT:
                        text = " ".join(words[:MAX_WORDS_PER_SEGMENT]) + "..."
                    speaker_segs.append({
                        "text": text,
                        "start": seg.get("start", 0),
                    })

        if speaker_segs:
            # Sample evenly across the call
            if len(speaker_segs) > MAX_SEGMENTS_PER_CALL:
                step = len(speaker_segs) // MAX_SEGMENTS_PER_CALL
                speaker_segs = speaker_segs[::step][:MAX_SEGMENTS_PER_CALL]

            llm_title = transcript.get("llm_analysis", {}).get("title", file_path.stem)
            samples.append({
                "call": llm_title or file_path.stem,
                "segments": speaker_segs,
            })

    # Limit total calls
    if len(samples) > MAX_CALLS:
        samples.sort(key=lambda s: -len(s["segments"]))
        samples = samples[:MAX_CALLS]

    return samples


def _format_speech_samples(samples: list[dict]) -> str:
    """Format speech samples for the prompt."""
    lines = []
    for i, sample in enumerate(samples, 1):
        lines.append(f"### Call {i}: {sample['call']}")
        for seg in sample["segments"]:
            mins = int(seg["start"]) // 60
            secs = int(seg["start"]) % 60
            lines.append(f"  [{mins:02d}:{secs:02d}] {seg['text']}")
        lines.append("")
    return "\n".join(lines)


def _format_candidates(candidates: list[dict]) -> str:
    """Format candidates for the prompt."""
    lines = []
    for c in candidates:
        parts = [c["name"]]
        if c.get("title"):
            parts.append(c["title"])
        if c.get("company"):
            parts.append(f"at {c['company']}")
        if c.get("email"):
            parts.append(f"({c['email']})")
        if c.get("meeting_overlap"):
            parts.append(f"[in {c['meeting_overlap']} matching meetings]")
        lines.append(f"- {', '.join(parts)}")
    return "\n".join(lines) if lines else "(no candidates found)"


@dataclass
class LabelResult:
    """Descriptive label for an unidentified speaker."""
    cluster_id: str
    label: str  # "Estate Attorney", "Family Member", etc.
    category: str  # family, advisor, business, client, friend, colleague
    reasoning: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "label": self.label,
            "category": self.category,
            "reasoning": self.reasoning,
        }


def label_unknown_speakers(
    profiles: dict[str, Any],
    transcripts: list[tuple[Path, dict]],
    llm: Any,
    min_segments: int = 5,
) -> list[LabelResult]:
    """Generate descriptive labels for speakers that couldn't be identified.

    Instead of leaving them as spk_xxx, give them labels like
    "Estate Attorney", "Family Member", "Tech Founder".
    """
    if not llm:
        return []

    template_path = Path(__file__).parent.parent / "llm" / "prompts" / "speaker_label.txt"
    if not template_path.exists():
        return []
    template = template_path.read_text()

    # Find unlabeled speakers
    unknowns = []
    for cid, profile in profiles.items():
        if profile.likely_name:
            continue
        samples = _gather_speech_samples(cid, transcripts)
        total_segs = sum(len(s["segments"]) for s in samples)
        if total_segs >= min_segments:
            unknowns.append((cid, profile, samples))

    if not unknowns:
        return []

    logger.info("Labeling %d unidentified speakers via LLM", len(unknowns))

    results: list[LabelResult] = []
    for cid, profile, samples in unknowns:
        co_names = []
        for co_cid in sorted(profile.co_speakers, key=lambda c: -profile.co_speakers[c])[:5]:
            if co_cid in profiles and profiles[co_cid].likely_name:
                co_names.append(profiles[co_cid].likely_name.split(" (")[0])

        prompt = template.format(
            call_count=profile.total_calls,
            speaking_minutes=profile.total_speaking_seconds / 60,
            co_speaker_names=", ".join(co_names) if co_names else "(unknown)",
            speech_samples=_format_speech_samples(samples),
        )

        response = llm.complete(prompt=prompt, system="Respond only with valid JSON.", max_tokens=300)
        if not response:
            continue

        try:
            text = response.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                text = text.rsplit("```", 1)[0]
            data = json.loads(text)

            label = data.get("label", "").strip()
            category = data.get("category", "unknown").strip()
            reasoning = data.get("reasoning", "").strip()

            # Prefer the combined label, fall back to role + topics
            if not label or label.lower() in ("speaker", "unknown", "unknown person", "participant"):
                role = data.get("role", "").strip()
                topics = data.get("topics", "").strip()
                if role and topics:
                    label = f"{role} — {topics}"
                elif role:
                    label = role

            if label and label.lower() not in ("speaker", "unknown", "unknown person", "participant"):
                results.append(LabelResult(
                    cluster_id=cid,
                    label=label,
                    category=category,
                    reasoning=reasoning,
                ))
                logger.info("Labeled %s as '%s' (%s)", cid, label, category)
        except (json.JSONDecodeError, KeyError):
            continue

    return results


def _parse_panel_response(cid: str, response: str) -> PanelResult | None:
    """Parse the LLM panel response JSON."""
    try:
        # Handle markdown-wrapped JSON
        text = response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            text = text.rsplit("```", 1)[0]

        data = json.loads(text)
        consensus = data.get("consensus", {})
        name = consensus.get("name", "")
        confidence = consensus.get("confidence", 0)

        if not name or name.upper() == "UNKNOWN":
            return None

        return PanelResult(
            cluster_id=cid,
            name=name,
            confidence=confidence,
            reasoning=consensus.get("reasoning", ""),
            key_evidence=consensus.get("key_evidence", []),
            analyst_picks={
                "conversation": data.get("analyst_1", {}).get("top_pick", ""),
                "professional": data.get("analyst_2", {}).get("top_pick", ""),
                "network": data.get("analyst_3", {}).get("top_pick", ""),
            },
        )
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning("Failed to parse panel response for %s: %s", cid, e)
        return None
