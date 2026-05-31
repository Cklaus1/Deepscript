"""Import transcripts from external meeting services (Circleback, Fireflies).

Fetches transcripts via BFlow's MCP clients, converts to DeepScript's
native JSON format, and saves for analysis.

Usage:
    deepscript import --source fireflies --days 30 --output-dir ./transcripts
    deepscript import --source circleback --days 7
    deepscript import --source fireflies --meeting-id abc123
    deepscript import --source fireflies --from-date 2025-02-01
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ImportResult:
    """Result of importing transcripts."""
    source: str
    imported: int
    skipped: int
    failed: int
    files: list[str] = field(default_factory=list)


# --- Format Converters ---


def fireflies_to_deepscript(
    meeting: dict,
    transcript_text: str,
    summary_data: dict | str | None = None,
    email_to_name: dict[str, str] | None = None,
    fallback_title: str | None = None,
) -> dict:
    """Convert a Fireflies meeting (transcript + summary) to DeepScript format.

    Args:
        meeting: Entry from fireflies_get_transcripts list.
        transcript_text: Raw text from fireflies_get_transcript (Speaker: text format).
        summary_data: Parsed summary from fireflies_get_summary.
        email_to_name: Email -> name mapping.
        fallback_title: Filename-based title when meeting title is missing.

    Returns: DeepScript-format transcript dict.
    """
    if email_to_name is None:
        email_to_name = {}

    title = meeting.get("title") or fallback_title or "untitled"
    date_str = meeting.get("dateString", "")
    duration_min = meeting.get("duration", 0) or 0
    meeting_id = meeting.get("id", "")

    # Parse transcript text into segments
    segments, speaker_names = _parse_fireflies_transcript(transcript_text)

    # Resolve attendee emails to names
    attendees = meeting.get("meetingAttendees") or []
    resolved_attendees = []
    for att in attendees:
        email = (att.get("email") or "").lower()
        name = att.get("displayName") or email_to_name.get(email, "")
        if not name and email:
            name = email.split("@")[0].replace(".", " ").title()
        resolved_attendees.append({"name": name, "email": email})

    # Merge speaker names from transcript with attendee names
    all_speakers = list(
        set(speaker_names + [a["name"] for a in resolved_attendees if a["name"]])
    )

    # Parse summary
    if summary_data is None:
        summary_data = {}
    if isinstance(summary_data, str):
        try:
            summary_data = json.loads(summary_data)
        except json.JSONDecodeError:
            summary_data = {"raw": summary_data}

    summary_fields = _extract_summary_fields(summary_data)

    # Build DeepScript format
    full_text = (
        " ".join(s["text"] for s in segments) if segments else summary_fields.get("short_summary", "")
    )

    return {
        "text": full_text,
        "language": "en",
        "segments": segments,
        "diarization": {
            "num_speakers": len(all_speakers),
            "speakers_resolved": [
                {
                    "local_label": f"SPEAKER_{i:02d}",
                    "display_name": name,
                    "speaker_cluster_id": "",
                    "status": "confirmed",
                }
                for i, name in enumerate(all_speakers)
            ],
        },
        "metadata": {
            "audio": {
                "duration_seconds": duration_min * 60,
                "format_tags": {"creation_time": date_str},
            },
            "file": {
                "name": f"{_sanitize(title)}.json",
                "extension": "json",
            },
            "source": "fireflies",
            "meeting_id": meeting_id,
            "transcript_url": f"https://app.fireflies.ai/view/{meeting_id}",
        },
        "llm_analysis": {
            "title": title,
            "summary": summary_fields.get("short_summary", ""),
            "overview": summary_fields.get("overview", ""),
            "notes": summary_fields.get("notes", ""),
            "keywords": summary_fields.get("keywords", []),
            "action_items_raw": summary_fields.get("action_items", ""),
            "action_items": _parse_action_items(summary_fields.get("action_items", "")),
            "bullet_gist": summary_fields.get("bullet_gist", ""),
            "speakers": [
                {"label": name, "likely_name": name, "evidence": "From Fireflies", "role": ""}
                for name in all_speakers
            ],
        },
    }


def circleback_to_deepscript(meeting: dict, fallback_title: str | None = None) -> dict:
    """Convert Circleback meeting data to DeepScript format.

    Circleback returns:
        transcript: [{speaker, text, timestamp}]
        attendees: [{name, email}]
        notes: str
        actionItems: [{title, description, assignee: {name}, status}]
    """
    transcript_entries = meeting.get("transcript", [])
    attendees = meeting.get("attendees", [])
    title = meeting.get("name") or fallback_title or ""
    duration = meeting.get("duration", 0)  # Already in seconds
    created = meeting.get("createdAt", "")

    # Parse timestamps and build segments
    segments = []
    for i, entry in enumerate(transcript_entries):
        start_sec = _parse_timestamp(entry.get("timestamp", ""))
        if i + 1 < len(transcript_entries):
            end_sec = _parse_timestamp(transcript_entries[i + 1].get("timestamp", ""))
            if end_sec <= start_sec:
                end_sec = start_sec + 10
        else:
            end_sec = start_sec + 10

        segments.append({
            "id": i,
            "start": start_sec,
            "end": end_sec,
            "text": entry.get("text", ""),
            "speaker": entry.get("speaker", ""),
            "speaker_cluster_id": "",
            "confidence": 1.0,
            "no_speech_prob": 0.0,
            "words": [],
        })

    # Build full text
    full_text = " ".join(s["text"] for s in segments)

    # Build diarization from attendees + transcript speakers
    speaker_names = list(
        {entry.get("speaker", "") for entry in transcript_entries if entry.get("speaker")}
    )
    speakers_resolved = []
    for i, name in enumerate(speaker_names):
        email = ""
        for att in attendees:
            if att.get("name", "").lower() == name.lower():
                email = att.get("email", "")
                break
        speakers_resolved.append({
            "local_label": f"SPEAKER_{i:02d}",
            "display_name": name,
            "speaker_cluster_id": "",
            "status": "confirmed",
            "email": email,
        })

    # Extract action items
    action_items = []
    for ai in meeting.get("actionItems", []):
        action_items.append({
            "text": ai.get("title") or ai.get("description", ""),
            "assignee": (ai.get("assignee") or {}).get("name", ""),
            "status": ai.get("status", ""),
        })

    return {
        "text": full_text,
        "language": "en",
        "segments": segments,
        "diarization": {
            "num_speakers": len(speaker_names),
            "speakers_resolved": speakers_resolved,
        },
        "metadata": {
            "audio": {
                "duration_seconds": duration,
                "format_tags": {"creation_time": created},
            },
            "file": {
                "name": f"{_sanitize(title)}.json",
                "extension": "json",
            },
            "source": "circleback",
            "meeting_id": meeting.get("id", ""),
            "meeting_url": meeting.get("url", ""),
        },
        "llm_analysis": {
            "title": title,
            "speakers": [
                {"label": name, "likely_name": name, "evidence": "From Circleback", "role": ""}
                for name in speaker_names
            ],
            "action_items": action_items,
            "summary": meeting.get("notes", ""),
        },
    }


# --- Fetchers ---


async def _fetch_fireflies(
    days: int = 30,
    meeting_id: str | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
    limit: int = 50,
    max_pages: int = 50,
) -> list[dict]:
    """Fetch transcripts from Fireflies via BFlow MCP client.

    Falls back to scanning transcripts/fireflies/ for pre-fetched meeting
    metadata JSONs when BFlow is unavailable.
    """
    try:
        from concierge.fireflies_mcp_client import (
            fireflies_list_meetings,
            fireflies_get_transcript,
        )
    except ImportError:
        logger.warning("BFlow not available — falling back to local transcripts/fireflies/")
        return _fetch_fireflies_local(
            days=days, meeting_id=meeting_id,
            from_date=from_date, to_date=to_date,
        )

    meetings = []

    if meeting_id:
        result = await fireflies_get_transcript(meeting_id)
        if result.get("success"):
            tx_text = result.get("result", "")
            if isinstance(tx_text, dict):
                tx_text = json.dumps(tx_text)
            meetings.append({"meeting": {"id": meeting_id}, "transcript_text": tx_text})
    else:
        list_result = await fireflies_list_meetings(
            limit=min(limit, 50),
            from_date=from_date,
            to_date=to_date,
            days=days if not from_date else 30,
            max_pages=max_pages,
        )
        if not list_result.get("success"):
            logger.error("Failed to list Fireflies meetings: %s", list_result.get("error"))
            return []

        meetings_raw = list_result.get("result", [])
        if isinstance(meetings_raw, str):
            try:
                meetings_raw = json.loads(meetings_raw)
            except json.JSONDecodeError:
                return []

        for m in meetings_raw:
            mid = m.get("id", "")
            if not mid:
                continue
            tx_result = await fireflies_get_transcript(mid)
            tx_text = ""
            if tx_result.get("success"):
                tx_text = tx_result.get("result", "")
                if isinstance(tx_text, dict):
                    tx_text = json.dumps(tx_text)
            meetings.append({"meeting": m, "transcript_text": tx_text})

    return meetings


def _fetch_fireflies_local(
    days: int = 30,
    meeting_id: str | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
) -> list[tuple[str, dict]]:
    """Fallback: scan transcripts/fireflies/ for pre-fetched meeting metadata.

    Each JSON file in the directory is expected to contain a Fireflies meeting
    object (with id, title, dateString, etc.) — the kind that would normally
    come from fireflies_list_meetings.  The actual transcript text must be
    fetched separately (e.g. via fetch_fireflies_direct.py) and stored as a
    parallel .transcript.txt file, or the meeting dict must already contain
    a "summary" key with the raw transcript text.

    Files already in DeepScript format (have segments[]) are returned as-is
    so the import loop can copy them directly.

    Returns list of (fallback_title, meeting_dict) tuples where fallback_title
    is the filename stem (used when meeting.title is None).
    """
    ff_dir = Path("transcripts/fireflies")
    if not ff_dir.is_dir():
        logger.warning("No local Fireflies directory at %s", ff_dir)
        return []

    cutoff = datetime.now(timezone.utc)
    if from_date:
        cutoff = datetime.strptime(from_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    elif days:
        cutoff = cutoff.replace(hour=0, minute=0, second=0)
        from datetime import timedelta
        cutoff = cutoff - timedelta(days=days)

    meetings = []
    for fname in sorted(ff_dir.iterdir()):
        if not fname.name.endswith(".json"):
            continue

        # If a specific meeting_id was requested, only fetch that one
        if meeting_id and meeting_id not in fname.stem:
            continue

        try:
            with open(fname) as f:
                meeting = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Skipping %s: %s", fname.name, e)
            continue

        # Check date filter — try JSON field first, then filename
        date_str = meeting.get("dateString", "")
        if not date_str:
            # Try to parse from filename: YYYY-MM-DD_...
            name_match = re.match(r"(\d{4}-\d{2}-\d{2})", fname.stem)
            if name_match:
                date_str = name_match.group(1)
        if date_str:
            try:
                m_date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                if m_date < cutoff:
                    continue
            except ValueError:
                pass  # Can't parse date, include anyway

        # If already in DeepScript format (has segments), return as-is
        if meeting.get("segments") and len(meeting.get("segments", [])) > 1:
            meetings.append(meeting)
            continue

        # Look for transcript text in a parallel .transcript.txt file
        tx_file = fname.with_suffix(".json.transcript.txt")
        tx_text = ""
        if tx_file.is_file():
            try:
                tx_text = tx_file.read_text(encoding="utf-8", errors="replace")
            except OSError:
                pass

        fallback_title = fname.stem
        meetings.append((fallback_title, {"meeting": meeting, "transcript_text": tx_text}))

    return meetings


async def _fetch_circleback(
    days: int = 30,
    meeting_id: str | None = None,
) -> list[dict]:
    """Fetch transcripts from Circleback via BFlow MCP client.

    Falls back to scanning transcripts/circleback/ for pre-fetched meeting
    JSONs when BFlow is unavailable.
    """
    try:
        from concierge.circleback_mcp_client import (
            circleback_list_meetings,
            circleback_get_meeting,
            circleback_get_transcript,
        )
    except ImportError:
        logger.warning("BFlow not available — falling back to local transcripts/circleback/")
        return _fetch_circleback_local(
            days=days, meeting_id=meeting_id,
        )

    meetings = []

    if meeting_id:
        result = await circleback_get_meeting(int(meeting_id))
        if result.get("success"):
            meeting = _parse_mcp_result(result)
            tx_result = await circleback_get_transcript(int(meeting_id))
            if tx_result.get("success"):
                if isinstance(meeting, dict):
                    meeting["transcript"] = _parse_mcp_result(tx_result)
            meetings.append(meeting)
    else:
        list_result = await circleback_list_meetings(days=days)
        if not list_result.get("success"):
            logger.error("Failed to list Circleback meetings: %s", list_result.get("error"))
            return []

        meeting_list = _parse_mcp_result(list_result)
        if isinstance(meeting_list, str):
            try:
                meeting_list = json.loads(meeting_list)
            except json.JSONDecodeError:
                meeting_list = []

        if isinstance(meeting_list, dict):
            meeting_list = meeting_list.get("meetings", [])

        for m in meeting_list:
            mid = m.get("id", "")
            if mid:
                result = await circleback_get_meeting(int(mid))
                if result.get("success"):
                    full = _parse_mcp_result(result)
                    tx_result = await circleback_get_transcript(int(mid))
                    if tx_result.get("success") and isinstance(full, dict):
                        full["transcript"] = _parse_mcp_result(tx_result)
                    meetings.append(full)

    return meetings


def _fetch_circleback_local(
    days: int = 30,
    meeting_id: str | None = None,
) -> list[dict]:
    """Fallback: scan transcripts/circleback/ for pre-fetched meeting JSONs.

    Files may already be in DeepScript format (have segments[]) — in that case
    they are returned as-is so the import loop can copy them directly.
    Files that are raw Circleback metadata (no segments) but have a parallel
    .transcript.txt are converted on-the-fly.
    """
    cb_dir = Path("transcripts/circleback")
    if not cb_dir.is_dir():
        logger.warning("No local Circleback directory at %s", cb_dir)
        return []

    cutoff = datetime.now(timezone.utc)
    if days:
        cutoff = cutoff.replace(hour=0, minute=0, second=0)
        from datetime import timedelta
        cutoff = cutoff - timedelta(days=days)

    meetings = []
    for fname in sorted(cb_dir.iterdir()):
        if not fname.name.endswith(".json"):
            continue

        if meeting_id and str(meeting_id) not in fname.stem:
            continue

        try:
            with open(fname) as f:
                meeting = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Skipping %s: %s", fname.name, e)
            continue

        # Check date filter — try JSON field first, then filename
        meeting_date = meeting.get("meeting_date", "") or meeting.get("date", "")
        if not meeting_date:
            name_match = re.match(r"(\d{4}-\d{2}-\d{2})", fname.stem)
            if name_match:
                meeting_date = name_match.group(1)
        if meeting_date:
            try:
                m_date = datetime.strptime(meeting_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                if m_date < cutoff:
                    continue
            except ValueError:
                pass

        # If already in DeepScript format (has segments), return as-is
        if meeting.get("segments") and len(meeting["segments"]) > 1:
            meetings.append(meeting)
            continue

        # Check for transcript in a parallel .transcript.txt file
        tx_file = fname.with_suffix(".json.transcript.txt")
        if tx_file.is_file():
            try:
                tx_text = tx_file.read_text(encoding="utf-8", errors="replace")
                if isinstance(meeting, dict):
                    meeting["transcript"] = tx_text
            except OSError:
                pass

        meetings.append((fname.stem, meeting))

    return meetings


# --- Main Import Function ---


def import_transcripts(
    source: str,
    output_dir: str | Path = "transcripts/imported",
    days: int = 30,
    meeting_id: str | None = None,
    from_date: str | None = None,
    to_date: str | None = None,
    limit: int = 50,
    max_pages: int = 50,
) -> ImportResult:
    """Import transcripts from an external service.

    Args:
        source: "fireflies" or "circleback"
        output_dir: Where to save converted transcripts.
        days: How many days back to fetch (used if from_date not set).
        meeting_id: Specific meeting ID to fetch (overrides days).
        from_date: Start date for Fireflies (YYYY-MM-DD).
        to_date: End date for Fireflies (YYYY-MM-DD).
        limit: Max meetings per page for Fireflies (max 50).
        max_pages: Max pages to fetch for Fireflies pagination.

    Returns: ImportResult with counts and file paths.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Fetch from service
    if source == "fireflies":
        raw_meetings = asyncio.run(
            _fetch_fireflies(
                days=days, meeting_id=meeting_id,
                from_date=from_date, to_date=to_date,
                limit=limit, max_pages=max_pages,
            )
        )
    elif source == "circleback":
        raw_meetings = asyncio.run(_fetch_circleback(days=days, meeting_id=meeting_id))
    else:
        logger.error("Unknown source: %s. Use 'fireflies' or 'circleback'", source)
        return ImportResult(source=source, imported=0, skipped=0, failed=0, files=[])

    if not raw_meetings:
        logger.info("No meetings found from %s (last %d days)", source, days)
        return ImportResult(source=source, imported=0, skipped=0, failed=0, files=[])

    # Convert and save
    imported = 0
    skipped = 0
    failed = 0
    files = []

    for raw in raw_meetings:
        try:
            # Unpack (stem, meeting) from fallback, or raw dict from MCP
            if source == "fireflies":
                if isinstance(raw, tuple):
                    fallback_title, raw = raw
                else:
                    fallback_title = None
                # If already in DeepScript format (has segments), use as-is
                if raw.get("segments") and len(raw.get("segments", [])) > 1:
                    converted = raw
                else:
                    m = raw.get("meeting", raw)
                    tx_text = raw.get("transcript_text", "")
                    converted = fireflies_to_deepscript(
                        meeting=m,
                        transcript_text=tx_text,
                        summary_data=m.get("summary"),
                        email_to_name=_load_email_map(),
                        fallback_title=fallback_title,
                    )
            else:
                if isinstance(raw, tuple):
                    fallback_title, raw = raw
                else:
                    fallback_title = None
                if isinstance(raw, str):
                    try:
                        raw = json.loads(raw)
                    except json.JSONDecodeError:
                        failed += 1
                        continue
                # If already in DeepScript format (has segments), use as-is
                if raw.get("segments") and len(raw.get("segments", [])) > 1:
                    converted = raw
                else:
                    converted = circleback_to_deepscript(raw, fallback_title=fallback_title)

            # Check if we got any segments
            if not converted.get("segments"):
                skipped += 1
                continue

            # Save — disambiguate identical meeting titles with date + short id
            base_name = converted["metadata"]["file"]["name"]
            if base_name.endswith(".json"):
                base_name = base_name[:-5]
            base_name = _sanitize(base_name)
            meta = converted.get("metadata", {})
            ct = (meta.get("audio", {}).get("format_tags", {}) or {}).get("creation_time", "")
            date_prefix = ct[:10] if ct and len(ct) >= 10 else ""
            mid = str(meta.get("meeting_id", "") or "")
            id_suffix = mid[:12] if mid else ""
            parts = [p for p in (date_prefix, base_name, id_suffix) if p]
            filename = "_".join(parts) + ".json"
            filepath = output_path / filename

            # Don't overwrite existing
            if filepath.exists():
                skipped += 1
                continue

            with open(filepath, "w") as f:
                json.dump(converted, f, indent=2, default=str)

            files.append(str(filepath))
            imported += 1
            seg_count = len(converted["segments"])
            logger.info("Imported: %s (%d segments)", filename, seg_count)

        except Exception as e:
            logger.warning("Failed to import meeting: %s", e)
            failed += 1

    return ImportResult(
        source=source,
        imported=imported,
        skipped=skipped,
        failed=failed,
        files=files,
    )


# --- Helpers ---


def _parse_fireflies_transcript(text: str) -> tuple[list[dict], list[str]]:
    """Parse Fireflies 'Speaker: text' format into segments.

    Returns: (segments, speaker_names)
    """
    segments = []
    speaker_names_set = set()

    if not text:
        return segments, []

    lines = text.strip().split("\n")
    time_estimate = 0.0
    seg_id = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Match "Speaker Name: text"
        m = re.match(r'^([A-Z][^:]{1,50}):\s+(.+)$', line)
        if not m:
            continue

        speaker = m.group(1).strip()
        seg_text = m.group(2).strip()

        # Skip metadata lines
        if speaker in (
            "Id", "Title", "DateString", "Privacy", "Speakers",
            "Host Email", "Organizer Email", "Calendar Id",
            "Fireflies Users", "Participants", "Date",
            "Transcript Url", "Audio Url", "Video Url",
            "Duration", "Meeting Attendees", "Cal Id",
            "Calendar Type", "Meeting Link", "Is Live",
            "Sentences",
        ):
            continue

        speaker_names_set.add(speaker)

        # Estimate timing (~2.5 words/sec)
        words = len(seg_text.split())
        duration = max(1.0, words / 2.5)

        segments.append({
            "id": seg_id,
            "start": round(time_estimate, 1),
            "end": round(time_estimate + duration, 1),
            "text": seg_text,
            "speaker": speaker,
            "speaker_cluster_id": "",
            "confidence": 1.0,
            "no_speech_prob": 0.0,
            "words": [],
        })

        time_estimate += duration
        seg_id += 1

    return segments, sorted(speaker_names_set)


def _parse_timestamp(ts: Any) -> float:
    """Parse timestamp to seconds. Handles float, int, HH:MM:SS, and MM:SS."""
    if not ts and ts != 0:
        return 0.0
    if isinstance(ts, (int, float)):
        return float(ts)
    ts_str = str(ts)
    parts = ts_str.split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        else:
            return float(ts_str)
    except (ValueError, IndexError):
        return 0.0


def _sanitize(name: str) -> str:
    clean = re.sub(r'[<>:"/\\|?*]', '', name)
    return re.sub(r'\s+', ' ', clean).strip()[:80]


def _parse_mcp_result(result: dict) -> Any:
    """Parse the result from an MCP tool call."""
    data = result.get("result", "")
    if isinstance(data, str):
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return data
    return data


def _extract_summary_fields(data: Any) -> dict:
    """Extract summary fields from various Fireflies response formats."""
    if isinstance(data, str):
        return {"short_summary": data}

    if not isinstance(data, dict):
        return {}

    result = {}

    # Direct fields
    for key in ("short_summary", "overview", "notes", "keywords",
                "action_items", "bullet_gist", "gist", "shorthand_bullet"):
        if key in data:
            result[key] = data[key]

    # Nested in summary object
    summary = data.get("summary", {})
    if isinstance(summary, dict):
        for key in ("short_summary", "overview", "notes", "keywords",
                    "action_items", "bullet_gist", "gist"):
            if key in summary and key not in result:
                result[key] = summary[key]
    elif isinstance(summary, str):
        # Parse concatenated summary string
        if "Keywords:" in summary:
            parts = summary.split("Keywords:", 1)
            remainder = parts[1] if len(parts) > 1 else ""
            if "Action Items:" in remainder:
                kw_part, ai_part = remainder.split("Action Items:", 1)
                result["keywords"] = [k.strip() for k in kw_part.split(",") if k.strip()]
                result["action_items"] = ai_part.strip()

        if "Short Summary:" in summary:
            result["short_summary"] = summary.split("Short Summary:", 1)[1].strip()

    return result


def _parse_action_items(raw: str) -> list[dict]:
    """Parse Fireflies action items text into structured format.

    Input format:
        **Mahima Kaul**
        Send email to Lainey... (09:25)
        Follow up with Seth... (15:05)

        **Chris Klaus**
        Introduce Carter to Adi... (10:59)
    """
    if not raw:
        return []

    items = []
    current_assignee = ""

    for line in raw.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        # Assignee header: **Name**
        m = re.match(r'\*\*(.+?)\*\*', line)
        if m:
            current_assignee = m.group(1).strip()
            continue

        # Action item line (may end with timestamp)
        if current_assignee and len(line) > 5:
            ts_match = re.search(r'\((\d{1,2}:\d{2})\)\s*$', line)
            timestamp = ts_match.group(1) if ts_match else ""
            text = line[:ts_match.start()].strip() if ts_match else line

            items.append({
                "text": text,
                "assignee": current_assignee,
                "timestamp": timestamp,
            })

    return items


def _load_email_map() -> dict[str, str]:
    """Load email -> name mapping from contacts cache."""
    cache = Path("/tmp/ms365-contacts.json")
    if not cache.exists():
        return {}

    try:
        with open(cache) as f:
            contacts = json.load(f)
        mapping = {}
        for c in contacts:
            name = c.get("displayName", "")
            for ea in (c.get("emailAddresses") or []):
                addr = ea.get("address", "").lower()
                if addr and name:
                    mapping[addr] = name
        # Add known team
        mapping.update({
            "cklaus@fusen.world": "Chris Klaus",
            "jwhitlow@fusen.world": "Jennifer Whitlow",
            "mkaul@fusen.world": "Mahima Kaul",
            "bwhatley@fusen.world": "Ben Whatley",
            "jlee@fusen.world": "Junseob Lee",
            "rrogers@fusen.world": "Ryan Rogers",
            "jboduch@fusen.world": "Jason Boduch",
            "cerven@fusen.world": "Chris Erven",
        })
        return mapping
    except Exception:
        return {}