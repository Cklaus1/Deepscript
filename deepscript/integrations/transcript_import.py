"""Import transcripts from external meeting services (Circleback, Fireflies).

Fetches transcripts via BFlow's MCP clients, converts to DeepScript's
native JSON format, and saves for analysis.

Usage:
    deepscript import --source fireflies --days 30 --output-dir ./transcripts
    deepscript import --source circleback --days 7
    deepscript import --source fireflies --meeting-id abc123
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
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
    files: list[str]


# --- Format Converters ---

def fireflies_to_deepscript(meeting: dict, transcript_data: dict) -> dict:
    """Convert Fireflies transcript to DeepScript format.

    Fireflies returns:
        sentences: [{speaker_id, speaker_name, text, start_time, end_time}]
        speakers: [{id, name}]
    """
    sentences = transcript_data.get("sentences", [])
    speakers_raw = transcript_data.get("speakers", [])
    title = transcript_data.get("title") or meeting.get("title", "")
    duration = meeting.get("duration", 0) * 60  # Fireflies returns minutes

    # Build segments
    segments = []
    for i, sent in enumerate(sentences):
        segments.append({
            "id": i,
            "start": float(sent.get("start_time", 0)),
            "end": float(sent.get("end_time", 0)),
            "text": sent.get("text", ""),
            "speaker": sent.get("speaker_name", ""),
            "speaker_cluster_id": "",
            "confidence": 1.0,
            "no_speech_prob": 0.0,
            "words": [],
        })

    # Build full text
    full_text = " ".join(s["text"] for s in segments)

    # Build diarization
    speaker_names = list({sent.get("speaker_name", "") for sent in sentences if sent.get("speaker_name")})
    speakers_resolved = []
    for i, name in enumerate(speaker_names):
        speakers_resolved.append({
            "local_label": f"SPEAKER_{i:02d}",
            "display_name": name,
            "speaker_cluster_id": "",
            "status": "confirmed" if name else "unknown",
        })

    # Build metadata
    meeting_date = meeting.get("date", "") or transcript_data.get("date", "")

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
                "format_tags": {
                    "creation_time": meeting_date,
                },
            },
            "file": {
                "name": f"{_sanitize_filename(title)}.json",
                "size_human": "",
                "extension": "json",
            },
            "source": "fireflies",
            "meeting_id": meeting.get("id", ""),
        },
        "llm_analysis": {
            "title": title,
            "speakers": [
                {"label": name, "likely_name": name, "evidence": "From Fireflies speaker ID", "role": ""}
                for name in speaker_names
            ],
        },
    }


def circleback_to_deepscript(meeting: dict) -> dict:
    """Convert Circleback meeting data to DeepScript format.

    Circleback returns:
        transcript: [{speaker, text, timestamp}]
        attendees: [{name, email}]
        notes: str
        actionItems: [{title, description, assignee: {name}, status}]
    """
    transcript_entries = meeting.get("transcript", [])
    attendees = meeting.get("attendees", [])
    title = meeting.get("name", "")
    duration = meeting.get("duration", 0)  # Already in seconds
    created = meeting.get("createdAt", "")

    # Parse timestamps and build segments
    segments = []
    for i, entry in enumerate(transcript_entries):
        start_sec = _parse_timestamp(entry.get("timestamp", ""))
        # Estimate end time from next segment or add 10 seconds
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
    speaker_names = list({entry.get("speaker", "") for entry in transcript_entries if entry.get("speaker")})
    speakers_resolved = []
    for i, name in enumerate(speaker_names):
        # Try to match with attendee for email
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
                "format_tags": {
                    "creation_time": created,
                },
            },
            "file": {
                "name": f"{_sanitize_filename(title)}.json",
                "size_human": "",
                "extension": "json",
            },
            "source": "circleback",
            "meeting_id": meeting.get("id", ""),
            "meeting_url": meeting.get("url", ""),
        },
        "llm_analysis": {
            "title": title,
            "speakers": [
                {"label": name, "likely_name": name, "evidence": "From Circleback speaker ID", "role": ""}
                for name in speaker_names
            ],
            "action_items": action_items,
            "summary": meeting.get("notes", ""),
        },
    }


# --- Fetchers ---

async def _fetch_fireflies(days: int = 30, meeting_id: str | None = None) -> list[dict]:
    """Fetch transcripts from Fireflies via BFlow MCP client."""
    try:
        from bflow.concierge.fireflies_mcp_client import (
            fireflies_list_meetings,
            fireflies_get_transcript,
        )
    except ImportError:
        logger.error("BFlow not available. Install bflow or run from BTask project.")
        return []

    meetings = []

    if meeting_id:
        result = await fireflies_get_transcript(meeting_id)
        if result.get("success"):
            transcript = _parse_mcp_result(result)
            meetings.append({"id": meeting_id, "transcript": transcript})
    else:
        list_result = await fireflies_list_meetings(limit=50, days=days)
        if not list_result.get("success"):
            logger.error("Failed to list Fireflies meetings: %s", list_result.get("error"))
            return []

        meeting_list = _parse_mcp_result(list_result)
        if isinstance(meeting_list, str):
            try:
                meeting_list = json.loads(meeting_list)
            except json.JSONDecodeError:
                meeting_list = []

        if isinstance(meeting_list, dict):
            meeting_list = meeting_list.get("transcripts", meeting_list.get("meetings", []))

        for m in meeting_list:
            mid = m.get("id", "")
            if mid:
                result = await fireflies_get_transcript(mid)
                if result.get("success"):
                    transcript = _parse_mcp_result(result)
                    meetings.append({"meeting": m, "transcript": transcript})

    return meetings


async def _fetch_circleback(days: int = 30, meeting_id: str | None = None) -> list[dict]:
    """Fetch transcripts from Circleback via BFlow MCP client."""
    try:
        from bflow.concierge.circleback_mcp_client import (
            circleback_list_meetings,
            circleback_get_meeting,
            circleback_get_transcript,
        )
    except ImportError:
        logger.error("BFlow not available. Install bflow or run from BTask project.")
        return []

    meetings = []

    if meeting_id:
        result = await circleback_get_meeting(int(meeting_id))
        if result.get("success"):
            meeting = _parse_mcp_result(result)
            # Also get transcript
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


# --- Main Import Function ---

def import_transcripts(
    source: str,
    output_dir: str | Path = "transcripts/imported",
    days: int = 30,
    meeting_id: str | None = None,
) -> ImportResult:
    """Import transcripts from an external service.

    Args:
        source: "fireflies" or "circleback"
        output_dir: Where to save converted transcripts.
        days: How many days back to fetch.
        meeting_id: Specific meeting ID to fetch (overrides days).

    Returns: ImportResult with counts and file paths.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Fetch from service
    if source == "fireflies":
        raw_meetings = asyncio.run(_fetch_fireflies(days=days, meeting_id=meeting_id))
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
            if source == "fireflies":
                meeting = raw.get("meeting", raw)
                transcript = raw.get("transcript", raw)
                if isinstance(transcript, str):
                    try:
                        transcript = json.loads(transcript)
                    except json.JSONDecodeError:
                        transcript = {"sentences": [], "title": meeting.get("title", "")}
                converted = fireflies_to_deepscript(meeting, transcript)
            else:
                if isinstance(raw, str):
                    try:
                        raw = json.loads(raw)
                    except json.JSONDecodeError:
                        failed += 1
                        continue
                converted = circleback_to_deepscript(raw)

            # Check if we got any segments
            if not converted.get("segments"):
                skipped += 1
                continue

            # Save
            filename = converted["metadata"]["file"]["name"]
            if not filename.endswith(".json"):
                filename += ".json"
            filepath = output_path / filename

            # Don't overwrite existing
            if filepath.exists():
                skipped += 1
                continue

            with open(filepath, "w") as f:
                json.dump(converted, f, indent=2, default=str)

            files.append(str(filepath))
            imported += 1
            logger.info("Imported: %s (%d segments)", filename, len(converted["segments"]))

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


def _sanitize_filename(name: str) -> str:
    """Sanitize a string for use as a filename."""
    clean = re.sub(r'[<>:"/\\|?*]', '', name)
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean[:80] if clean else "untitled"


def _parse_mcp_result(result: dict) -> Any:
    """Parse the result from an MCP tool call."""
    data = result.get("result", "")
    if isinstance(data, str):
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return data
    return data
