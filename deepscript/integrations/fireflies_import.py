"""Import transcripts from Fireflies via MCP tools.

Fetches both full transcripts (speaker-attributed sentences) and
summaries (action items, notes, keywords) for each meeting.

Requires Fireflies MCP tools to be available (claude mcp add fireflies).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def import_fireflies(
    days: int = 30,
    output_dir: str | Path = "transcripts/fireflies",
    email_to_name: dict[str, str] | None = None,
    limit: int = 50,
) -> dict:
    """Import Fireflies meetings with full transcripts + summaries.

    Uses BFlow's Fireflies MCP client to fetch data.

    Args:
        days: Fetch meetings from last N days.
        output_dir: Where to save converted transcripts.
        email_to_name: Email → display name mapping for attendees.
        limit: Max meetings to fetch.

    Returns: {"imported": int, "skipped": int, "failed": int, "files": []}
    """
    import asyncio
    import sys

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if email_to_name is None:
        email_to_name = _load_email_map()

    # Add BFlow to path
    bflow_path = Path("/root/projects/BTask/packages/bflow")
    if str(bflow_path) not in sys.path:
        sys.path.insert(0, str(bflow_path))

    try:
        from concierge.fireflies_mcp_client import (
            fireflies_list_meetings,
            fireflies_get_transcript,
        )
    except ImportError:
        logger.error("BFlow not available. Install or add to path.")
        return {"imported": 0, "skipped": 0, "failed": 0, "files": []}

    async def _fetch_all():
        # List meetings
        list_result = await fireflies_list_meetings(limit=limit, days=days)
        if not list_result.get("success"):
            logger.error("Failed to list meetings: %s", list_result.get("error"))
            return []

        meetings_raw = list_result.get("result", [])
        if isinstance(meetings_raw, str):
            try:
                meetings_raw = json.loads(meetings_raw)
            except json.JSONDecodeError:
                return []

        results = []
        for m in meetings_raw:
            mid = m.get("id", "")
            title = m.get("title", "untitled")
            if not mid:
                continue

            # Fetch full transcript
            tx_result = await fireflies_get_transcript(mid)
            tx_text = ""
            if tx_result.get("success"):
                tx_text = tx_result.get("result", "")
                if isinstance(tx_text, dict):
                    tx_text = json.dumps(tx_text)

            results.append({"meeting": m, "transcript_text": tx_text})
            logger.info("Fetched: %s (%s)", title[:40], mid[:12])

        return results

    fetched = asyncio.run(_fetch_all())

    imported = 0
    skipped = 0
    failed = 0
    files = []

    for item in fetched:
        try:
            m = item["meeting"]
            title = m.get("title", "untitled")
            date = m.get("dateString", "")[:10]
            summary = m.get("summary") or {}

            converted = convert_fireflies_meeting(
                meeting_list_entry=m,
                transcript_text=item["transcript_text"],
                summary_data=summary,
                email_to_name=email_to_name,
            )

            safe_name = re.sub(r'[<>:"/\\|?*]', '', title).strip()[:60]
            filename = f"{date}_{safe_name}.json"
            filepath = output_path / filename

            if filepath.exists():
                skipped += 1
                continue

            with open(filepath, "w") as f:
                json.dump(converted, f, indent=2, default=str)

            files.append(str(filepath))
            seg_count = len(converted.get("segments", []))
            logger.info("Imported: %s (%d segments)", filename, seg_count)
            imported += 1

        except Exception as e:
            logger.warning("Failed to import: %s", e)
            failed += 1

    return {"imported": imported, "skipped": skipped, "failed": failed, "files": files}


def convert_fireflies_meeting(
    meeting_list_entry: dict,
    transcript_text: str,
    summary_data: dict | str,
    email_to_name: dict[str, str] | None = None,
) -> dict:
    """Convert a Fireflies meeting (transcript + summary) to DeepScript format.

    Args:
        meeting_list_entry: Entry from fireflies_get_transcripts list.
        transcript_text: Raw text from fireflies_get_transcript (Speaker: text format).
        summary_data: Parsed summary from fireflies_get_summary.
        email_to_name: Email → name mapping.

    Returns: DeepScript-format transcript dict.
    """
    if email_to_name is None:
        email_to_name = {}

    title = meeting_list_entry.get("title", "untitled")
    date_str = meeting_list_entry.get("dateString", "")
    duration_min = meeting_list_entry.get("duration", 0) or 0
    meeting_id = meeting_list_entry.get("id", "")

    # Parse transcript text into segments
    segments, speaker_names = _parse_transcript_text(transcript_text)

    # Resolve attendee emails to names
    attendees = meeting_list_entry.get("meetingAttendees") or []
    resolved_attendees = []
    for att in attendees:
        email = (att.get("email") or "").lower()
        name = att.get("displayName") or email_to_name.get(email, "")
        if not name and email:
            name = email.split("@")[0].replace(".", " ").title()
        resolved_attendees.append({"name": name, "email": email})

    # Merge speaker names from transcript with attendee names
    all_speakers = list(set(speaker_names + [a["name"] for a in resolved_attendees if a["name"]]))

    # Parse summary
    if isinstance(summary_data, str):
        try:
            summary_data = json.loads(summary_data)
        except json.JSONDecodeError:
            summary_data = {"raw": summary_data}

    summary_fields = _extract_summary_fields(summary_data)

    # Build DeepScript format
    full_text = " ".join(s["text"] for s in segments) if segments else summary_fields.get("short_summary", "")

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


def _parse_transcript_text(text: str) -> tuple[list[dict], list[str]]:
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
        if speaker in ("Id", "Title", "DateString", "Privacy", "Speakers",
                        "Host Email", "Organizer Email", "Calendar Id",
                        "Fireflies Users", "Participants", "Date",
                        "Transcript Url", "Audio Url", "Video Url",
                        "Duration", "Meeting Attendees", "Cal Id",
                        "Calendar Type", "Meeting Link", "Is Live",
                        "Sentences"):
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
            # Extract timestamp if present
            ts_match = re.search(r'\((\d{1,2}:\d{2})\)\s*$', line)
            timestamp = ts_match.group(1) if ts_match else ""
            text = line[:ts_match.start()].strip() if ts_match else line

            items.append({
                "text": text,
                "assignee": current_assignee,
                "timestamp": timestamp,
            })

    return items


def _sanitize(name: str) -> str:
    clean = re.sub(r'[<>:"/\\|?*]', '', name)
    return re.sub(r'\s+', ' ', clean).strip()[:80]


def _load_email_map() -> dict[str, str]:
    """Load email → name mapping from contacts cache."""
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
