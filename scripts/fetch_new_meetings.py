"""Incremental fetcher for Circleback and Fireflies transcripts.

Only fetches meetings created since the last successful run.
Tracks state in a JSON manifest so we never re-fetch old meetings.

Usage:
    python scripts/fetch_new_meetings.py [--source circleback] [--source fireflies] [--dry-run]
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# Add BFlow to path
sys.path.insert(0, "/root/projects/BTask/packages/bflow")

TRANSCRIPTS_DIR = Path("/root/projects/deepscript/transcripts")
CB_DIR = TRANSCRIPTS_DIR / "circleback"
FF_DIR = TRANSCRIPTS_DIR / "fireflies"
STATE_FILE = TRANSCRIPTS_DIR / ".import_state.json"
LOG_FILE = TRANSCRIPTS_DIR / "fetch.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# --- State Management ---

def load_state() -> dict:
    """Load import state from disk."""
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            log.warning("Corrupt state file, starting fresh")
    return {"last_run": {}, "fetched_ids": {}}


def save_state(state: dict) -> None:
    """Persist import state to disk."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def get_last_run(source: str) -> datetime | None:
    """Get the timestamp of the last successful run for a source."""
    state = load_state()
    ts = state["last_run"].get(source)
    if ts:
        return datetime.fromisoformat(ts)
    return None


# --- Transcript File Tracking ---

def get_existing_ids(directory: Path) -> dict[str, str]:
    """Map meeting_id -> filepath for all transcripts in a directory."""
    result = {}
    if not directory.exists():
        return result
    for f in directory.glob("*.json"):
        try:
            with open(f) as fh:
                data = json.load(fh)
            mid = data.get("metadata", {}).get("meeting_id", "")
            if mid:
                result[mid] = str(f)
        except (json.JSONDecodeError, OSError):
            continue
    return result


# --- Format Converters ---

def _sanitize_filename(name: str) -> str:
    clean = "".join(c for c in name if c.isalnum() or c in " -_.").strip()
    return clean[:80] if clean else "untitled"


def _parse_timestamp(ts: Any) -> float:
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


def _build_deepscript(meeting: dict, source: str, segments: list[dict], speakers: list[str]) -> dict:
    """Build a DeepScript-format transcript from raw meeting data."""
    title = meeting.get("title") or meeting.get("name", "")
    duration = meeting.get("duration", 0)
    created = meeting.get("createdAt", "") or meeting.get("date", "")

    full_text = " ".join(s["text"] for s in segments)

    speakers_resolved = []
    for i, name in enumerate(speakers):
        speakers_resolved.append({
            "local_label": f"SPEAKER_{i:02d}",
            "display_name": name,
            "speaker_cluster_id": "",
            "status": "confirmed",
        })

    return {
        "text": full_text,
        "language": "en",
        "segments": segments,
        "diarization": {
            "num_speakers": len(speakers),
            "speakers_resolved": speakers_resolved,
        },
        "metadata": {
            "audio": {
                "duration_seconds": duration,
                "format_tags": {"creation_time": created},
            },
            "file": {
                "name": f"{_sanitize_filename(title)}.json",
                "size_human": "",
                "extension": "json",
            },
            "source": source,
            "meeting_id": meeting.get("id", ""),
        },
        "llm_analysis": {
            "title": title,
            "speakers": [
                {"label": n, "likely_name": n, "evidence": f"From {source} speaker ID", "role": ""}
                for n in speakers
            ],
        },
    }


def _save_transcript(data: dict, directory: Path) -> str:
    """Save transcript to disk, return filepath."""
    directory.mkdir(parents=True, exist_ok=True)
    filename = data["metadata"]["file"]["name"]
    if not filename.endswith(".json"):
        filename += ".json"
    filepath = directory / filename
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)
    return str(filepath)


# --- MCP Retry Wrappers ---

async def _cb_call(func, *args, retries=3, base_delay=10, **kwargs):
    """Call a Circleback MCP function with retry and backoff for rate limits."""
    for attempt in range(retries):
        try:
            result = await asyncio.wait_for(func(*args, **kwargs), timeout=120)
            if result.get("success"):
                return result
            error_msg = result.get("error", "unknown")
            # Retry on rate limits and MCP parse failures (both indicate rate limiting)
            if ("Rate limit" in error_msg or "rate limit" in error_msg.lower()
                    or "Failed to parse" in error_msg or "Expecting value" in error_msg):
                delay = base_delay * (2 ** attempt)
                log.warning("Circleback rate limited (attempt %d/%d): %s", attempt + 1, retries, error_msg)
                await asyncio.sleep(delay)
                continue
            return result  # Non-rate-limit error, return as-is
        except asyncio.TimeoutError:
            log.warning("Circleback timeout, retrying in %ds (attempt %d/%d)", base_delay, attempt + 1, retries)
            await asyncio.sleep(base_delay * (2 ** attempt))
    return {"success": False, "error": f"Failed after {retries} retries"}


async def _ff_call(func, *args, retries=3, base_delay=5, **kwargs):
    """Call a Fireflies MCP function with retry and backoff."""
    for attempt in range(retries):
        try:
            result = await asyncio.wait_for(func(*args, **kwargs), timeout=60)
            if result.get("success"):
                return result
            error_msg = result.get("error", "unknown")
            if "Rate limit" in error_msg or "rate limit" in error_msg.lower():
                delay = base_delay * (2 ** attempt)
                log.warning("Fireflies rate limited, retrying in %ds (attempt %d/%d)", delay, attempt + 1, retries)
                await asyncio.sleep(delay)
                continue
            return result
        except asyncio.TimeoutError:
            log.warning("Fireflies timeout, retrying in %ds (attempt %d/%d)", base_delay, attempt + 1, retries)
            await asyncio.sleep(base_delay * (2 ** attempt))
    return {"success": False, "error": f"Failed after {retries} retries"}


# --- Circleback Fetcher ---

async def fetch_circleback_new(state: dict, dry_run: bool = False) -> dict:
    """Fetch only new Circleback meetings since last run."""
    from concierge.circleback_mcp_client import (
        circleback_list_meetings,
        circleback_get_meeting,
        circleback_get_transcript,
    )

    last_run = get_last_run("circleback")
    if last_run is None:
        days = 7
        log.info("First Circleback run — fetching last 7 days")
    else:
        days = 1
        log.info(f"Circleback incremental — last run: {last_run.isoformat()}")

    # Get existing IDs to skip
    existing = get_existing_ids(CB_DIR) | get_existing_ids(FF_DIR)
    fetched_ids = set(state.get("fetched_ids", {}).get("circleback", []))

    list_result = await _cb_call(circleback_list_meetings, days=days, max_pages=5)

    if not list_result.get("success"):
        log.error("Circleback list_meetings error: %s", list_result.get("error"))
        return {"updated": 0, "skipped": 0, "failed": 0, "error": list_result.get("error", "unknown")}

    meetings = list_result.get("result", [])
    log.info(f"Circleback: {len(meetings)} meetings in last {days} day(s)")

    updated = skipped = failed = 0
    new_ids = set()

    for meeting in meetings:
        mid = str(meeting.get("id", ""))
        if not mid:
            continue

        # Skip if already fetched
        if mid in existing or mid in fetched_ids:
            continue

        try:
            # Get full meeting details
            meeting_result = await _cb_call(circleback_get_meeting, mid, retries=2, base_delay=5)
            if not meeting_result.get("success"):
                log.warning("Circleback get_meeting(%s) failed: %s", mid, meeting_result.get("error"))
                failed += 1
                continue

            full_meeting = meeting_result.get("result", {})
            if isinstance(full_meeting, str):
                try:
                    full_meeting = json.loads(full_meeting)
                except json.JSONDecodeError:
                    failed += 1
                    continue
            # Circleback returns a list: [{meeting data}]
            if isinstance(full_meeting, list) and len(full_meeting) > 0:
                full_meeting = full_meeting[0]

            # Get transcript
            tx_result = await _cb_call(circleback_get_transcript, mid, retries=2, base_delay=5)
            if not tx_result.get("success"):
                log.warning("Circleback get_transcript(%s) failed: %s", mid, tx_result.get("error"))
                failed += 1
                continue

            tx_data = tx_result.get("result", "")
            if isinstance(tx_data, str):
                try:
                    tx_data = json.loads(tx_data)
                except json.JSONDecodeError:
                    tx_data = {}

            # Circleback returns a list: [{meetingId, transcript: [...]}]
            if isinstance(tx_data, list) and len(tx_data) > 0:
                tx_data = tx_data[0]

            # Build segments from transcript entries
            transcript_entries = tx_data.get("transcript", [])
            if not transcript_entries or len(transcript_entries) <= 1:
                skipped += 1
                new_ids.add(mid)
                continue

            segments = []
            speakers = set()
            for i, entry in enumerate(transcript_entries):
                speaker = entry.get("speaker", "Unknown")
                text = entry.get("text", "")
                if not text.strip():
                    continue
                speakers.add(speaker)
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
                    "text": text,
                    "speaker": speaker,
                    "speaker_cluster_id": "",
                    "confidence": 1.0,
                    "no_speech_prob": 0.0,
                    "words": [],
                })

            if not segments:
                skipped += 1
                new_ids.add(mid)
                continue

            deepscript = _build_deepscript(full_meeting, "circleback", segments, sorted(speakers))
            filename = f"{_sanitize_filename(deepscript['metadata']['file']['name'])}"
            if dry_run:
                log.info("DRY RUN: would save %s (%d segments)", filename, len(segments))
                updated += 1
                new_ids.add(mid)
                continue
            filepath = _save_transcript(deepscript, CB_DIR)
            log.info("Circleback: saved %s (%d segments)", filepath, len(segments))
            updated += 1
            new_ids.add(mid)

        except asyncio.TimeoutError:
            log.error("Circleback fetch(%s) timed out", mid)
            failed += 1
        except Exception as e:
            log.error("Circleback fetch(%s) failed: %s", mid, e, exc_info=True)
            failed += 1

        # Rate limit: 3s between calls
        await asyncio.sleep(3)

    # Update state
    if not dry_run:
        if "fetched_ids" not in state:
            state["fetched_ids"] = {}
        state["fetched_ids"]["circleback"] = list(fetched_ids | new_ids)
        state["last_run"]["circleback"] = datetime.now(timezone.utc).isoformat()
        save_state(state)

    return {"updated": updated, "skipped": skipped, "failed": failed}


# --- Fireflies Fetcher ---

async def fetch_fireflies_new(state: dict, dry_run: bool = False) -> dict:
    """Fetch only new Fireflies meetings since last run."""
    from concierge.fireflies_mcp_client import (
        fireflies_list_meetings,
        fireflies_get_transcript,
    )

    last_run = get_last_run("fireflies")
    if last_run is None:
        days = 7
        log.info("First Fireflies run — fetching last 7 days")
    else:
        days = 1
        log.info(f"Fireflies incremental — last run: {last_run.isoformat()}")

    # Get existing IDs to skip
    existing = get_existing_ids(FF_DIR) | get_existing_ids(CB_DIR)
    fetched_ids = set(state.get("fetched_ids", {}).get("fireflies", []))

    list_result = await _ff_call(fireflies_list_meetings, limit=50, days=days)

    if not list_result.get("success"):
        log.error("Fireflies list_meetings error: %s", list_result.get("error"))
        return {"updated": 0, "skipped": 0, "failed": 0, "error": list_result.get("error", "unknown")}

    meeting_list = list_result.get("result", [])
    if isinstance(meeting_list, str):
        try:
            meeting_list = json.loads(meeting_list)
        except json.JSONDecodeError:
            meeting_list = []

    log.info(f"Fireflies: {len(meeting_list)} meetings in last {days} day(s)")

    updated = skipped = failed = 0
    new_ids = set()

    for meeting in meeting_list:
        mid = meeting.get("id", "")
        if not mid:
            continue

        # Skip if already fetched
        if mid in existing or mid in fetched_ids:
            continue

        tx_result = await _ff_call(fireflies_get_transcript, mid, retries=2, base_delay=3)
        if not tx_result.get("success"):
            log.warning("Fireflies get_transcript(%s) failed: %s", mid, tx_result.get("error"))
            failed += 1
            continue

        tx_data = tx_result.get("result", "")
        if isinstance(tx_data, str):
            try:
                tx_data = json.loads(tx_data)
            except json.JSONDecodeError:
                tx_data = {}

        sentences = tx_data.get("sentences", [])
        if not sentences:
            skipped += 1
            new_ids.add(mid)
            continue

        # Build segments from sentences
        segments = []
        speakers = set()
        time_est = 0.0
        for i, sent in enumerate(sentences):
            speaker = sent.get("speaker_name") or "Unknown"
            text = (sent.get("text") or "").strip()
            if not text:
                continue
            speakers.add(speaker)
            words = len(text.split())
            dur = max(1.0, words / 2.5)
            segments.append({
                "id": i,
                "start": round(time_est, 1),
                "end": round(time_est + dur, 1),
                "text": text,
                "speaker": speaker,
                "speaker_cluster_id": "",
                "confidence": 1.0,
                "no_speech_prob": 0.0,
                "words": [],
            })
            time_est += dur

        if not segments:
            skipped += 1
            new_ids.add(mid)
            continue

        title = tx_data.get("title") or meeting.get("title", "")
        duration = meeting.get("duration", 0) * 60  # Fireflies returns minutes
        created = meeting.get("date", "")

        deepscript = {
            "text": " ".join(s["text"] for s in segments),
            "language": "en",
            "segments": segments,
            "diarization": {
                "num_speakers": len(speakers),
                "speakers_resolved": [
                    {"local_label": f"SPEAKER_{j:02d}", "display_name": n,
                     "speaker_cluster_id": "", "status": "confirmed"}
                    for j, n in enumerate(sorted(speakers))
                ],
            },
            "metadata": {
                "audio": {"duration_seconds": duration, "format_tags": {"creation_time": created}},
                "file": {"name": f"{_sanitize_filename(title)}.json", "size_human": "", "extension": "json"},
                "source": "fireflies",
                "meeting_id": mid,
            },
            "llm_analysis": {
                "title": title,
                "speakers": [
                    {"label": n, "likely_name": n, "evidence": "From Fireflies speaker ID", "role": ""}
                    for n in speakers
                ],
            },
        }

        filename = f"{_sanitize_filename(deepscript['metadata']['file']['name'])}"
        if dry_run:
            log.info("DRY RUN: would save %s (%d segments)", filename, len(segments))
            updated += 1
            new_ids.add(mid)
            continue
        filepath = _save_transcript(deepscript, FF_DIR)
        log.info("Fireflies: saved %s (%d segments)", filepath, len(segments))
        updated += 1
        new_ids.add(mid)

        # Rate limit: 1s between calls
        await asyncio.sleep(1)

    # Update state
    if not dry_run:
        if "fetched_ids" not in state:
            state["fetched_ids"] = {}
        state["fetched_ids"]["fireflies"] = list(fetched_ids | new_ids)
        state["last_run"]["fireflies"] = datetime.now(timezone.utc).isoformat()
        save_state(state)

    return {"updated": updated, "skipped": skipped, "failed": failed}


# --- Main ---

async def main(sources: list[str] | None = None, dry_run: bool = False) -> dict:
    """Run incremental fetch for specified sources."""
    if sources is None:
        sources = ["circleback", "fireflies"]

    if dry_run:
        log.info("DRY RUN mode — will not write files or update state")

    state = load_state()
    results = {}

    for source in sources:
        log.info("=" * 60)
        log.info("Fetching %s...", source)
        log.info("=" * 60)

        if source == "circleback":
            results["circleback"] = await fetch_circleback_new(state, dry_run=dry_run)
        elif source == "fireflies":
            results["fireflies"] = await fetch_fireflies_new(state, dry_run=dry_run)
        else:
            log.error("Unknown source: %s", source)

    # Summary
    log.info("=" * 60)
    log.info("FETCH SUMMARY")
    log.info("=" * 60)
    total_updated = 0
    total_skipped = 0
    total_failed = 0
    for source, r in results.items():
        u = r.get("updated", 0)
        s = r.get("skipped", 0)
        f = r.get("failed", 0)
        total_updated += u
        total_skipped += s
        total_failed += f
        error = r.get("error")
        if error:
            log.info("  %s: %d updated, %d skipped, %d failed — ERROR: %s", source, u, s, f, error)
        else:
            log.info("  %s: %d updated, %d skipped, %d failed", source, u, s, f)

    log.info("  TOTAL: %d updated, %d skipped, %d failed", total_updated, total_skipped, total_failed)

    if dry_run:
        log.info("DRY RUN — no files written, state not saved")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Incremental transcript fetcher")
    parser.add_argument("--source", nargs="+", choices=["circleback", "fireflies"],
                        help="Sources to fetch (default: both)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be fetched without writing")
    args = parser.parse_args()

    asyncio.run(main(sources=args.source, dry_run=args.dry_run))