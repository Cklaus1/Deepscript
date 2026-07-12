"""Fetch new Fireflies meetings and save transcripts.

Uses the BFlow MCP client (OAuth) which correctly handles authentication
and the Fireflies MCP endpoint Accept headers.

Usage:
    python scripts/fetch_fireflies_new.py [--days 7] [--limit 50] [--dry-run]
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, "/root/projects/BTask/packages/bflow")

from concierge.fireflies_mcp_client import (
    fireflies_get_transcript,
    fireflies_list_meetings,
)

FF_DIR = Path("/root/projects/deepscript/transcripts/fireflies")
STATE_FILE = Path("/root/projects/deepscript/transcripts/.import_state.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


# --- State ---

def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {"last_run": {}, "fetched_ids": {}}


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def get_existing_ids(directory: Path) -> set[str]:
    result = set()
    if not directory.exists():
        return result
    for f in directory.glob("*.json"):
        try:
            with open(f) as fh:
                data = json.load(fh)
            mid = data.get("metadata", {}).get("meeting_id", "")
            if mid:
                result.add(mid)
        except (json.JSONDecodeError, OSError):
            continue
    return result


def _sanitize_filename(name: str) -> str:
    clean = "".join(c for c in name if c.isalnum() or c in " -_.").strip()
    return clean[:80] if clean else "untitled"


# --- Fetch ---

async def fetch_fireflies(days: int = 7, limit: int = 50, dry_run: bool = False) -> dict:
    state = load_state()

    # First run: use --days, incremental: use 1 day
    if not state.get("last_run", {}).get("fireflies"):
        log.info("First Fireflies run — fetching last %d days", days)
    else:
        days = 1
        log.info("Fireflies incremental — last run: %s", state["last_run"]["fireflies"])

    existing = get_existing_ids(FF_DIR)
    fetched_ids = set(state.get("fetched_ids", {}).get("fireflies", []))

    list_result = await fireflies_list_meetings(limit=limit, days=days)
    if not list_result.get("success"):
        log.error("list_meetings error: %s", list_result.get("error"))
        return {"updated": 0, "skipped": 0, "failed": 0, "error": list_result.get("error")}

    meeting_list = list_result.get("result", [])
    if isinstance(meeting_list, str):
        try:
            meeting_list = json.loads(meeting_list)
        except json.JSONDecodeError:
            meeting_list = []

    log.info("Found %d Fireflies meetings in last %d day(s)", len(meeting_list), days)

    updated = skipped = failed = 0
    new_ids = set()

    for meeting in meeting_list:
        mid = meeting.get("id", "")
        if not mid:
            continue
        if mid in existing or mid in fetched_ids:
            continue

        tx_result = await fireflies_get_transcript(mid)
        if not tx_result.get("success"):
            log.warning("get_transcript(%s) failed: %s", mid, tx_result.get("error"))
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

        # Build segments
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
        duration = meeting.get("duration", 0) * 60
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

        FF_DIR.mkdir(parents=True, exist_ok=True)
        filepath = FF_DIR / f"{filename}.json"
        with open(filepath, "w") as f:
            json.dump(deepscript, f, indent=2, default=str)
        log.info("Saved %s (%d segments)", filepath, len(segments))
        updated += 1
        new_ids.add(mid)

        await asyncio.sleep(1)

    # Update state
    if not dry_run:
        if "fetched_ids" not in state:
            state["fetched_ids"] = {}
        state["fetched_ids"]["fireflies"] = list(fetched_ids | new_ids)
        state["last_run"]["fireflies"] = datetime.now(timezone.utc).isoformat()
        save_state(state)

    return {"updated": updated, "skipped": skipped, "failed": failed}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch new Fireflies transcripts")
    parser.add_argument("--days", type=int, default=7, help="Days to look back (first run only)")
    parser.add_argument("--limit", type=int, default=50, help="Max meetings to fetch")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be fetched")
    args = parser.parse_args()

    result = asyncio.run(fetch_fireflies(days=args.days, limit=args.limit, dry_run=args.dry_run))
    log.info("Result: %s", result)