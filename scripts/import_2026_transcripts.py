#!/usr/bin/env python3
"""Register transcripts already on disk in DeepScript format.

Scans fireflies/ and circleback/ transcript directories, extracts meeting_id
from metadata, and updates .import_state.json so the pipeline knows they're
already processed. Supports --start and --end date filters (YYYY-MM-DD) to
scope which files to register.

These files were fetched by fetch_fireflies_new.py and fetch_circleback_slow.py
but never registered in .import_state.json.
"""

import argparse
import json
import glob
import os
import sys
from datetime import datetime, timezone

STATE_FILE = os.path.join(os.path.dirname(__file__), "..", "transcripts", ".import_state.json")


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"fireflies": {}, "circleback": {}}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def import_source(source_type, patterns, label):
    state = load_state()
    if source_type not in state:
        state[source_type] = {}

    files = []
    for p in patterns:
        if isinstance(p, str):
            files.extend(glob.glob(p))
        else:
            files.extend(p)
    files = sorted(set(files))
    print(f"\n{'='*60}")
    print(f"{label}: {len(files)} files found on disk")
    print(f"{'='*60}")

    new_count = 0
    skip_count = 0
    error_count = 0

    for filepath in files:
        filename = os.path.basename(filepath)
        try:
            with open(filepath) as f:
                data = json.load(f)

            meeting_id = data.get("metadata", {}).get("meeting_id")
            if not meeting_id:
                print(f"  SKIP {filename}: no meeting_id")
                error_count += 1
                continue

            if meeting_id in state[source_type]:
                skip_count += 1
                continue

            # Register the transcript
            state[source_type][meeting_id] = {
                "source": source_type,
                "filename": filename,
                "imported_at": datetime.now(timezone.utc).isoformat(),
                "status": "imported",
            }
            new_count += 1

        except Exception as e:
            print(f"  ERROR {filename}: {e}")
            error_count += 1
            continue

    save_state(state)
    print(f"\n{label} summary:")
    print(f"  New imports: {new_count}")
    print(f"  Already imported: {skip_count}")
    print(f"  Errors: {error_count}")
    return new_count, skip_count, error_count


def filter_by_date(files, start, end):
    """Filter files whose YYYY-MM-*.json date falls within [start, end]."""
    if not start and not end:
        return files
    result = []
    for f in files:
        filename = os.path.basename(f)
        # filename format: YYYY-MM-DD-*.json
        parts = filename.split("-")
        if len(parts) < 3:
            continue
        try:
            file_date = datetime.strptime(f"{parts[0]}-{parts[1]}-{parts[2]}", "%Y-%m-%d").date()
        except ValueError:
            continue
        if start and file_date < start:
            continue
        if end and file_date > end:
            continue
        result.append(f)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Register transcripts already on disk so the pipeline knows they're processed."
    )
    parser.add_argument("--start", help="Start date filter (YYYY-MM-DD), inclusive")
    parser.add_argument("--end", help="End date filter (YYYY-MM-DD), inclusive")
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").date() if args.start else None
    end = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end else None

    base = os.path.join(os.path.dirname(__file__), "..", "transcripts")

    ff_files = filter_by_date(glob.glob(os.path.join(base, "fireflies", "*.json")), start, end)
    cb_files = filter_by_date(glob.glob(os.path.join(base, "circleback", "*.json")), start, end)

    ff_count, ff_skip, ff_err = import_source(
        "fireflies",
        ff_files,
        f"Fireflies ({args.start or 'all'} to {args.end or 'all'})",
    )
    cb_count, cb_skip, cb_err = import_source(
        "circleback",
        cb_files,
        f"Circleback ({args.start or 'all'} to {args.end or 'all'})",
    )

    total_new = ff_count + cb_count
    print(f"\n{'='*60}")
    print(f"TOTAL: {total_new} new transcripts registered")
    print(f"{'='*60}")

    if total_new > 0:
        print(f"\nRun 'make analyze' to process the {total_new} new transcripts.")
    else:
        print("\nAll matching transcripts were already registered. Nothing to do.")


if __name__ == "__main__":
    main()