#!/usr/bin/env python3
"""Import 2026 transcripts that are already on disk in DeepScript format.

These files were fetched by fetch_fireflies_new.py and fetch_circleback_slow.py
but never registered in .import_state.json. This script scans both directories,
extracts meeting_id from metadata, and updates the import state so the pipeline
knows they're already processed.
"""

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


def import_source(source_type, pattern, label):
    state = load_state()
    if source_type not in state:
        state[source_type] = {}

    files = sorted(glob.glob(pattern))
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


def main():
    ff_count, ff_skip, ff_err = import_source(
        "fireflies",
        os.path.join(os.path.dirname(__file__), "..", "transcripts", "fireflies", "2026-*.json"),
        "Fireflies 2026",
    )
    cb_count, cb_skip, cb_err = import_source(
        "circleback",
        os.path.join(os.path.dirname(__file__), "..", "transcripts", "circleback", "2026-*.json"),
        "Circleback 2026",
    )

    total_new = ff_count + cb_count
    print(f"\n{'='*60}")
    print(f"TOTAL: {total_new} new transcripts registered")
    print(f"{'='*60}")

    if total_new > 0:
        print(f"\nRun 'make analyze' to process the {total_new} new transcripts.")
    else:
        print("\nAll 2026 transcripts were already registered. Nothing to do.")


if __name__ == "__main__":
    main()