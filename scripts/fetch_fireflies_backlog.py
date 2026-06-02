"""Fetch all missing Fireflies transcripts (CB + FF folders) via direct GraphQL API.

Targets:
  - transcripts/fireflies/ files with no segments (any ID format)
  - transcripts/circleback/ files with ULID-style IDs and no segments
    (these are actually Fireflies meetings misplaced in the CB folder)

Idempotent — re-run anytime, skips files that already have transcripts.

Usage: python scripts/fetch_fireflies_backlog.py [--limit N] [--delay 1]
"""

import json
import os
import sys
import time
import argparse
import urllib.request
import urllib.error
from dotenv import load_dotenv

load_dotenv("/root/projects/deepscript/.env")
load_dotenv("/root/projects/BTask/packages/bflow/.env")

FF_DIR = "/root/projects/deepscript/transcripts/fireflies"
CB_DIR = "/root/projects/deepscript/transcripts/circleback"
# Meeting IDs the API has confirmed have no transcript — skip on future runs
# so the backlog actually drains instead of re-fetching dead items daily.
EMPTY_CACHE = "/root/projects/deepscript/transcripts/.fireflies_empty.json"
API_KEY = os.environ.get("FIREFLIES_API_KEY", "")
API_URL = "https://api.fireflies.ai/graphql"

QUERY = """
query Transcript($transcriptId: String!) {
  transcript(id: $transcriptId) {
    id
    title
    sentences {
      speaker_name
      text
    }
  }
}
"""


def log(msg):
    # Print only — the caller (cron `>> logfile`, or terminal) owns the sink.
    # Writing to a file here too would double every line when cron also redirects.
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def load_empty_cache():
    """Return set of meeting IDs previously confirmed empty by the API."""
    try:
        with open(EMPTY_CACHE) as f:
            return set(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return set()


def save_empty_cache(ids):
    os.makedirs(os.path.dirname(EMPTY_CACHE), exist_ok=True)
    with open(EMPTY_CACHE, 'w') as f:
        json.dump(sorted(ids), f, indent=2)


def fetch_transcript(meeting_id):
    """Fetch transcript from Fireflies GraphQL API. Returns (sentences, error)."""
    payload = json.dumps({
        "query": QUERY,
        "variables": {"transcriptId": meeting_id}
    }).encode('utf-8')

    req = urllib.request.Request(
        API_URL,
        data=payload,
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {API_KEY}',
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        body = e.read().decode('utf-8', errors='replace')
        return None, f"HTTP {e.code}: {body[:200]}"
    except Exception as e:
        return None, str(e)

    transcript = data.get('data', {}).get('transcript')
    if not transcript:
        errors = data.get('errors', [])
        if errors:
            return None, errors[0].get('message', 'Unknown error')
        return None, "No transcript data"

    sentences = transcript.get('sentences') or []
    if not sentences:
        return None, "No sentences"

    return sentences, None


def sentences_to_segments(sentences):
    """Convert Fireflies sentences to DeepScript segments + speaker set."""
    segments = []
    speakers = set()
    time_est = 0.0
    seg_id = 0
    for sent in sentences:
        speaker = sent.get('speaker_name') or 'Unknown'
        text = (sent.get('text') or '').strip()
        if not text:
            continue
        speakers.add(speaker)
        words = len(text.split())
        dur = max(1.0, words / 2.5)
        segments.append({
            "id": seg_id,
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
        seg_id += 1
    return segments, speakers


def update_file(fp, segments, speakers):
    """Write transcript into a DeepScript JSON file. Preserves other metadata."""
    with open(fp) as f:
        existing = json.load(f)
    existing['segments'] = segments
    existing['text'] = ' '.join(s['text'] for s in segments)
    diar = existing.get('diarization') or {}
    diar['speakers_resolved'] = [
        {
            "local_label": f"SPEAKER_{j:02d}",
            "display_name": name,
            "speaker_cluster_id": "",
            "status": "confirmed",
        }
        for j, name in enumerate(sorted(speakers))
    ]
    diar['num_speakers'] = len(speakers)
    existing['diarization'] = diar
    with open(fp, 'w') as f:
        json.dump(existing, f, indent=2, default=str)


def find_needs(directory, require_ulid=False):
    """Return [(mid, filepath)] for files missing transcripts.

    If require_ulid: only files whose meeting_id is a 26-char ULID starting with '01'.
    Otherwise: any non-empty ID.
    """
    needs = []
    if not os.path.isdir(directory):
        return needs
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith('.json'):
            continue
        fp = os.path.join(directory, fname)
        try:
            with open(fp) as f:
                d = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if len(d.get('segments') or []) > 1:
            continue
        mid = d.get('metadata', {}).get('meeting_id', '')
        if not mid or not isinstance(mid, str):
            continue
        if require_ulid:
            if not (mid.startswith('01') and len(mid) >= 20):
                continue
        needs.append((mid, fp))
    return needs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=0, help='Max to fetch (0=all)')
    parser.add_argument('--delay', type=float, default=0.5, help='Seconds between calls')
    args = parser.parse_args()

    if not API_KEY:
        log("ERROR: FIREFLIES_API_KEY not set")
        sys.exit(1)

    ff_needs = find_needs(FF_DIR, require_ulid=False)
    cb_needs = find_needs(CB_DIR, require_ulid=True)
    all_needs = ff_needs + cb_needs

    log(f"FF folder needs: {len(ff_needs)}")
    log(f"CB folder (ULID) needs: {len(cb_needs)}")
    log(f"Total: {len(all_needs)}")

    # Skip meetings already confirmed empty upstream — these will never have a
    # transcript, so re-fetching them every run just wastes API calls.
    empty_cache = load_empty_cache()
    if empty_cache:
        before = len(all_needs)
        all_needs = [(mid, fp) for mid, fp in all_needs if mid not in empty_cache]
        skipped = before - len(all_needs)
        if skipped:
            log(f"Skipping {skipped} known-empty (cached); {len(all_needs)} to fetch")

    if not all_needs:
        log("DONE: nothing to fetch")
        return

    if args.limit > 0:
        all_needs = all_needs[:args.limit]
        log(f"Limited to first {args.limit}")

    updated = empty = failed = 0
    new_empty = set()
    for i, (mid, fp) in enumerate(all_needs):
        sentences, error = fetch_transcript(mid)
        if error:
            if "No sentences" in error or "not found" in error.lower():
                empty += 1
                new_empty.add(mid)
            else:
                failed += 1
                if failed <= 5:
                    log(f"  ERROR {mid}: {error[:120]}")
        else:
            segments, speakers = sentences_to_segments(sentences)
            if not segments or len(segments) <= 1:
                empty += 1
                new_empty.add(mid)
            else:
                update_file(fp, segments, speakers)
                updated += 1

        if (i + 1) % 50 == 0:
            log(f"Progress: {i+1}/{len(all_needs)} (ok={updated} empty={empty} fail={failed})")

        time.sleep(args.delay)

    if new_empty:
        save_empty_cache(empty_cache | new_empty)
        log(f"Cached {len(new_empty)} new empty IDs ({len(empty_cache | new_empty)} total)")

    log(f"DONE: ok={updated} empty={empty} fail={failed}")


if __name__ == '__main__':
    main()
