"""Fetch Fireflies transcripts directly via GraphQL API and update JSON files.

Usage: python scripts/fetch_fireflies_direct.py [--batch-size 10] [--delay 1]
"""

import json
import os
import re
import sys
import time
import argparse
import urllib.request
import urllib.error
from dotenv import load_dotenv

load_dotenv()

TRANSCRIPTS_DIR = "/root/projects/deepscript/transcripts/fireflies"
API_KEY = os.environ.get("FIREFLIES_API_KEY", "")
API_URL = "https://api.fireflies.ai/graphql"

SKIP_KEYS = {
    'Id', 'Title', 'DateString', 'Privacy', 'Speakers',
    'Host Email', 'Organizer Email', 'Sentences',
    'Fireflies Users', 'Participants', 'Date',
    'Transcript Url', 'Audio Url', 'Video Url',
    'Duration', 'Meeting Attendees', 'Cal Id',
    'Calendar Type', 'Meeting Link', 'Is Live',
    'Calendar Id',
}

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


def fetch_transcript(meeting_id):
    """Fetch transcript from Fireflies GraphQL API."""
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
    """Convert Fireflies sentences to DeepScript segments."""
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


def update_transcript_file(filepath, segments, speakers):
    """Update a transcript JSON file with parsed segments."""
    with open(filepath) as f:
        existing = json.load(f)

    existing['segments'] = segments
    existing['text'] = ' '.join(s['text'] for s in segments)
    existing['diarization']['speakers_resolved'] = [
        {
            "local_label": f"SPEAKER_{j:02d}",
            "display_name": name,
            "speaker_cluster_id": "",
            "status": "confirmed",
        }
        for j, name in enumerate(sorted(speakers))
    ]
    existing['diarization']['num_speakers'] = len(speakers)

    with open(filepath, 'w') as f:
        json.dump(existing, f, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--delay', type=float, default=1.0)
    parser.add_argument('--limit', type=int, default=0, help='Max transcripts to fetch (0=all)')
    args = parser.parse_args()

    # Find files needing transcripts
    needs_transcript = []
    for fname in sorted(os.listdir(TRANSCRIPTS_DIR)):
        if not fname.endswith('.json'):
            continue
        fp = os.path.join(TRANSCRIPTS_DIR, fname)
        with open(fp) as f:
            data = json.load(f)
        segs = len(data.get('segments', []))
        mid = data.get('metadata', {}).get('meeting_id', '')
        if segs <= 1 and mid and not mid.startswith('01'):
            needs_transcript.append((mid, fp, fname))

    total = len(needs_transcript)
    if args.limit > 0:
        needs_transcript = needs_transcript[:args.limit]

    print(f"Total needing transcripts: {total}")
    print(f"Will fetch: {len(needs_transcript)}")

    updated = 0
    failed = 0
    no_sentences = 0

    for i, (mid, fp, fname) in enumerate(needs_transcript):
        sentences, error = fetch_transcript(mid)

        if error:
            if "No sentences" in error:
                no_sentences += 1
            else:
                failed += 1
                if failed <= 10:
                    print(f"  ERROR {mid} ({fname}): {error}")
            continue

        segments, speakers = sentences_to_segments(sentences)
        if not segments or len(segments) <= 1:
            no_sentences += 1
            continue

        update_transcript_file(fp, segments, speakers)
        updated += 1

        if (i + 1) % args.batch_size == 0:
            print(f"  Progress: {i+1}/{len(needs_transcript)} ({updated} updated, {failed} failed, {no_sentences} empty)")
            time.sleep(args.delay)
        elif (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(needs_transcript)} ({updated} updated, {failed} failed, {no_sentences} empty)")

    print(f"\nDone: {updated} updated, {failed} failed, {no_sentences} empty/no-sentences")
    print(f"Remaining: {total - updated}")


if __name__ == '__main__':
    main()
