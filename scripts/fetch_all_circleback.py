"""Fetch ALL Circleback transcripts across both folders.
Handles: CB folder (numeric IDs), FF folder (ULID IDs needing lookup).
Rate-limit aware with auto-stop."""
import asyncio, json, os, sys, time
sys.path.insert(0, '/root/projects/BTask/packages/bflow')
from concierge.circleback_mcp_client import circleback_get_transcript

CB_DIR = '/root/projects/deepscript/transcripts/circleback'
FF_DIR = '/root/projects/deepscript/transcripts/fireflies'
LOG = '/root/projects/deepscript/logs/circleback_fetch.log'

def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line)
    with open(LOG, 'a') as f:
        f.write(line + '\n')

def update_file(fp, entries):
    """Write transcript entries into a DeepScript JSON file."""
    segments, speakers = [], set()
    for j, entry in enumerate(entries):
        speaker = entry.get('speaker', 'Unknown')
        text = entry.get('text', '')
        ts = entry.get('timestamp', 0)
        if not text.strip(): continue
        speakers.add(speaker)
        dur = max(1.0, len(text.split()) / 2.5)
        segments.append({"id": j, "start": round(ts, 1), "end": round(ts + dur, 1),
            "text": text, "speaker": speaker, "speaker_cluster_id": "",
            "confidence": 1.0, "no_speech_prob": 0.0, "words": []})
    if not segments:
        return False
    with open(fp) as fh:
        existing = json.load(fh)
    existing['segments'] = segments
    existing['text'] = ' '.join(s['text'] for s in segments)
    existing['diarization']['speakers_resolved'] = [
        {"local_label": f"SPEAKER_{j:02d}", "display_name": n,
         "speaker_cluster_id": "", "status": "confirmed"}
        for j, n in enumerate(sorted(speakers))]
    existing['diarization']['num_speakers'] = len(speakers)
    with open(fp, 'w') as fh:
        json.dump(existing, fh, indent=2, default=str)
    return True

def is_rate_limited(result):
    """Check if result is an actual rate limit error, not transcript content containing those words."""
    text = result.get('result', '')
    # Actual rate limit: result is a short error string, not a JSON array of transcript data
    if isinstance(text, str) and 'rate limit' in text.lower():
        # If it parses as JSON with transcript data, it's NOT a rate limit
        try:
            data = json.loads(text)
            if isinstance(data, list) and data and 'transcript' in data[0]:
                return False  # It's real transcript data that mentions "rate limit"
        except (json.JSONDecodeError, IndexError, TypeError):
            pass
        return True  # Actual rate limit error
    # Check the rate_limited flag from the client
    if result.get('rate_limited'):
        return True
    return False

async def fetch_by_numeric_id(mid, fp):
    """Fetch transcript using numeric Circleback meeting ID."""
    result = await asyncio.wait_for(circleback_get_transcript(int(mid)), timeout=20)
    if is_rate_limited(result):
        return 'rate_limited'
    text = result.get('result', '')
    data = json.loads(text) if isinstance(text, str) else text
    if not data or not data[0].get('transcript') or len(data[0]['transcript']) <= 1:
        return 'empty'
    return data[0]['transcript']

async def main():
    os.makedirs(os.path.dirname(LOG), exist_ok=True)

    # === Phase 1: CB folder (numeric IDs) ===
    cb_needs = []
    for f in sorted(os.listdir(CB_DIR), reverse=True):
        if not f.endswith('.json'): continue
        fp = os.path.join(CB_DIR, f)
        with open(fp) as fh:
            d = json.load(fh)
        if len(d.get('segments', [])) <= 1:
            mid = d.get('metadata', {}).get('meeting_id', '')
            if mid: cb_needs.append((mid, fp))

    # === Phase 2: FF folder — copy from CB where possible, then fetch ===
    cb_transcripts = {}
    cb_id_lookup = {}
    for f in os.listdir(CB_DIR):
        if not f.endswith('.json'): continue
        fp = os.path.join(CB_DIR, f)
        with open(fp) as fh:
            d = json.load(fh)
        title = d.get('llm_analysis', {}).get('title', '').lower().strip()
        date = f[:10]
        mid = d.get('metadata', {}).get('meeting_id', '')
        if title and mid:
            cb_id_lookup[(title, date)] = mid
        if len(d.get('segments', [])) > 1:
            cb_transcripts[(title, date)] = d.get('segments', []), d.get('diarization', {})

    ff_needs = []
    copied = 0
    for f in sorted(os.listdir(FF_DIR), reverse=True):
        if not f.endswith('.json'): continue
        fp = os.path.join(FF_DIR, f)
        with open(fp) as fh:
            d = json.load(fh)
        mid = d.get('metadata', {}).get('meeting_id', '')
        if not mid.startswith('01'): continue
        if len(d.get('segments', [])) > 1: continue
        title = d.get('llm_analysis', {}).get('title', '').lower().strip()
        date = f[:10]
        key = (title, date)
        if key in cb_transcripts:
            segs, diar = cb_transcripts[key]
            d['segments'] = segs
            d['text'] = ' '.join(s['text'] for s in segs)
            d['diarization'] = diar
            with open(fp, 'w') as fh:
                json.dump(d, fh, indent=2, default=str)
            copied += 1
            continue
        if key in cb_id_lookup:
            ff_needs.append((cb_id_lookup[key], fp))

    if copied:
        log(f"Copied {copied} transcripts from CB folder to FF folder")

    all_needs = cb_needs + ff_needs
    log(f"Fetching: {len(cb_needs)} CB + {len(ff_needs)} FF = {len(all_needs)} total")

    updated = skipped = failed = 0
    for i, (mid, fp) in enumerate(all_needs):
        try:
            result = await fetch_by_numeric_id(mid, fp)
            if result == 'rate_limited':
                log(f"Rate limited after {updated} updates ({i+1} calls). Stopping.")
                break
            elif result == 'empty':
                skipped += 1
            else:
                if update_file(fp, result):
                    updated += 1
                else:
                    skipped += 1
        except Exception as e:
            failed += 1

        if (i + 1) % 50 == 0:
            log(f"Progress: {i+1}/{len(all_needs)} ({updated} ok, {skipped} skip, {failed} fail)")

        await asyncio.sleep(3)

    log(f"DONE: {updated} updated, {skipped} skipped, {failed} failed, {copied} copied")

asyncio.run(main())
