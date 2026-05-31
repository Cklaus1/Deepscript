"""Fetch Circleback transcripts with conservative rate limiting.
Run this periodically until all transcripts are fetched.
Uses ~3s delay between calls to stay under rate limits."""
import asyncio, json, os, sys, time
sys.path.insert(0, '/root/projects/BTask/packages/bflow')
from concierge.circleback_mcp_client import circleback_get_transcript

CB_DIR = '/root/projects/deepscript/transcripts/circleback'

async def main():
    needs = []
    for f in sorted(os.listdir(CB_DIR)):
        if not f.endswith('.json'): continue
        fp = os.path.join(CB_DIR, f)
        with open(fp) as fh:
            d = json.load(fh)
        if len(d.get('segments', [])) <= 1:
            mid = d.get('metadata', {}).get('meeting_id', '')
            if mid: needs.append((mid, fp))
    
    print(f'{len(needs)} meetings need transcripts')
    updated = skipped = failed = rate_limited = 0
    
    for i, (mid, fp) in enumerate(needs):
        try:
            result = await asyncio.wait_for(
                circleback_get_transcript(int(mid)), timeout=20
            )
            result_text = result.get('result', '')
            
            if 'rate limit' in str(result_text).lower():
                rate_limited += 1
                print(f'  Rate limited after {updated} updates. Stopping.')
                print(f'  Wait ~1 hour and re-run this script.')
                break
            
            data = json.loads(result_text) if isinstance(result_text, str) else result_text
            if not data or not data[0].get('transcript') or len(data[0]['transcript']) <= 1:
                skipped += 1; continue
            
            entries = data[0]['transcript']
            segments, speakers = [], set()
            for j, entry in enumerate(entries):
                speaker = entry.get('speaker', 'Unknown')
                text = entry.get('text', ''); ts = entry.get('timestamp', 0)
                if not text.strip(): continue
                speakers.add(speaker)
                dur = max(1.0, len(text.split()) / 2.5)
                segments.append({"id": j, "start": round(ts, 1), "end": round(ts + dur, 1),
                    "text": text, "speaker": speaker, "speaker_cluster_id": "",
                    "confidence": 1.0, "no_speech_prob": 0.0, "words": []})
            
            if segments:
                with open(fp) as fh: existing = json.load(fh)
                existing['segments'] = segments
                existing['text'] = ' '.join(s['text'] for s in segments)
                existing['diarization']['speakers_resolved'] = [
                    {"local_label": f"SPEAKER_{j:02d}", "display_name": n,
                     "speaker_cluster_id": "", "status": "confirmed"}
                    for j, n in enumerate(sorted(speakers))]
                existing['diarization']['num_speakers'] = len(speakers)
                with open(fp, 'w') as fh: json.dump(existing, fh, indent=2, default=str)
                updated += 1
            else:
                skipped += 1
            
        except (asyncio.TimeoutError, Exception) as e:
            failed += 1
        
        if (i + 1) % 25 == 0:
            print(f'  Progress: {i+1}/{len(needs)} ({updated} updated, {skipped} skipped, {failed} failed)')
        
        await asyncio.sleep(3)  # 3 seconds between requests = ~20/min
    
    print(f'DONE: {updated} updated, {skipped} skipped, {failed} failed, {rate_limited} rate limited')
    if rate_limited:
        print(f'Re-run this script in ~1 hour to continue.')

asyncio.run(main())
