# Deploy: cron schedule

`crontab` is a snapshot of the production crontab. Install with:

```bash
crontab deploy/crontab
```

## Schedule

| Time          | Job                                                         |
|---------------|-------------------------------------------------------------|
| every 3h, :07 | fetch → import → analyze new calls (chained, one line)      |
| 06:15         | `fetch_fireflies_backlog.py` (historical backfill)          |
| 06:25         | `speakers pages` — regenerate CRM (people/companies/etc.)   |

The 3-hour job is the main pipeline: it fetches new meetings, imports them,
and analyzes anything new. It's cheap when idle — `analyze --new-only` dedups
all transcripts in ~0.01s, so a no-op run finishes in well under 20s; only
genuinely new calls incur an LLM analysis. A measured end-to-end run with one
new call took ~19s.

CRM regeneration is split out to once daily because it scans every speaker
profile (~7s) and only needs to be current daily, not every 3 hours. The 3h
job uses `--no-crm` to stay fast.

## Gotchas (learned the hard way)

1. **Use `.` not `source`.** cron runs jobs under `/bin/sh`, which is `dash` on
   this box. `dash` has no `source` builtin — every line using
   `source /root/projects/deepscript/.env` died at line 1 with
   `source: not found`, silently importing nothing for days. POSIX `.` works in
   both dash and bash.

2. **`cd` before relative-path commands.** The `import` lines write to
   `-o transcripts/imported` (relative). cron's default CWD is `/root`, so
   without `cd /root/projects/deepscript` the import scanned the wrong directory
   and found nothing.

When in doubt, verify a cron line actually fired by checking the log file's
mtime — a frozen mtime means the job is dying before it writes.
