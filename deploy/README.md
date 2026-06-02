# Deploy: cron schedule

`crontab` is a snapshot of the production crontab. Install with:

```bash
crontab deploy/crontab
```

## Schedule

| Time   | Job                                              |
|--------|--------------------------------------------------|
| 06:00  | `fetch_new_meetings.py --source circleback`      |
| 06:05  | `deepscript import circleback --days 1`           |
| 06:10  | `deepscript import fireflies --days 1`            |
| 06:15  | `fetch_fireflies_backlog.py`                      |
| 18:00  | `fetch_new_meetings.py --source circleback fireflies` |

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
