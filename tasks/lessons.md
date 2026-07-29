# Lessons Learned

## CostTracker re-persist bug — quadratic usage-log growth (2026-07-16)

`CostTracker.__post_init__` seeded the budget total by loading the **entire**
`~/.deepscript/usage.jsonl` into `self.entries`, and `persist()` appended
**all** of `self.entries` back. So every tracker (one per `LLMProvider`, one per
analyze run under the 3h cron) re-copied the whole history → the log grew
quadratically to **19 GB / 81.6M lines** in 8 days (~98% duplicated). Side
effect: `pytest` hung >8 min because any test constructing a `CostTracker`
blocked on `read_text()` of the 19 GB file.

**Rules:**
- A seed/accumulate step must **stream** and keep only aggregate totals — never
  retain every historical record in memory just to sum it.
- Keep the invariant that a "session" collection (`self.entries`) holds only
  what *this* run produced, so an append-persist can't rewrite history. If you
  need cumulative state (budget guard), keep it in separate scalar fields.
- A monotonically-growing append-only log needs a size guard / rotation, and
  reads over it must be O(1) memory (line-by-line), not `read_text()`.

**Rotation added:** `rotate_usage_log()` prunes `usage.jsonl` to the most
recent `ROTATE_KEEP_LINES` entries (default 200k) once it passes
`ROTATE_MAX_BYTES` (default 100 MB), via write-temp + `os.replace` (atomic, so
a crash can't corrupt the live log). It's called from `persist()`, so the log
self-bounds with no external cron. Both thresholds are env-overridable
(`DEEPSCRIPT_USAGE_MAX_BYTES`, `DEEPSCRIPT_USAGE_KEEP_LINES`).

## Rate-Limit Handling in `fetch_all_circleback.py`

Rate-limit handling is **not** delegated to third-party client libraries. It is implemented directly within the project's own scripts (`fetch_all_circleback.py`).

- **Mechanism:** Empirically-tuned delays discovered through trial and error.
- **Safety net:** Reactive backoff / stop-on-429 (HTTP 429 error) — the process halts or slows down when the API signals overload.

## Process: Verify Before You Search

During the rate-limit investigation, a recurring mistake was re-running generic search commands (like `grep`) on project directories when the underlying files were known to be missing or the context was unclear.

**Rule:** Always verify file existence first using `find` before attempting to read or search their contents. This prevents unnecessary resource usage and provides clearer diagnostic output.

## Summary

| Area | Lesson |
|---|---|
| API interaction | Trust the internal, empirically-tuned logic for rate-limiting (backoff/429 handling) |
| Project management | Prioritize verification (`find`) over searching (`grep`) when dealing with project structure |