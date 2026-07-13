# Lessons Learned

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