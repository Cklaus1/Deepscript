# DeepScript — Architectural Review & Fix Spec

**Date:** 2026-07-06
**Method:** 6 parallel Fable-5 review agents across the full source tree (~19.5k LOC, ~60 files).
**Raw findings:** 102 (9 HIGH, 46 MEDIUM, 47 LOW). Several verified against live source by the architect.
**Baseline:** 235 tests passing; no bare `except:`, no mutable-default args.

## Scope decision

- **This pass fixes ALL HIGH + MEDIUM (55 items)** — real bugs affecting the running pipeline.
- **LOW items:** apply the *safe, mechanical* ones (dead code, unused imports/params, tag/table escaping) opportunistically inside each file's fix batch; **defer** LOW items that need judgment or new options (listed in §Deferred).
- Fixes are **partitioned by file into 7 disjoint groups** so parallel Sonnet fixers never touch the same file. Group G (shared refactors) runs the base-helper changes; groups that depend on those helpers only *call* them, they don't redefine them — and no two groups share a file.

## Cross-cutting root causes (theme)

1. **Swallowed failures → cron sees success on failure.** `asyncio.gather(return_exceptions=True)` results discarded (analyze, benchmark); debug-level `except` hiding a live `NameError` (speaker panel); `subprocess.run` return code ignored (import). *Highest-value theme.*
2. **`.get(k, default)` returns `None` when the key is present-but-null** → `None[:80]` / `None.replace()` crashes. Recurs across formatters and analyzers.
3. **Substring name matching** attaches evidence to the wrong person (3 files).
4. **Non-ASCII / unicode** dropped or collided (sanitize, speaker regex, tags).
5. **Path/key normalization** — manifest & output naming inconsistent across cwd → redundant reprocessing + clobbers.

---

## GROUP A — Speaker core (agent 1)
**Files:** `core/speaker_panel.py`, `core/name_corrections.py`, `core/speaker_signals.py`, `core/speaker_dossier.py`, `core/speaker_enrichment.py`
**Not** `speaker_intelligence.py` (that's Group B, to keep the 2011-line file single-owner).

- [x] **A1 (HIGH)** `speaker_panel.py:225` — `_find_candidates_for_speaker` references `profiles` but it's not a param → `NameError` aborts panel runs. Add `profiles: dict[str, Any]` param; pass `profiles` at call site line 120. **[VERIFIED]**
- [x] **A2 (HIGH)** `name_corrections.py:200` — `compile_corrections` joins keys with no word boundary → "jos" rewrites "Jose"→"Joshe". Wrap: `re.compile(r"\b(?:" + "|".join(escaped) + r")\b", re.IGNORECASE)`. **[VERIFIED]**
- [x] **A3 (MED)** `speaker_signals.py:638` — owner filter `startswith(account_owner_name)` backwards for first-name-only titles. Match owner first name too. **[VERIFIED]**
- [x] **A4 (LOW→incl)** `speaker_signals.py:139` — delete dead `extract_relationships`/`build_relationship_graph`/`compute_speaking_durations`/`match_duration_to_calendar` + `Relationship` dataclass (unused repo-wide; latent bugs). Confirm no imports first. **[VERIFIED]**
- [x] **A5 (LOW→incl)** `speaker_dossier.py:359` — dossier filename from label only → generic labels overwrite. Append `_{cluster_id[:12]}`. **[VERIFIED]**
- [x] **A6 (LOW→incl)** `speaker_enrichment.py:107` — `prev_end=0.0` misassigns first segment. Init to first segment's start. **[VERIFIED]**
- [x] **A7 (LOW→incl)** `speaker_panel.py:490/527` + dossier:212 — extract `_parse_llm_json` helper (dedupe fence-strip). *Keep helper in speaker_panel.py; dossier imports it.* **[VERIFIED]**
- [x] **A8 (LOW→incl)** `speaker_panel.py:88/115` — `_gather_speech_samples` called twice per speaker; cache in the `unknowns` tuples. **[VERIFIED]**

## GROUP B — speaker_intelligence.py (agent 1) — single file, single owner
**File:** `core/speaker_intelligence.py` only.

- [ ] **B1 (MED)** `:573` & `:592` — `except ... logger.debug("LLM panel skipped")` hides the A1 NameError. Raise to `logger.warning(..., exc_info=True)`.
- [ ] **B2 (MED)** `:174` — `total_calls += 1` per resolved entry double-counts when two local labels map to one cluster in a call. Skip clusters already counted this call (`if cid in call_clusters: continue` after the empty check).
- [ ] **B3 (MED)** `:167` — `local_label` fallback ("SPEAKER_00") merges unrelated speakers cross-call. Namespace per call: `cid = sr.get("speaker_cluster_id") or f"{call_id}:{sr.get('local_label','')}"` (or drop fallback).
- [ ] **B4 (MED)** `:1649` — `soft_merge_speakers` overwrites an existing `linked_to` → stat corruption. Refuse re-link: `if b.get("linked_to"): logger.warning(...); return False`.
- [ ] **B5 (MED)** `:1870` — `_hard_merge` `b["first_seen"]` KeyError / clobbers valid ts. Guard truthiness with sentinel `"￿"`.
- [ ] **B6 (MED)** `:1050` — `_names_match` raw substring ("Al"⊂"Salvador") + IndexError on blank. Guard empties; use token-subset containment.
- [ ] **B7 (LOW→incl)** `:424` — add `metadata.audio.creation_time` fallback to `transcript_events` (mirror lines 801-803).
- [ ] **B8 (LOW→incl)** `:1226`/`:1269` — remove redundant confidence guard + dead `else`.
- [ ] **B9 (LOW→incl)** `:1138` — `_is_upgrade` unused `new_confidence` param: use it (require higher confidence to replace) — *prefer using over deleting*; if ambiguous, delete param + call arg.
- [ ] **B10 (LOW→incl)** `:282/448/767` — extract `_detect_account_owner(events)` helper (dedupe 3 copies).

## GROUP C — Analyzers + classifier (agent 2)
**Files:** `analyzers/__init__.py`, `analyzers/base.py`, `analyzers/business.py`, `analyzers/sales.py`, `analyzers/discovery.py`, `analyzers/specialized.py`, `analyzers/pmf.py`, `analyzers/relationship.py`, `analyzers/support.py`, `analyzers/recruiting.py`, `analyzers/pitch.py`, `core/classifier.py`, `core/communication.py`, `core/topic_segmenter.py`, `core/chunk_handler.py`, `core/tagger.py`

- [ ] **C1 (HIGH)** `__init__.py:57` — duplicate `supported_types`: `"standup"` claimed by Business (business.py:48) AND Operations → all OperationsAnalyzer standup logic dead. Remove `"standup"` from `BusinessAnalyzer.supported_types`; add `logger.warning` on duplicate registration. **[VERIFIED]**
- [ ] **C2 (MED)** `__init__.py:59/62` — `except Exception: pass` + DEBUG import failures hide vanished analyzers. Raise to `logger.warning`.
- [ ] **C3 (MED)** `sales.py:68` — `{{ }}` in schema string reach LLM literally (render_prompt formats the template, not kwarg values). Replace `{{`/`}}` → `{`/`}`. **[VERIFIED]**
- [ ] **C4 (MED)** `business.py:59` & `:133` — `.get()` on list-of-strings items → AttributeError crashes `analyze()`. Filter `isinstance(i, dict)` first. (Root cause #2 sibling.)
- [ ] **C5 (MED)** `specialized.py:101` — `SimpleAnalyzer` returns Business's `call_type` ("business-meeting") for voice memo/medical/legal. Add `call_type` param wired via registry; return with correct type.
- [ ] **C6 (MED)** hardcoded result `call_type` disagrees with classification — `sales.py:107`, `discovery.py:82`, `support.py:54`, `recruiting.py:58`, `pitch.py:60`, `business.py:80`. Thread resolved `call_type` into `analyze()` / store on instance (registry knows `ct`).
- [ ] **C7 (MED)** `communication.py:119` — `speaking_balance` floor is 1/n, contradicts "0.0=one talks". Return 0.0 for n==1; normalize Gini by `(n-1)/n`. *(Note: has tests — update expectations.)*
- [ ] **C8 (MED)** `topic_segmenter.py:161` — `_name_topics_with_llm` ignores `boundaries`, duplicates full LLM segmentation. Either name boundary segments only, or route hybrid→`_segment_with_llm` with rule fallback.
- [ ] **C9 (MED)** `pmf.py:154` — rule-based composite averages 2 hardcoded-0 dims → caps score at 7.5, Ellis ≥7 unreachable. Exclude `evidence == "requires LLM"` dims from the average.
- [ ] **C10 (LOW→incl)** `classifier.py:180` — `except: pass` on keyword collection → `logger.warning`.
- [ ] **C11 (LOW→incl)** `classifier.py:193` — dead ternary `if patterns else 0.0`. Simplify.
- [ ] **C12 (LOW→incl)** `business.py:35`/`sales.py:41` — duplicate `classification_keywords` always ignored. Delete one source (prefer deleting the class attrs, keep CLASSIFICATION_KEYWORDS).
- [ ] **C13 (LOW→incl)** `base.py:8` — remove unused `lru_cache` import.
- [ ] **C14 (LOW→incl)** `tagger.py:42` — sanitize speaker tags (spaces → `-`) like topic tags.
- [ ] **C15 (LOW→incl)** `discovery.py:109` — delete unused `text_lower`.
- [ ] **C16 (LOW→incl)** `relationship.py:105` — remove unused `text` param from `_we_i_language` + call at :59.
- [ ] **C17 (LOW→incl)** `topic_segmenter.py:123` — `"."`-only summary on empty text; guard.
- [ ] **C18 (LOW→incl)** `chunk_handler.py:104` — `get_chunk_metadata` guard inconsistent with `is_chunked`. Use `if not is_chunked(transcript): return {}`.

## GROUP D — LLM / config / cms_bridge (agent 3)
**Files:** `llm/provider.py`, `llm/cost_tracker.py`, `utils/manifest.py`, `cms_bridge/episode.py`, `cms_bridge/writer.py`, `cms_bridge/dashboard.py`, `cms_bridge/working_memory.py`

- [ ] **D1 (HIGH)** `provider.py:237/295` — latency/provider patched onto `cost_tracker.entries[-1]` races under parallel batch (shared provider). Pass `latency_ms`/`provider` into `record()` directly; add `threading.Lock` in `CostTracker.record()`.
- [ ] **D2 (MED, security)** `provider.py:193` — for ollama/vllm/sglang the `OPENAI_API_KEY` fallback runs *before* the NO_KEY check → real OpenAI secret sent as bearer to arbitrary `base_url`. Reorder: NO_KEY_PROVIDERS → `api_key or "not-needed"`, never fall back to env; only `openai` consults `OPENAI_API_KEY`. (Extends the shipped NIM fix.)
- [ ] **D3 (MED)** `cost_tracker.py:60` — unknown models priced at Claude rates → free local providers accrue fake cost, flip budget breaker. Price 0.0 for no-cost providers / absent models.
- [ ] **D4 (MED)** `provider.py:99` — only `nim` gets default-model remap; openai/vllm/sglang send Claude model → permanent 404, silent None. Extend remap to all non-Claude, or error when non-claude + Claude default.
- [ ] **D5 (MED)** `provider.py:101` — monthly budget only checks in-memory session total (starts $0). Seed `total_cost_usd` from persisted usage history for current month in init.
- [ ] **D6 (MED)** `manifest.py:43/72` — key on raw `str(file_path)` → rel vs abs path reprocesses. Normalize `str(Path(file_path).resolve())` in both `is_processed`/`record` (keep reading legacy keys). *(Coordinate w/ CLI E4 which resolves paths before calling — both safe together.)*
- [ ] **D7 (LOW→incl)** `cost_tracker.py:104` — `persist()` misattributes call_type + never clears entries (dup rows on 2nd persist). Clear entries after write; only set fields when empty.
- [ ] **D8 (LOW→incl)** `episode.py:160` — hardcoded `provider="anthropic"`. Add `provider` param (caller passes settings.llm.provider). *(Caller edit is in analyze.py = Group E; coordinate: add param w/ default "" so E can pass it, non-breaking.)*
- [ ] **D9 (LOW→incl)** `writer.py:35` — prefix-match path guard is dead + latent. Replace with `resolved.parent == expected` or `relative_to`.
- [ ] **D10 (LOW→incl)** `dashboard.py:51` — `strongest_signals` collected, never rendered. Render a `## Strongest Signals` section (most_common(10)).
- [ ] **D11 (LOW→incl)** `working_memory.py:136` — `if r.get("overall_score")` drops legit 0.0. Use `is not None`.
- [ ] **D12 (LOW, defer-optional)** `manifest.py:138` — 16-char SHA truncation vs docstring. Low risk; fix docstring at minimum.

## GROUP E — CLI (agent 4)
**Files:** `cli/main.py`, `cli/output.py`, `cli/commands/*.py`

- [ ] **E1 (HIGH)** `analyze.py:408/412` — `gather(return_exceptions=True)` discarded → parallel failures silent, exit 0. Capture results; log + `manifest.record(fp,"failed")` per exception; exit non-zero when failures>0 and processed==0. **[VERIFIED]**
- [ ] **E2 (HIGH)** `speakers_cmd.py:269` — `--writeback` auto-merge block nested in `else` → `-q dedup --writeback` merges nothing. Move `if writeback:` out of `else`; emit structured `merged` count for JSON/QUIET/YAML. **[VERIFIED]**
- [ ] **E3 (MED)** `analyze.py:452/610` — `typer.Exit` caught by `except Exception` → wrong exit code + spurious traceback. Add `except typer.Exit: raise` first.
- [ ] **E4 (MED)** `analyze.py:449/467` — manifest keys = paths as passed; cwd-relative manifest dir. Resolve files after `_collect_files`; default manifest_dir to input dir. *(Pairs with D6.)*
- [ ] **E5 (MED)** `analyze.py:83` — `_save_output` names by `.stem` → recursive same-name collisions clobber. Derive collision-free name from path relative to input root.
- [ ] **E6 (MED)** `analyze.py:75` — `_collect_files` globs dotfiles/manifest/`*.analysis.json`/`speaker_identities.json`. Filter them out. **[VERIFIED glob picks dotfiles]**
- [ ] **E7 (MED)** `analyze.py:498` — sequential-with-progress miscounts failed/deferred as `skipped`. Return real skipped count.
- [ ] **E8 (MED)** `analyze.py:444` — `--calendar` doesn't force-enable settings (unlike --cms/--notify). Add `if calendar: settings.calendar.enabled = True`.
- [ ] **E9 (MED)** `import_cmd.py:60` — `subprocess.run` return code ignored → `import -a` exits 0 on analysis failure. Check rc, raise `typer.Exit(rc)`; handle FileNotFoundError.
- [ ] **E10 (MED)** `speakers_cmd.py:132/312/350/353` — error prints return exit 0. Add `raise typer.Exit(1)`.
- [ ] **E11 (MED)** `benchmark_cmd.py:142` — `--top` repurposed as `max_parallel` → 50 workers vs 35rpm. Add dedicated `--max-parallel` (default 5).
- [ ] **E12 (MED)** `main.py:69` — markdown format writes console to stdout, interleaves piped doc. Add "markdown" to stderr-console set.
- [ ] **E13 (LOW→incl)** `output.py:79` — `filter_fields` crashes on non-dict intermediate (`--fields a.b.c`). Guard `isinstance(src, dict)`.
- [ ] **E14 (LOW→incl)** `analyze.py:550` — dead CMS candidate + hardcoded cwd paths ignore `--output-dir`. Fix candidate list; derive analysis_dir from output_dir.
- [ ] **E15 (LOW→incl)** `benchmark_cmd.py:208` — `_compare_runs` negative-index not bounded. Validate `>=0`.
- [ ] **E16 (LOW→incl)** `playbook_cmd.py:43` — invalid dashboard type prints stdout + exit 0. `emit_error` + `typer.Exit(VALIDATION_ERROR)`.
- [ ] **E17 (LOW→incl)** `speakers_cmd.py:242` — remove dead `unmerge_speakers` import in dedup branch.
- [ ] **E18 (LOW→incl)** `analyze.py:518` — `KeyboardInterrupt` returns 0. `if interrupted: raise typer.Exit(130)`.

## GROUP F — Integrations (agent 5)
**Files:** `integrations/transcript_import.py`, `integrations/calendar.py`, `integrations/notifications.py`
**Not** `integrations/minotes.py` → Group F2 (avoid overlap with my recently-shipped collision fix + its own big finding set).

- [ ] **F1 (HIGH)** `transcript_import.py:654` — non-atomic `json.dump` to final path; killed mid-write leaves truncated file that `.exists()` treats as imported forever. Write `.tmp` + `os.replace`.
- [ ] **F2 (MED)** `transcript_import.py:378/511` — `strptime(date_str,"%Y-%m-%d")` raises on ISO dateString, swallowed → `--days` filter silently disabled. Parse `date_str[:10]`.
- [ ] **F3 (MED)** `transcript_import.py:349/484` — cutoff keeps microseconds → boundary-day meetings dropped (off-by-one). Add `microsecond=0`.
- [ ] **F4 (MED)** `transcript_import.py:308/279` — failed Fireflies fetch swallowed, counted "skipped" → outage looks like success. Log error, count as failed (tag `fetch_failed`), optional retry.
- [ ] **F5 (MED)** `transcript_import.py:699` — speaker regex `^([A-Z]...)` ASCII-only drops non-ASCII/lowercase speakers entirely. Unicode-aware class `^([^\W\d_][^:]{1,50}):`.
- [ ] **F6 (LOW→incl)** `transcript_import.py:385/518/601/625` — `len(segments)>1` misclassifies single-segment transcript. Use presence of `segments` key (`>=1`).
- [ ] **F7 (LOW→incl)** `transcript_import.py:319` — `_fetch_fireflies_local` ignores `to_date`. Apply it.
- [ ] **F8 (LOW→incl)** `transcript_import.py:428/455` — `int(mid)` unguarded crashes whole run on bad ID. try/except ValueError, continue.
- [ ] **F9 (MED)** `calendar.py:74/149` — naive datetime formatted with hardcoded "Z" → tz mislabel, wrong window. Normalize to UTC before formatting.
- [ ] **F10 (MED)** `calendar.py:90` — ms365 path emits naive UTC strings, CLI reads as local → window shift. Convert to local or emit offset.
- [ ] **F11 (LOW→incl)** `calendar.py:23` — `attendees: list[str] = None` → `field(default_factory=list)`, drop `__post_init__`.
- [ ] **F12 (LOW→incl)** `calendar.py:118/186` — `except (JSONDecodeError, Exception)` = `except Exception`, hides real errors. Split handlers / `logger.exception`.
- [ ] **F13 (LOW→incl)** `notifications.py:49` — sequential placeholder `.replace` corrupts if summary contains `{title}` etc. Single-pass `re.sub` with `shlex.quote`.
- [ ] **F14 (note)** packaging: `_archive_fireflies_import.py` ships importable — add setuptools exclude or move out of package. (pyproject edit — architect handles, not a fixer.)

## GROUP F2 — minotes.py (agent 5) — single file, single owner
**File:** `integrations/minotes.py` only. (Already partially fixed for interaction collisions; these are the remaining items.)

- [ ] **F2a (HIGH)** `:572` — `_sanitize` strips all non-ASCII → "李明"/"王芳" both → "untitled" (person pages collide); accented names lose letters. Keep unicode word chars: `re.sub(r"[^\w \-]", "", name, flags=re.UNICODE).strip() or "untitled"`.
- [ ] **F2b (MED)** `:80/93` — person/company pages still last-writer-wins on same sanitized name. Mirror interaction disambiguation (Counter + cluster_id suffix).
- [ ] **F2c (MED)** `:223` — action-item assignment `speaker_name in assignee` substring → "Chris" claims "Chris Klaus" & "Chris Erven"; dup tasks. Token/exact match, longest-match wins.
- [ ] **F2d (MED)** `:596` — legacy `generate_contact_pages` globs top-level `*.md` but pages live in subdirs → returns nothing. Return flattened `written` values.
- [ ] **F2e (LOW→incl)** `:256/467` — frontmatter values raw-quoted → breaks YAML on embedded `"`. Use `json.dumps(value)`.
- [ ] **F2f (LOW→incl)** `:560` — `_write_task_index` unescaped `|` corrupts table + shows raw cluster_id. Escape pipes; pass profiles for display_name.
- [ ] **F2g (LOW→incl)** `:288` — unused `contact_company`. Emit as company fallback, or delete.

## GROUP H — Formatters + benchmark (agent 6)
**Files:** `formatters/markdown_formatter.py`, `formatters/json_formatter.py`, `benchmark/runner.py`, `benchmark/ground_truth.py`, `benchmark/history.py`

- [ ] **H1 (HIGH)** `runner.py:504` + `history.py:22/127` — `benchmark-latest.json` matches `benchmark-*.json` glob → history double-counts + treats partial as newest. Rename incremental to `latest.json` (outside glob) AND skip it explicitly in history.
- [ ] **H2 (MED)** `markdown_formatter.py:188/278/409/583/584` — `.get(k,"")[:N]` / `.get()` chains crash on present-null (`None[:80]`). Apply `str(x or "")[:N]` / `(x or {})` pattern (root cause #2, main site).
- [ ] **H3 (MED)** `markdown_formatter.py:14` — `_esc` doesn't strip newlines → embedded `\n` splits table rows. Add `.replace("\n"," ").replace("\r","")`.
- [ ] **H4 (MED)** `markdown_formatter.py:327` — `_render_pmf` renders "Score: None/10" when dims exist but score missing. Conditional heading.
- [ ] **H5 (MED)** `runner.py:334/339` — `item.get("text", item.get("quote", str(item)))` yields None on present-null → verify crash zeroes valid response. `v = item.get("text") or item.get("quote") or str(item)`.
- [ ] **H6 (MED)** `runner.py:588` — `asyncio.gather(return_exceptions=True)` discards exceptions → failed model vanishes. Log + append failed ModelBenchmark. (Theme #1.)
- [ ] **H7 (MED)** `ground_truth.py:138` — empty-GT returns f1=0 → correct extraction on empty-action fixtures zeroed, inconsistent w/ runner's 0.5. Return neutral for empty GT.
- [ ] **H8 (MED)** `ground_truth.py:193` — grounding needs only 2 bag-of-words overlaps → stopwords ground anything, anti-hallucination check is a no-op. Require ≥50% of non-stopword item words.
- [ ] **H9 (LOW→incl)** `markdown_formatter.py:445` — `_render_support` `.get('type','unknown').replace` crashes on null type. `(x or 'unknown')`.
- [ ] **H10 (LOW→incl)** `markdown_formatter.py:539/542` — dead `we_i`; `validation_moments`/`growth` missing from `any()` gate → dropped. Add to gate (+ render or delete we_i).
- [ ] **H11 (LOW→incl)** `json_formatter.py:42` — non-"all" string filter drops all sections. Treat as single-section filter.
- [ ] **H12 (LOW, defer)** `runner.py:48` — `_get_rate_limiter` ignores rpm after first + unsynced. Make rate_limiter required param. *(Behavioral; defer to avoid benchmark churn.)*
- [ ] **H13 (LOW→incl)** `markdown_formatter.py:616` — extract shared `_render_speaker_table`/`_render_signal_list`/`_render_dimension_table` (so the H2/H3 escaping fix isn't applied 3×). *Do H2/H3 first, then dedupe.*

---

## Deferred (LOW, need judgment / new surface — not this pass)
- CLI `ctx` boilerplate refactor across 10 commands (E-agent LOW) — large mechanical churn, no bug; separate PR.
- `speakers_cmd.py` `_resolve_db_path` helper extraction — same.
- manifest full-hash migration (D12 beyond docstring) — needs version bump + migration.
- benchmark rate-limiter redesign (H12) — behavioral.

## Execution plan
1. Architect handles the two **non-fixer** edits directly: **F14** (pyproject exclude) and confirms **C7/communication** test expectations exist.
2. Fan out **7 Sonnet fixers**, one per group (A, B, C, D, E, F, F2, H) — **disjoint files, no write conflicts**. (8 agents; F and F2 split integrations by file.)
3. Each fixer: make edits, run the **relevant** test file(s), report what changed + any finding it judged a false positive (do NOT force a fix that isn't real).
4. Architect: run full `pytest` (all 235+), fix any cross-group breakage, update test expectations for behavioral changes (C7, H7/H8 scoring).
5. Update `tasks/lessons.md`; commit in logical groups.

## Review section
_(filled in after implementation)_
