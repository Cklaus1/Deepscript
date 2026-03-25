"""deepscript analyze — Main analysis command with parallel batch processing."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import typer

from deepscript.analyzers import build_analyzer_registry
from deepscript.analyzers.base import AnalysisResult, BaseAnalyzer
from deepscript.cli.output import CLIContext, ExitCode, OutputFormat, emit, emit_error
from deepscript.cms_bridge.episode import build_episode
from deepscript.cms_bridge.writer import write_episode
from deepscript.config.settings import DeepScriptConfig, get_settings
from deepscript.core.classifier import Classification, classify_transcript
from deepscript.core.communication import CommunicationMetrics, analyze_communication
from deepscript.core.tagger import generate_tags
from deepscript.core.topic_segmenter import Topic, segment_topics
from deepscript.formatters.json_formatter import format_json
from deepscript.formatters.markdown_formatter import format_markdown
from deepscript.integrations.calendar import CalendarContext, get_calendar_context
from deepscript.integrations.notifications import send_notifications
from deepscript.llm.provider import LLMProvider
from deepscript.utils.manifest import MANIFEST_FILENAME, ProcessingManifest

logger = logging.getLogger(__name__)


@dataclass
class AnalysisContext:
    """All intermediate results from analyzing a single transcript."""

    file_path: Path
    transcript: dict[str, Any]
    classification: Classification
    communication: CommunicationMetrics | None = None
    topics: list[Topic] | None = None
    analysis: AnalysisResult | None = None
    calendar_context: CalendarContext | None = None
    tags: dict[str, Any] = field(default_factory=dict)
    json_result: dict[str, Any] = field(default_factory=dict)


# --- Helpers ---


def _load_transcript(file_path: Path) -> dict:
    if not file_path.exists():
        raise FileNotFoundError(f"Transcript not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "text" not in data and "segments" not in data:
        raise ValueError(f"Invalid transcript: {file_path} must have 'text' or 'segments'")
    if "text" not in data and "segments" in data:
        data["text"] = " ".join(s.get("text", "") for s in data["segments"])
    return data


def _collect_files(file_arg: str, recursive: bool) -> list[Path]:
    path = Path(file_arg)
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(path.glob("**/*.json" if recursive else "*.json"))
    parent = path.parent if path.parent.exists() else Path(".")
    return sorted(parent.glob(path.name))


def _save_output(file_path: Path, result: dict[str, Any], out_path: Path) -> None:
    try:
        out_path.mkdir(parents=True, exist_ok=True)
        with open(out_path / f"{file_path.stem}.analysis.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    except (PermissionError, OSError) as e:
        logger.warning("Failed to save output for %s: %s", file_path, e)


# --- Single file analysis ---


def _analyze_single(
    file_path: Path,
    settings: DeepScriptConfig,
    llm: LLMProvider | None,
    analyzers: dict[str, BaseAnalyzer],
    call_type_override: str | None,
    calendar_enabled: bool,
    relationship_insights: bool,
    cms_enabled: bool,
) -> AnalysisContext | None:
    """Analyze a single transcript. Thread-safe."""
    start_time = time.time()

    try:
        transcript = _load_transcript(file_path)
    except (FileNotFoundError, ValueError, json.JSONDecodeError, PermissionError) as e:
        logger.warning("Skipping %s: %s", file_path, e)
        return None

    # Speaker enrichment — map diarization metadata to segments
    from deepscript.core.speaker_enrichment import enrich_speakers
    enrich_speakers(transcript)

    # Calendar
    calendar_context: CalendarContext | None = None
    if calendar_enabled and settings.calendar.enabled:
        try:
            recording_time = transcript.get("metadata", {}).get("audio", {}).get("creation_time")
            duration = transcript.get("metadata", {}).get("audio", {}).get("duration_seconds", 0)
            calendar_context = get_calendar_context(recording_time, duration, settings.calendar)
        except Exception as e:
            logger.warning("Calendar failed for %s: %s", file_path, e)

    # Classify
    if call_type_override:
        classification = Classification(call_type=call_type_override, confidence=1.0, scores={call_type_override: 1.0})
    elif settings.classify:
        classification = classify_transcript(transcript, settings.custom_classifications, llm=llm)
    else:
        classification = Classification(call_type="unknown", confidence=0.0, scores={})

    if classification.call_type in ("family", "partner", "personal") and not relationship_insights:
        classification = Classification(call_type="business-meeting", confidence=classification.confidence, scores=classification.scores)

    # Communication
    communication = analyze_communication(transcript) if settings.communication.enabled else None

    # Topics
    topics: list[Topic] | None = None
    if settings.topics.enabled:
        try:
            topics = segment_topics(transcript, llm=llm, min_duration=settings.topics.min_duration,
                                     max_topics=settings.topics.max_topics, method=settings.topics.method)
        except Exception as e:
            logger.warning("Topic segmentation failed for %s: %s", file_path, e)

    # Analyze
    analyzer = analyzers.get(classification.call_type, analyzers.get("unknown"))
    if analyzer is None:
        from deepscript.analyzers.specialized import SimpleAnalyzer
        analyzer = SimpleAnalyzer(llm=llm)
    analysis = analyzer.analyze(transcript)
    analysis.call_type = classification.call_type

    # Tags + format
    tags = generate_tags(classification, communication, topics, source_file=file_path.name)
    json_result = format_json(classification, communication, analysis, topics=topics,
                               source_file=str(file_path), sections_filter=settings.output.sections)
    json_result["tags"] = tags
    if calendar_context:
        json_result["calendar_context"] = calendar_context.to_dict()

    # CMS
    if cms_enabled:
        try:
            elapsed_ms = int((time.time() - start_time) * 1000)
            episode = build_episode(classification, analysis, communication,
                                     source_file=str(file_path), model=settings.llm.model if llm else "",
                                     execution_time_ms=elapsed_ms)
            ep_path = write_episode(episode, settings.cms.store_path)
            json_result["cms_episode"] = {"episode_id": episode.episode_id, "store_path": str(ep_path)}
        except Exception as e:
            logger.warning("CMS write failed for %s: %s", file_path, e)

    # Notifications
    if settings.notifications.enabled:
        try:
            summary_text = analysis.sections.get("summary", {}).get("text", "")[:500] if analysis else ""
            send_notifications(settings.notifications, call_type=classification.call_type, summary=summary_text, title=file_path.name)
        except Exception as e:
            logger.warning("Notification failed for %s: %s", file_path, e)

    return AnalysisContext(
        file_path=file_path, transcript=transcript, classification=classification,
        communication=communication, topics=topics, analysis=analysis,
        calendar_context=calendar_context, tags=tags, json_result=json_result,
    )


# --- Parallel batch processing ---


async def _analyze_parallel(
    files: list[Path],
    settings: DeepScriptConfig,
    llm: LLMProvider | None,
    analyzers: dict[str, BaseAnalyzer],
    call_type: str | None,
    calendar: bool,
    relationship_insights: bool,
    cms_enabled: bool,
    new_only: bool,
    force: bool,
    manifest: ProcessingManifest | None,
    output_dir: str | None,
    concurrency: int,
    show_progress: bool,
) -> tuple[list[AnalysisContext], int]:
    """Process files in parallel using asyncio + thread pool.

    Returns (contexts, skipped_count).
    """
    sem = asyncio.Semaphore(concurrency)
    contexts: list[AnalysisContext] = []
    skipped = 0
    lock = asyncio.Lock()

    # Filter files first
    files_to_process: list[Path] = []
    for fp in files:
        if new_only and manifest and not force and manifest.is_processed(fp):
            skipped += 1
        else:
            files_to_process.append(fp)

    if not files_to_process:
        return contexts, skipped

    async def process_one(fp: Path, progress_cb: Any = None) -> None:
        nonlocal skipped
        async with sem:
            # Run the sync analysis in a thread to not block the event loop
            ac = await asyncio.to_thread(
                _analyze_single, fp, settings, llm, analyzers,
                call_type, calendar, relationship_insights, cms_enabled,
            )

        async with lock:
            if ac:
                contexts.append(ac)
                if manifest:
                    manifest.record(fp, "completed", call_type=ac.classification.call_type)
                if output_dir:
                    _save_output(fp, ac.json_result, Path(output_dir))
            else:
                if manifest:
                    manifest.record(fp, "failed")

        if progress_cb:
            progress_cb(fp, ac)

    if show_progress:
        from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TextColumn("{task.fields[status]}"),
        ) as progress:
            task = progress.add_task("Analyzing", total=len(files), status="", completed=skipped)

            def on_done(fp: Path, ac: AnalysisContext | None) -> None:
                status = f"[green]{ac.classification.call_type}[/green]" if ac else "[red]failed[/red]"
                progress.update(task, advance=1, status=status)

            tasks = [process_one(fp, on_done) for fp in files_to_process]
            await asyncio.gather(*tasks, return_exceptions=True)
            progress.update(task, status=f"[bold green]Done — {len(contexts)} processed[/bold green]")
    else:
        tasks = [process_one(fp) for fp in files_to_process]
        await asyncio.gather(*tasks, return_exceptions=True)

    return contexts, skipped


# --- CLI command ---


def analyze(
    file: str = typer.Argument(help="Path to transcript JSON file or directory."),
    call_type: Optional[str] = typer.Option(None, "--type", "-t", help="Override auto-classification."),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", help="Save analysis output."),
    config_file: Optional[str] = typer.Option(None, "--config", "-c", help=".deepscript.yaml path."),
    no_llm: bool = typer.Option(False, "--no-llm", help="Rule-based only, no LLM API calls."),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Process directories recursively."),
    new_only: bool = typer.Option(False, "--new-only", help="Skip already-analyzed files."),
    force: bool = typer.Option(False, "--force", help="Re-analyze all files."),
    cms: bool = typer.Option(False, "--cms", help="Write CMS episode after analysis."),
    calendar: bool = typer.Option(False, "--calendar", help="Enrich with calendar context."),
    relationship_insights: bool = typer.Option(False, "--relationship-insights", help="Enable relationship analysis."),
    notify: bool = typer.Option(False, "--notify", help="Send notifications after analysis."),
    parallel: bool = typer.Option(False, "--parallel", help="Process files in parallel (async)."),
    concurrency: Optional[int] = typer.Option(None, "--concurrency", help="Max parallel files (default from config)."),
    ctx: typer.Context = typer.Option(None, hidden=True),
) -> None:
    """Analyze transcripts for classification, communication metrics, and insights."""
    cli_ctx: CLIContext = ctx.obj if ctx and ctx.obj else CLIContext()

    try:
        config_path = Path(config_file) if config_file else None
        settings = get_settings(config_path=config_path)
        if cms:
            settings.cms.enabled = True
        if notify:
            settings.notifications.enabled = True

        files = _collect_files(file, recursive)
        if not files:
            emit_error(f"No transcript files found: {file}", cli_ctx, ExitCode.VALIDATION_ERROR)
            raise typer.Exit(code=ExitCode.VALIDATION_ERROR)

        if cli_ctx.dry_run:
            emit({"dry_run": True, "files": [str(f) for f in files], "count": len(files)}, cli_ctx)
            return

        llm: LLMProvider | None = None
        if not no_llm:
            llm = LLMProvider.create(settings.llm)

        analyzers = build_analyzer_registry(llm=llm, settings=settings)

        manifest: ProcessingManifest | None = None
        manifest_path: Path | None = None
        if new_only or len(files) > 1:
            manifest_dir = Path(output_dir) if output_dir else Path(".")
            manifest_path = manifest_dir / MANIFEST_FILENAME
            manifest = ProcessingManifest.load(manifest_path)

        is_batch = len(files) > 1
        show_progress = is_batch and sys.stdout.isatty() and cli_ctx.format not in (
            OutputFormat.JSON, OutputFormat.QUIET, OutputFormat.YAML)
        use_parallel = parallel or (is_batch and llm is not None)
        max_concurrency = concurrency or settings.llm.concurrency

        contexts: list[AnalysisContext] = []
        skipped = 0
        interrupted = False

        try:
            if use_parallel and is_batch:
                # Parallel batch with asyncio
                contexts, skipped = asyncio.run(
                    _analyze_parallel(
                        files, settings, llm, analyzers, call_type, calendar,
                        relationship_insights, cms or settings.cms.enabled,
                        new_only, force, manifest, output_dir,
                        max_concurrency, show_progress,
                    )
                )
            elif show_progress:
                _process_sequential_with_progress(
                    files, settings, llm, analyzers, call_type, calendar,
                    relationship_insights, cms, new_only, force, manifest,
                    output_dir, contexts,
                )
                skipped = len(files) - len(contexts)
            else:
                for fp in files:
                    if new_only and manifest and not force and manifest.is_processed(fp):
                        skipped += 1
                        continue

                    ac = _analyze_single(
                        fp, settings, llm, analyzers,
                        call_type_override=call_type,
                        calendar_enabled=calendar,
                        relationship_insights=relationship_insights,
                        cms_enabled=cms or settings.cms.enabled,
                    )
                    if ac:
                        contexts.append(ac)
                        if manifest:
                            manifest.record(fp, "completed", call_type=ac.classification.call_type)
                        if output_dir:
                            _save_output(fp, ac.json_result, Path(output_dir))
                    else:
                        if manifest:
                            manifest.record(fp, "failed")
        except KeyboardInterrupt:
            interrupted = True
            logger.info("Interrupted — saving progress (%d processed)", len(contexts))
        finally:
            if manifest and manifest_path:
                manifest.save(manifest_path)

        # Emit results
        if len(contexts) == 1 and not interrupted:
            ac = contexts[0]
            if cli_ctx.format == OutputFormat.MARKDOWN:
                print(format_markdown(ac.classification, ac.communication, ac.analysis,
                                       topics=ac.topics, source_file=ac.file_path.name))
            else:
                emit(ac.json_result, cli_ctx)
        else:
            result_summary: dict[str, Any] = {
                "processed": len(contexts),
                "skipped": skipped,
                "total": len(files),
                "results": [ac.json_result for ac in contexts],
            }
            if interrupted:
                result_summary["interrupted"] = True
                result_summary["remaining"] = len(files) - len(contexts) - skipped
            if use_parallel:
                result_summary["parallel"] = True
                result_summary["concurrency"] = max_concurrency
            emit(result_summary, cli_ctx)

        # LLM usage
        if llm and llm.config.cost_tracking:
            usage_info = llm.cost_tracker.summary()
            if usage_info["calls"] > 0:
                logger.info("LLM: %d calls, %d in, %d out, $%.4f",
                            usage_info["calls"], usage_info["total_input_tokens"],
                            usage_info["total_output_tokens"], usage_info["total_cost_usd"])
                llm.cost_tracker.persist(
                    source_file=str(files[0]) if len(files) == 1 else f"{len(files)} files",
                    call_type=contexts[-1].classification.call_type if contexts else "",
                )

    except (FileNotFoundError, ValueError) as e:
        emit_error(str(e), cli_ctx, ExitCode.VALIDATION_ERROR)
        raise typer.Exit(code=ExitCode.VALIDATION_ERROR)
    except Exception as e:
        logger.exception("Analysis failed")
        emit_error(str(e), cli_ctx, ExitCode.classify(e))
        raise typer.Exit(code=ExitCode.classify(e))


def _process_sequential_with_progress(
    files: list[Path],
    settings: DeepScriptConfig,
    llm: LLMProvider | None,
    analyzers: dict[str, BaseAnalyzer],
    call_type: str | None,
    calendar: bool,
    relationship_insights: bool,
    cms: bool,
    new_only: bool,
    force: bool,
    manifest: ProcessingManifest | None,
    output_dir: str | None,
    contexts: list[AnalysisContext],
) -> None:
    """Sequential processing with Rich progress bar."""
    from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("{task.fields[status]}"),
    ) as progress:
        task = progress.add_task("Analyzing", total=len(files), status="")

        for fp in files:
            progress.update(task, status=f"[dim]{fp.name}[/dim]")

            if new_only and manifest and not force and manifest.is_processed(fp):
                progress.advance(task)
                continue

            ac = _analyze_single(
                fp, settings, llm, analyzers,
                call_type_override=call_type,
                calendar_enabled=calendar,
                relationship_insights=relationship_insights,
                cms_enabled=cms or settings.cms.enabled,
            )

            if ac:
                contexts.append(ac)
                if manifest:
                    manifest.record(fp, "completed", call_type=ac.classification.call_type)
                if output_dir:
                    _save_output(fp, ac.json_result, Path(output_dir))
                progress.update(task, status=f"[green]{ac.classification.call_type}[/green]")
            else:
                if manifest:
                    manifest.record(fp, "failed")
                progress.update(task, status="[red]failed[/red]")

            progress.advance(task)

        progress.update(task, status=f"[bold green]Done — {len(contexts)} processed[/bold green]")
