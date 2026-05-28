"""deepscript import — Import transcripts from Circleback, Fireflies, or other sources."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from deepscript.cli.output import CLIContext, OutputFormat, emit


def import_transcripts_cmd(
    source: str = typer.Argument(help="Source: fireflies | circleback"),
    output_dir: str = typer.Option("transcripts/imported", "--output-dir", "-o", help="Where to save imported transcripts."),
    days: int = typer.Option(30, "--days", "-d", help="Fetch meetings from last N days."),
    meeting_id: Optional[str] = typer.Option(None, "--meeting-id", "-m", help="Import a specific meeting by ID."),
    analyze: bool = typer.Option(False, "--analyze", "-a", help="Run deepscript analyze after import."),
    ctx: typer.Context = typer.Option(None, hidden=True),
) -> None:
    """Import transcripts from external meeting services for deep analysis."""
    cli_ctx: CLIContext = ctx.obj if ctx and ctx.obj else CLIContext()

    cli_ctx.console.print(f"[bold]Importing from {source}...[/bold]")

    if source == "fireflies":
        from deepscript.integrations.fireflies_import import import_fireflies
        result_dict = import_fireflies(days=days, output_dir=output_dir, limit=50)
        from dataclasses import dataclass
        @dataclass
        class _R:
            source: str; imported: int; skipped: int; failed: int; files: list
        result = _R(source=source, **result_dict)
    else:
        from deepscript.integrations.transcript_import import import_transcripts
        result = import_transcripts(source=source, output_dir=output_dir, days=days, meeting_id=meeting_id)

    if cli_ctx.format in (OutputFormat.JSON, OutputFormat.QUIET):
        emit({
            "source": result.source,
            "imported": result.imported,
            "skipped": result.skipped,
            "failed": result.failed,
            "files": result.files,
        }, cli_ctx)
    else:
        cli_ctx.console.print(f"[green]Imported:[/green] {result.imported} transcripts")
        if result.skipped:
            cli_ctx.console.print(f"[yellow]Skipped:[/yellow] {result.skipped} (already exist or empty)")
        if result.failed:
            cli_ctx.console.print(f"[red]Failed:[/red] {result.failed}")
        for f in result.files:
            cli_ctx.console.print(f"  {f}")

    # Auto-analyze if requested
    if analyze and result.files:
        cli_ctx.console.print(f"\n[bold]Analyzing {len(result.files)} imported transcripts...[/bold]")
        import subprocess
        cmd = ["deepscript", "analyze", output_dir, "-r", "--new-only"]
        subprocess.run(cmd)
