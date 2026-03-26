"""DeepScript CLI — Transcript Intelligence Engine.

Global options set via callback, commands registered directly on app.
Rich Console writes to stderr; structured output (JSON/YAML) goes to stdout.
"""

import os
from typing import Optional

import typer
from rich.console import Console

from deepscript import __version__
from deepscript.cli.output import CLIContext, auto_detect_format

app = typer.Typer(
    name="deepscript",
    help="Transcript Intelligence Engine — classification, insights, and analysis.",
    add_completion=False,
    invoke_without_command=True,
    no_args_is_help=True,
)


def version_callback(value: bool) -> None:
    if value:
        print(f"DeepScript {__version__}")
        raise typer.Exit()


@app.callback()
def global_options(
    ctx: typer.Context,
    format: Optional[str] = typer.Option(
        None,
        "--format",
        "-o",
        help="Output format: json, table, quiet, yaml, markdown. Default: auto-detect.",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Suppress UI output, emit minimal JSON.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate inputs without processing.",
    ),
    fields: Optional[str] = typer.Option(
        None,
        "--fields",
        help="Comma-separated dot-notation fields to include in output.",
    ),
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """Global options applied to all subcommands."""
    effective_format = format or os.environ.get("DEEPSCRIPT_FORMAT", "auto")
    resolved_format = auto_detect_format(effective_format, quiet)

    if resolved_format.value in ("json", "quiet", "yaml"):
        console = Console(stderr=True)
    else:
        console = Console()

    parsed_fields = None
    if fields:
        parsed_fields = [f.strip() for f in fields.split(",") if f.strip()]

    ctx.ensure_object(dict)
    ctx.obj = CLIContext(
        format=resolved_format,
        dry_run=dry_run,
        fields=parsed_fields,
        console=console,
    )


# Register commands directly on app
from deepscript.cli.commands.analyze import analyze  # noqa: E402
from deepscript.cli.commands.benchmark_cmd import benchmark  # noqa: E402
from deepscript.cli.commands.classify_cmd import classify  # noqa: E402
from deepscript.cli.commands.playbook_cmd import dashboard, playbook  # noqa: E402
from deepscript.cli.commands.prep_cmd import prep  # noqa: E402
from deepscript.cli.commands.speakers_cmd import speakers  # noqa: E402
from deepscript.cli.commands.usage_cmd import usage  # noqa: E402

app.command(name="analyze", help="Analyze transcript for insights.")(analyze)
app.command(name="classify", help="Classify transcript type.")(classify)
app.command(name="usage", help="View LLM token usage and costs.")(usage)
app.command(name="playbook", help="Generate playbook from CMS episodes.")(playbook)
app.command(name="dashboard", help="Generate cross-call analytics dashboard.")(dashboard)
app.command(name="prep", help="Assemble call prep notes.")(prep)
app.command(name="benchmark", help="Benchmark LLM models for transcript analysis.")(benchmark)
app.command(name="speakers", help="Cross-call speaker identification.")(speakers)


if __name__ == "__main__":
    app()
