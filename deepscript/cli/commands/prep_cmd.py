"""deepscript prep — Assemble working memory for call preparation."""

from __future__ import annotations

from typing import Optional

import typer

from deepscript.cli.output import CLIContext, OutputFormat, emit
from deepscript.cms_bridge.working_memory import assemble_working_memory, format_prep_markdown
from deepscript.config.settings import get_settings


def prep(
    call_type: str = typer.Argument(help="Call type to prep for (e.g., sales-call, discovery-call)."),
    config_file: Optional[str] = typer.Option(None, "--config", "-c", help="Config file path."),
    ctx: typer.Context = typer.Option(None, hidden=True),
) -> None:
    """Assemble call prep notes from previous call patterns and playbooks."""
    cli_ctx: CLIContext = ctx.obj if ctx and ctx.obj else CLIContext()
    settings = get_settings(config_path=config_file)
    store_path = settings.cms.store_path

    wm = assemble_working_memory(call_type, store_path)

    if cli_ctx.format in (OutputFormat.JSON, OutputFormat.QUIET, OutputFormat.YAML):
        emit(wm, cli_ctx)
    else:
        md = format_prep_markdown(wm)
        print(md)
