"""deepscript playbook — Generate and view playbooks from CMS episodes."""

from __future__ import annotations

from typing import Optional

import typer

from deepscript.cli.output import CLIContext, emit
from deepscript.cms_bridge.dashboard import generate_pmf_dashboard
from deepscript.cms_bridge.playbook import generate_playbook
from deepscript.config.settings import get_settings


def playbook(
    call_type: str = typer.Argument(help="Call type to generate playbook for (e.g., sales-call, discovery-call)."),
    config_file: Optional[str] = typer.Option(None, "--config", "-c", help="Config file path."),
    ctx: typer.Context = typer.Option(None, hidden=True),
) -> None:
    """Generate a playbook from analyzed call episodes in CMS."""
    cli_ctx: CLIContext = ctx.obj if ctx and ctx.obj else CLIContext()
    settings = get_settings(config_path=config_file)
    store_path = settings.cms.store_path

    md = generate_playbook(call_type, store_path)
    print(md)


def dashboard(
    dashboard_type: str = typer.Argument(default="pmf", help="Dashboard type (currently: pmf)."),
    config_file: Optional[str] = typer.Option(None, "--config", "-c", help="Config file path."),
    ctx: typer.Context = typer.Option(None, hidden=True),
) -> None:
    """Generate a cross-call analytics dashboard."""
    cli_ctx: CLIContext = ctx.obj if ctx and ctx.obj else CLIContext()
    settings = get_settings(config_path=config_file)
    store_path = settings.cms.store_path

    if dashboard_type == "pmf":
        md = generate_pmf_dashboard(store_path, settings.pmf.ellis_threshold)
        print(md)
    else:
        print(f"Unknown dashboard type: {dashboard_type}. Available: pmf")
