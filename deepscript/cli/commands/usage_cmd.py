"""deepscript usage — View LLM token usage and costs."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import typer

from deepscript.cli.output import CLIContext, emit
from deepscript.config.settings import get_settings
from deepscript.llm.cost_tracker import clear_usage, load_usage_history, usage_summary


def usage(
    days: Optional[int] = typer.Option(
        None,
        "--days",
        "-d",
        help="Show usage for last N days.",
    ),
    month: Optional[str] = typer.Option(
        None,
        "--month",
        "-m",
        help="Show usage for a specific month (YYYY-MM). Default: current month.",
    ),
    all_time: bool = typer.Option(
        False,
        "--all",
        help="Show all-time usage.",
    ),
    clear: bool = typer.Option(
        False,
        "--clear",
        help="Clear usage history.",
    ),
    ctx: typer.Context = typer.Option(None, hidden=True),
) -> None:
    """View LLM token usage, costs, and budget status."""
    cli_ctx: CLIContext = ctx.obj if ctx and ctx.obj else CLIContext()

    if clear:
        count = clear_usage()
        emit({"cleared": count, "message": f"Cleared {count} usage entries."}, cli_ctx)
        return

    settings = get_settings()

    # Determine filter
    filter_days = days
    filter_month = month
    if not all_time and days is None and month is None:
        # Default to current month
        filter_month = datetime.now(timezone.utc).strftime("%Y-%m")

    entries = load_usage_history(days=filter_days, month=filter_month)
    result = usage_summary(entries, budget_limit=settings.llm.budget_per_month)

    # Add filter info
    if filter_month:
        result["period"] = filter_month
    elif filter_days:
        result["period"] = f"last {filter_days} days"
    else:
        result["period"] = "all time"

    emit(result, cli_ctx)
