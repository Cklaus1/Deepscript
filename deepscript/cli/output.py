"""Structured output formatting for DeepScript CLI."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional

import yaml
from rich.console import Console


class OutputFormat(str, Enum):
    JSON = "json"
    TABLE = "table"
    QUIET = "quiet"
    YAML = "yaml"
    MARKDOWN = "markdown"


class ExitCode:
    SUCCESS = 0
    ANALYSIS_ERROR = 1
    AUTH_ERROR = 2
    VALIDATION_ERROR = 3
    INTERNAL_ERROR = 4

    @staticmethod
    def classify(exc: Exception) -> int:
        """Map exception types to exit codes."""
        msg = str(exc).lower()
        if "token" in msg or "auth" in msg or "api_key" in msg or "credentials" in msg:
            return ExitCode.AUTH_ERROR
        if isinstance(exc, (ValueError, FileNotFoundError, TypeError)):
            return ExitCode.VALIDATION_ERROR
        if isinstance(exc, (RuntimeError, OSError)):
            return ExitCode.ANALYSIS_ERROR
        return ExitCode.INTERNAL_ERROR


@dataclass
class CLIContext:
    """Shared state across all subcommands."""

    format: OutputFormat = OutputFormat.JSON
    dry_run: bool = False
    fields: Optional[List[str]] = None
    console: Console = field(default_factory=lambda: Console(stderr=True))
    start_time: float = field(default_factory=time.time)

    @property
    def is_structured(self) -> bool:
        """True when output should be machine-readable."""
        return self.format in (OutputFormat.JSON, OutputFormat.QUIET, OutputFormat.YAML)


def auto_detect_format(explicit: Optional[str], quiet: bool) -> OutputFormat:
    """Determine output format from flags or TTY detection."""
    if quiet:
        return OutputFormat.QUIET
    if explicit and explicit != "auto":
        return OutputFormat(explicit)
    if sys.stdout.isatty():
        return OutputFormat.TABLE
    return OutputFormat.JSON


def filter_fields(data: Any, fields: List[str]) -> Any:
    """Filter data dict to only include requested dot-notation fields."""
    if not isinstance(data, dict):
        return data
    result: dict[str, Any] = {}
    for f in fields:
        parts = f.split(".")
        src, dst = data, result
        for i, part in enumerate(parts[:-1]):
            if part not in src:
                break
            if part not in dst:
                dst[part] = {}
            src = src[part]
            dst = dst[part]
        else:
            last = parts[-1]
            if last in src:
                dst[last] = src[last]
    return result


def emit(data: Any, ctx: CLIContext) -> None:
    """Emit structured output to stdout."""
    if ctx.fields:
        data = filter_fields(data, ctx.fields)

    if ctx.format in (OutputFormat.JSON, OutputFormat.QUIET):
        print(json.dumps(data, indent=2, ensure_ascii=False, default=str))
    elif ctx.format == OutputFormat.YAML:
        print(yaml.dump(data, default_flow_style=False, allow_unicode=True), end="")
    elif ctx.format == OutputFormat.MARKDOWN:
        # Markdown formatting is handled by the markdown formatter
        # This fallback just dumps JSON
        print(json.dumps(data, indent=2, ensure_ascii=False, default=str))
    else:
        # Table format — simple key-value for now
        _emit_table(data, ctx.console)


def _emit_table(data: Any, console: Console) -> None:
    """Render data as a Rich table."""
    from rich.table import Table

    if isinstance(data, dict):
        table = Table(show_header=True)
        table.add_column("Field", style="bold")
        table.add_column("Value")
        for k, v in data.items():
            if isinstance(v, (dict, list)):
                v = json.dumps(v, ensure_ascii=False, default=str)
            table.add_row(str(k), str(v))
        console.print(table)
    elif isinstance(data, list):
        if not data:
            console.print("[dim]No results[/dim]")
            return
        table = Table(show_header=True)
        if isinstance(data[0], dict):
            for col in data[0].keys():
                table.add_column(str(col))
            for row in data:
                table.add_row(*(str(row.get(c, "")) for c in data[0].keys()))
        else:
            table.add_column("Value")
            for item in data:
                table.add_row(str(item))
        console.print(table)
    else:
        console.print(str(data))


def emit_error(error: str, ctx: CLIContext, exit_code: int = 1) -> None:
    """Emit an error in the appropriate format."""
    if ctx.is_structured:
        print(json.dumps({"error": error, "exit_code": exit_code}))
    else:
        ctx.console.print(f"[red]Error:[/red] {error}")
