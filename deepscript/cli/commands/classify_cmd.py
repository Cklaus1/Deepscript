"""deepscript classify — Classify transcript type without full analysis."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import typer

from deepscript.cli.output import CLIContext, ExitCode, emit, emit_error
from deepscript.config.settings import get_settings
from deepscript.core.classifier import classify_transcript

logger = logging.getLogger(__name__)


def classify(
    file: str = typer.Argument(help="Path to transcript JSON file."),
    config_file: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to .deepscript.yaml config file.",
    ),
    ctx: typer.Context = typer.Option(None, hidden=True),
) -> None:
    """Classify a transcript and show the detected call type."""
    cli_ctx: CLIContext = ctx.obj if ctx and ctx.obj else CLIContext()

    try:
        config_path = Path(config_file) if config_file else None
        settings = get_settings(config_path=config_path)

        file_path = Path(file)
        if not file_path.exists():
            raise FileNotFoundError(f"Transcript not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            transcript = json.load(f)

        if "text" not in transcript and "segments" in transcript:
            transcript["text"] = " ".join(
                s.get("text", "") for s in transcript["segments"]
            )

        classification = classify_transcript(
            transcript, settings.custom_classifications
        )

        emit(
            {
                "source": str(file_path),
                "call_type": classification.call_type,
                "confidence": classification.confidence,
                "scores": classification.scores,
            },
            cli_ctx,
        )

    except (FileNotFoundError, ValueError) as e:
        emit_error(str(e), cli_ctx, ExitCode.VALIDATION_ERROR)
        raise typer.Exit(code=ExitCode.VALIDATION_ERROR)
    except Exception as e:
        logger.exception("Classification failed")
        emit_error(str(e), cli_ctx, ExitCode.classify(e))
        raise typer.Exit(code=ExitCode.classify(e))
