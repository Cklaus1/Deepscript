"""Logging configuration for DeepScript."""

import logging


def setup_logging(level: int = logging.WARNING) -> None:
    """Configure structured logging."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
