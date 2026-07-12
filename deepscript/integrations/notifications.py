"""Post-analysis notification routing via ms365-cli or gwscli."""

from __future__ import annotations

import logging
import re
import shlex
import subprocess
from typing import Any

from deepscript.config.settings import NotificationConfig

logger = logging.getLogger(__name__)


def send_notifications(
    config: NotificationConfig,
    call_type: str,
    summary: str,
    title: str = "",
    user: str = "",
) -> list[dict[str, Any]]:
    """Send post-analysis notifications via configured channels.

    Args:
        config: Notification configuration.
        call_type: The classified call type.
        summary: Short summary text to send.
        title: Title for the notification.
        user: User email for email notifications.

    Returns:
        List of send results (channel type, success, error).
    """
    if not config.enabled or not config.channels:
        return []

    results: list[dict[str, Any]] = []

    for channel in config.channels:
        # Check if this channel is configured for this call type
        if channel.on and call_type not in channel.on:
            continue

        if not channel.command:
            continue

        # Substitute placeholders with shell-escaped values (single-pass)
        command = channel.command
        values = {
            "summary": shlex.quote(summary[:500]),
            "title": shlex.quote(title),
            "user": shlex.quote(user),
            "call_type": shlex.quote(call_type),
        }
        command = re.sub(
            r"\{(summary|title|user|call_type)\}",
            lambda m: values[m.group(1)],
            command,
        )

        result = _execute_notification(channel.type, command)
        results.append(result)

    return results


def _execute_notification(channel_type: str, command: str) -> dict[str, Any]:
    """Execute a notification command."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            logger.info("Notification sent via %s", channel_type)
            return {"channel": channel_type, "success": True}
        else:
            logger.warning(
                "Notification via %s failed: %s",
                channel_type,
                result.stderr[:200],
            )
            return {
                "channel": channel_type,
                "success": False,
                "error": result.stderr[:200],
            }
    except FileNotFoundError:
        msg = f"CLI tool for {channel_type} not found"
        logger.warning(msg)
        return {"channel": channel_type, "success": False, "error": msg}
    except subprocess.TimeoutExpired:
        msg = f"Notification via {channel_type} timed out"
        logger.warning(msg)
        return {"channel": channel_type, "success": False, "error": msg}
