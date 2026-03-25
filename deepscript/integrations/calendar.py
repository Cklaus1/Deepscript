"""Calendar context enrichment via ms365-cli or gwscli."""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from deepscript.config.settings import CalendarConfig

logger = logging.getLogger(__name__)


@dataclass
class CalendarContext:
    """Meeting context from calendar lookup."""

    subject: str = ""
    organizer: str = ""
    attendees: list[str] = None
    recurring: bool = False
    series_name: str = ""
    location: str = ""
    body_preview: str = ""

    def __post_init__(self) -> None:
        if self.attendees is None:
            self.attendees = []

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "organizer": self.organizer,
            "attendees": self.attendees,
            "recurring": self.recurring,
            "series_name": self.series_name,
            "location": self.location,
            "body_preview": self.body_preview,
        }


def get_calendar_context(
    recording_time: str | datetime | None,
    duration_seconds: float = 0,
    config: CalendarConfig | None = None,
) -> CalendarContext | None:
    """Look up calendar event matching the recording time.

    Args:
        recording_time: ISO 8601 timestamp or datetime of recording.
        duration_seconds: Duration of the recording.
        config: Calendar configuration.

    Returns:
        CalendarContext if a matching event is found, None otherwise.
    """
    if config is None or config.provider == "none" or not config.enabled:
        return None

    if recording_time is None:
        return None

    # Parse recording time
    if isinstance(recording_time, str):
        try:
            rec_time = datetime.fromisoformat(recording_time.replace("Z", "+00:00"))
        except ValueError:
            logger.warning("Invalid recording time: %s", recording_time)
            return None
    else:
        rec_time = recording_time

    window = timedelta(minutes=config.time_window)
    start = rec_time - window
    end = rec_time + timedelta(seconds=duration_seconds) + window

    if config.provider == "ms365":
        return _lookup_ms365(start, end)
    elif config.provider == "google":
        return _lookup_google(start, end)

    return None


def _lookup_ms365(start: datetime, end: datetime) -> CalendarContext | None:
    """Look up calendar event via ms365-cli."""
    start_str = start.strftime("%Y-%m-%dT%H:%M:%S")
    end_str = end.strftime("%Y-%m-%dT%H:%M:%S")

    try:
        result = subprocess.run(
            ["ms365", "calendar", "view", "--start", start_str, "--end", end_str, "-o", "json"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            logger.warning("ms365 calendar lookup failed: %s", result.stderr[:200])
            return None

        events = json.loads(result.stdout)
        if not events:
            return None

        # Take the first/closest event
        event = events[0] if isinstance(events, list) else events
        return _parse_ms365_event(event)

    except FileNotFoundError:
        logger.warning("ms365 CLI not found — install ms365-cli for calendar enrichment")
        return None
    except subprocess.TimeoutExpired:
        logger.warning("ms365 calendar lookup timed out")
        return None
    except (json.JSONDecodeError, Exception) as e:
        logger.warning("Calendar lookup error: %s", e)
        return None


def _parse_ms365_event(event: dict[str, Any]) -> CalendarContext:
    """Parse ms365 calendar event JSON."""
    attendees = []
    for att in event.get("attendees", []):
        email = att.get("emailAddress", {})
        name = email.get("name") or email.get("address", "")
        if name:
            attendees.append(name)

    organizer = ""
    org = event.get("organizer", {}).get("emailAddress", {})
    organizer = org.get("name") or org.get("address", "")

    return CalendarContext(
        subject=event.get("subject", ""),
        organizer=organizer,
        attendees=attendees,
        recurring=event.get("recurrence") is not None,
        series_name=event.get("seriesMasterId", ""),
        location=event.get("location", {}).get("displayName", ""),
        body_preview=event.get("bodyPreview", "")[:200],
    )


def _lookup_google(start: datetime, end: datetime) -> CalendarContext | None:
    """Look up calendar event via gwscli."""
    start_str = start.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_str = end.strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        result = subprocess.run(
            [
                "gws", "calendar", "events", "list",
                "--params", json.dumps({
                    "timeMin": start_str,
                    "timeMax": end_str,
                    "singleEvents": True,
                    "maxResults": 5,
                }),
                "--format", "json",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            logger.warning("gws calendar lookup failed: %s", result.stderr[:200])
            return None

        data = json.loads(result.stdout)
        items = data.get("items", [])
        if not items:
            return None

        event = items[0]
        return _parse_google_event(event)

    except FileNotFoundError:
        logger.warning("gws CLI not found — install gwscli for calendar enrichment")
        return None
    except subprocess.TimeoutExpired:
        logger.warning("gws calendar lookup timed out")
        return None
    except (json.JSONDecodeError, Exception) as e:
        logger.warning("Calendar lookup error: %s", e)
        return None


def _parse_google_event(event: dict[str, Any]) -> CalendarContext:
    """Parse Google Calendar event JSON."""
    attendees = []
    for att in event.get("attendees", []):
        name = att.get("displayName") or att.get("email", "")
        if name:
            attendees.append(name)

    organizer = (
        event.get("organizer", {}).get("displayName")
        or event.get("organizer", {}).get("email", "")
    )

    return CalendarContext(
        subject=event.get("summary", ""),
        organizer=organizer,
        attendees=attendees,
        recurring=event.get("recurringEventId") is not None,
        location=event.get("location", ""),
        body_preview=event.get("description", "")[:200],
    )
