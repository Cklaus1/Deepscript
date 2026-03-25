"""Tests for calendar context integration."""

from unittest.mock import patch, MagicMock
import json

from deepscript.config.settings import CalendarConfig
from deepscript.integrations.calendar import get_calendar_context, CalendarContext


def test_calendar_disabled():
    config = CalendarConfig(enabled=False)
    result = get_calendar_context("2026-03-25T10:00:00Z", config=config)
    assert result is None


def test_calendar_none_provider():
    config = CalendarConfig(enabled=True, provider="none")
    result = get_calendar_context("2026-03-25T10:00:00Z", config=config)
    assert result is None


def test_calendar_no_recording_time():
    config = CalendarConfig(enabled=True, provider="ms365")
    result = get_calendar_context(None, config=config)
    assert result is None


def test_ms365_calendar_lookup():
    config = CalendarConfig(enabled=True, provider="ms365")
    mock_event = [{
        "subject": "Weekly Standup",
        "organizer": {"emailAddress": {"name": "Alice", "address": "alice@test.com"}},
        "attendees": [
            {"emailAddress": {"name": "Bob", "address": "bob@test.com"}},
        ],
        "recurrence": {"pattern": "weekly"},
        "location": {"displayName": "Teams Meeting"},
        "bodyPreview": "Discuss sprint progress",
    }]

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = json.dumps(mock_event)

    with patch("deepscript.integrations.calendar.subprocess.run", return_value=mock_result):
        result = get_calendar_context("2026-03-25T10:00:00Z", config=config)

    assert result is not None
    assert result.subject == "Weekly Standup"
    assert result.organizer == "Alice"
    assert "Bob" in result.attendees
    assert result.recurring is True


def test_ms365_cli_not_found():
    config = CalendarConfig(enabled=True, provider="ms365")

    with patch("deepscript.integrations.calendar.subprocess.run", side_effect=FileNotFoundError):
        result = get_calendar_context("2026-03-25T10:00:00Z", config=config)

    assert result is None


def test_calendar_context_to_dict():
    ctx = CalendarContext(
        subject="Test Meeting",
        organizer="Alice",
        attendees=["Bob", "Carol"],
        recurring=False,
    )
    d = ctx.to_dict()
    assert d["subject"] == "Test Meeting"
    assert len(d["attendees"]) == 2
