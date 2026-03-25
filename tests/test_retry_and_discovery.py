"""Tests for retry logic, async support, and auto-discovery registry."""

from unittest.mock import MagicMock, patch
import time

from deepscript.analyzers import build_analyzer_registry, collect_keywords, discover_analyzer_classes
from deepscript.analyzers.base import BaseAnalyzer
from deepscript.config.settings import DeepScriptConfig, LLMConfig
from deepscript.llm.provider import LLMProvider, _is_transient


# --- Retry logic ---


def test_is_transient_connection_error():
    assert _is_transient(ConnectionError("refused")) is True


def test_is_transient_timeout():
    assert _is_transient(TimeoutError("timed out")) is True


def test_is_transient_os_error():
    assert _is_transient(OSError("network unreachable")) is True


def test_is_transient_rate_limit():
    exc = type("RateLimitError", (Exception,), {})()
    assert _is_transient(exc) is True


def test_is_not_transient_auth_error():
    exc = type("AuthenticationError", (Exception,), {})()
    assert _is_transient(exc) is False


def test_is_not_transient_bad_request():
    exc = type("BadRequestError", (Exception,), {})()
    assert _is_transient(exc) is False


def test_is_transient_http_429():
    exc = Exception("rate limited")
    exc.status_code = 429
    assert _is_transient(exc) is True


def test_is_transient_http_500():
    exc = Exception("server error")
    exc.status_code = 500
    assert _is_transient(exc) is True


def test_retry_succeeds_on_second_attempt():
    config = LLMConfig(provider="claude", model="test", max_retries=3)
    provider = LLMProvider(config)

    call_count = 0

    def mock_create(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ConnectionError("first attempt fails")
        resp = MagicMock()
        resp.content = [MagicMock(text="success")]
        resp.usage = MagicMock(input_tokens=10, output_tokens=5)
        return resp

    mock_client = MagicMock()
    mock_client.messages.create = mock_create
    provider._client = mock_client

    with patch("time.sleep"):  # Don't actually sleep in tests
        result = provider.complete("test")

    assert result == "success"
    assert call_count == 2


def test_retry_gives_up_after_max():
    config = LLMConfig(provider="claude", model="test", max_retries=2)
    provider = LLMProvider(config)

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = ConnectionError("always fails")
    provider._client = mock_client

    with patch("time.sleep"):
        result = provider.complete("test")

    assert result is None
    assert mock_client.messages.create.call_count == 2


def test_no_retry_on_permanent_error():
    config = LLMConfig(provider="claude", model="test", max_retries=3)
    provider = LLMProvider(config)

    exc = type("AuthenticationError", (Exception,), {})("bad key")
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = exc
    provider._client = mock_client

    result = provider.complete("test")
    assert result is None
    # Should NOT retry — only 1 call
    assert mock_client.messages.create.call_count == 1


# --- Auto-discovery ---


def test_discover_analyzer_classes():
    # Reset cache
    import deepscript.analyzers as pkg
    pkg._discovered = None

    classes = discover_analyzer_classes()
    assert len(classes) > 10  # Should find many call types
    assert "business-meeting" in classes
    assert "sales-call" in classes
    assert "discovery-call" in classes
    assert "pmf-call" in classes
    assert "interview-behavioral" in classes
    assert "support-escalation" in classes
    assert "qbr" in classes
    assert "family" in classes
    assert "podcast" in classes


def test_collect_keywords():
    import deepscript.analyzers as pkg
    pkg._discovered = None

    keywords = collect_keywords()
    # Should find keywords from analyzer classes that define classification_keywords
    assert isinstance(keywords, dict)
    # BusinessAnalyzer and SalesAnalyzer define classification_keywords
    if "business-meeting" in keywords:
        assert "agenda" in keywords["business-meeting"]
    if "sales-call" in keywords:
        assert "pricing" in keywords["sales-call"]


def test_build_analyzer_registry():
    import deepscript.analyzers as pkg
    pkg._discovered = None

    registry = build_analyzer_registry(llm=None)
    assert "business-meeting" in registry
    assert "unknown" in registry
    assert "sales-call" in registry
    assert "pmf-call" in registry

    # All values should be BaseAnalyzer instances
    for ct, analyzer in registry.items():
        assert isinstance(analyzer, BaseAnalyzer), f"{ct} is not BaseAnalyzer"


def test_build_analyzer_registry_with_settings():
    import deepscript.analyzers as pkg
    pkg._discovered = None

    settings = DeepScriptConfig()
    registry = build_analyzer_registry(llm=None, settings=settings)
    assert len(registry) > 20  # Should cover all call types


def test_new_analyzer_auto_discovered():
    """Verify that adding supported_types to a new class would auto-discover it."""
    import deepscript.analyzers as pkg
    pkg._discovered = None

    classes = discover_analyzer_classes()
    # voice-memo should be discovered from SimpleAnalyzer
    assert "voice-memo" in classes
