"""Tests for LLM provider — uses mocked Anthropic client."""

from unittest.mock import MagicMock, patch

from deepscript.config.settings import LLMConfig
from deepscript.llm.provider import LLMProvider
from deepscript.llm.cost_tracker import CostTracker


def test_create_returns_none_for_no_provider():
    config = LLMConfig(provider="none")
    assert LLMProvider.create(config) is None


def test_create_returns_none_without_api_key():
    config = LLMConfig(provider="claude")
    with patch.dict("os.environ", {}, clear=True):
        result = LLMProvider.create(config)
    assert result is None


def test_create_returns_provider_with_api_key():
    config = LLMConfig(provider="claude")
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        result = LLMProvider.create(config)
    assert result is not None
    assert isinstance(result, LLMProvider)


def test_complete_calls_anthropic():
    config = LLMConfig(provider="claude", model="claude-sonnet-4-6")
    provider = LLMProvider(config)

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Hello world")]
    mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response
    provider._client = mock_client

    result = provider.complete("test prompt")
    assert result == "Hello world"
    mock_client.messages.create.assert_called_once()


def test_complete_json_parses_response():
    config = LLMConfig(provider="claude")
    provider = LLMProvider(config)

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text='{"key": "value"}')]
    mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response
    provider._client = mock_client

    result = provider.complete_json("test prompt")
    assert result == {"key": "value"}


def test_complete_json_handles_markdown_wrapped():
    config = LLMConfig(provider="claude")
    provider = LLMProvider(config)

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text='```json\n{"key": "value"}\n```')]
    mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response
    provider._client = mock_client

    result = provider.complete_json("test prompt")
    assert result == {"key": "value"}


def test_complete_returns_none_on_error():
    config = LLMConfig(provider="claude")
    provider = LLMProvider(config)

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = Exception("API error")
    provider._client = mock_client

    result = provider.complete("test prompt")
    assert result is None


def test_cost_tracker_records_usage():
    tracker = CostTracker(budget_limit=10.0)
    tracker.record("claude-sonnet-4-6", input_tokens=1000, output_tokens=500)

    assert tracker.total_input_tokens == 1000
    assert tracker.total_output_tokens == 500
    assert tracker.total_cost_usd > 0
    assert len(tracker.entries) == 1


def test_cost_tracker_summary():
    tracker = CostTracker(budget_limit=10.0)
    tracker.record("claude-sonnet-4-6", input_tokens=1000, output_tokens=500)

    summary = tracker.summary()
    assert summary["calls"] == 1
    assert summary["total_input_tokens"] == 1000


def test_render_prompt():
    config = LLMConfig(provider="claude")
    provider = LLMProvider(config)

    prompt = provider.render_prompt("classify", transcript="Hello world")
    assert "Hello world" in prompt
    assert "call_type" in prompt


# --- Local provider tests ---


def test_create_ollama_no_key_needed():
    config = LLMConfig(provider="ollama", model="llama3")
    result = LLMProvider.create(config)
    assert result is not None
    assert result.config.provider == "ollama"


def test_create_vllm_no_key_needed():
    config = LLMConfig(provider="vllm", model="meta-llama/Llama-3-8B")
    result = LLMProvider.create(config)
    assert result is not None


def test_create_sglang_no_key_needed():
    config = LLMConfig(provider="sglang", model="meta-llama/Llama-3-8B")
    result = LLMProvider.create(config)
    assert result is not None


def test_create_nim_needs_key():
    config = LLMConfig(provider="nim", model="meta/llama-3.1-8b-instruct")
    with patch.dict("os.environ", {}, clear=True):
        result = LLMProvider.create(config)
    assert result is None


def test_create_nim_with_key():
    config = LLMConfig(provider="nim", model="meta/llama-3.1-8b-instruct")
    with patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"}):
        result = LLMProvider.create(config)
    assert result is not None


def test_ollama_uses_openai_compat():
    config = LLMConfig(provider="ollama", model="llama3")
    provider = LLMProvider(config)

    # Mock OpenAI-compatible response
    mock_choice = MagicMock()
    mock_choice.message.content = "test response"
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    provider._client = mock_client

    result = provider.complete("test prompt")
    assert result == "test response"
    mock_client.chat.completions.create.assert_called_once()


def test_custom_base_url():
    config = LLMConfig(provider="vllm", model="custom-model", base_url="http://gpu-server:8000/v1")
    provider = LLMProvider.create(config)
    assert provider is not None
    assert provider.config.base_url == "http://gpu-server:8000/v1"


def test_openai_with_custom_base_url():
    """OpenAI provider can use custom base_url for proxies or Azure."""
    config = LLMConfig(provider="openai", model="gpt-4", base_url="https://my-proxy.com/v1", api_key="test")
    provider = LLMProvider.create(config)
    assert provider is not None


def test_nim_prefers_nvidia_key_over_openai_key():
    """Regression: NIM must use NVIDIA_API_KEY even when OPENAI_API_KEY is set.

    A shared .env sourced in cron put OPENAI_API_KEY in the environment, which
    was sent to NIM's endpoint and 401'd. The provider-specific key must win.
    """
    config = LLMConfig(provider="nim", model="meta/llama-3.1-8b-instruct")
    provider = LLMProvider(config)
    with patch.dict("os.environ", {"NVIDIA_API_KEY": "nvidia-key", "OPENAI_API_KEY": "openai-key"}):
        with patch("openai.OpenAI") as mock_openai:
            provider._build_client(async_mode=False)
    assert mock_openai.call_args.kwargs["api_key"] == "nvidia-key"


def test_failure_counter_increments_on_llm_error():
    """A failed LLM call (exhausted retries) bumps the thread's failure count."""
    config = LLMConfig(provider="claude", max_retries=1)
    provider = LLMProvider(config)
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = Exception("API error")
    provider._client = mock_client

    provider.reset_failures()
    assert provider.failure_count == 0
    assert provider.complete("test") is None
    assert provider.failure_count == 1
    # A second failure accumulates
    assert provider.complete("test") is None
    assert provider.failure_count == 2


def test_failure_counter_not_incremented_on_success():
    """A successful LLM call leaves the failure count at zero."""
    config = LLMConfig(provider="claude", model="claude-sonnet-4-6")
    provider = LLMProvider(config)
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="ok")]
    mock_response.usage = MagicMock(input_tokens=1, output_tokens=1)
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response
    provider._client = mock_client

    provider.reset_failures()
    assert provider.complete("test") == "ok"
    assert provider.failure_count == 0
