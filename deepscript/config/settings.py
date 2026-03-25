"""DeepScript configuration — Pydantic models + YAML loading."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, field_validator

logger = logging.getLogger(__name__)


class OutputConfig(BaseModel):
    format: str = "markdown"
    sections: str | list[str] = "all"


class CommunicationConfig(BaseModel):
    enabled: bool = True
    speaking_balance: bool = True
    engagement: bool = True


class BusinessConfig(BaseModel):
    summary: bool = True
    action_items: bool = True
    decisions: bool = True
    questions: bool = True


class SalesConfig(BaseModel):
    enabled: bool = True
    methodology: str = "meddic"
    competitors: list[str] = []
    close_coaching: bool = True
    phase_detection: bool = True

    @field_validator("methodology")
    @classmethod
    def validate_methodology(cls, v: str) -> str:
        allowed = {"meddic", "bant", "spin", "challenger"}
        if v not in allowed:
            raise ValueError(f"methodology must be one of {allowed}, got '{v}'")
        return v


class DiscoveryConfig(BaseModel):
    enabled: bool = True
    framework: str = "mom_test"
    pain_extraction: bool = True
    insight_mapping: bool = True


class TopicsConfig(BaseModel):
    enabled: bool = True
    method: str = "hybrid"
    min_duration: int = 60
    max_topics: int = 20
    index: bool = True


class LLMConfig(BaseModel):
    provider: str = "claude"  # LLM-first; falls back to rule-based if no API key
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 4096
    max_retries: int = 3  # Retry transient failures with exponential backoff
    concurrency: int = 5  # Max parallel LLM calls in async mode
    rate_limit_rpm: int = 0  # Requests per minute (0 = use provider default)
    base_url: str | None = None  # Custom API endpoint (for ollama, vllm, sglang)
    api_key: str | None = None  # Explicit API key (overrides env var)
    redact_names: bool = False
    budget_per_month: float = 50.0
    cost_tracking: bool = True

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        allowed = {"claude", "openai", "ollama", "vllm", "sglang", "nim", "none"}
        if v not in allowed:
            raise ValueError(f"llm.provider must be one of {allowed}, got '{v}'")
        return v


class CMSConfig(BaseModel):
    enabled: bool = False
    store_path: str = "/root/projects/BTask/packages/cms/store"


class CalendarConfig(BaseModel):
    enabled: bool = False
    provider: str = "ms365"  # ms365 | google | none
    enrich_classification: bool = True
    match_by: str = "time_proximity"
    time_window: int = 30  # minutes

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        allowed = {"ms365", "google", "none"}
        if v not in allowed:
            raise ValueError(f"calendar.provider must be one of {allowed}, got '{v}'")
        return v


class NotificationChannel(BaseModel):
    type: str = "teams"
    command: str = ""
    on: list[str] = []


class NotificationConfig(BaseModel):
    enabled: bool = False
    channels: list[NotificationChannel] = []


class RelationshipConfig(BaseModel):
    enabled: bool = False
    frameworks: list[str] = ["gottman", "nvc"]
    appreciation_ratio: bool = True
    bids_for_connection: bool = True
    growth_suggestions: bool = True
    export_to_minotes: bool = False
    consent_note: bool = True


class PMFConfig(BaseModel):
    enabled: bool = True
    ellis_threshold: float = 0.40
    min_calls_for_dashboard: int = 10
    segment_by: list[str] = ["company_size", "role", "use_case"]
    track_trend: bool = True


class DeepScriptConfig(BaseModel):
    """Main configuration model."""

    classify: bool = True
    custom_classifications: dict[str, Any] = {}
    output: OutputConfig = OutputConfig()
    communication: CommunicationConfig = CommunicationConfig()
    business: BusinessConfig = BusinessConfig()
    sales: SalesConfig = SalesConfig()
    discovery: DiscoveryConfig = DiscoveryConfig()
    topics: TopicsConfig = TopicsConfig()
    llm: LLMConfig = LLMConfig()
    cms: CMSConfig = CMSConfig()
    calendar: CalendarConfig = CalendarConfig()
    notifications: NotificationConfig = NotificationConfig()
    relationship: RelationshipConfig = RelationshipConfig()
    pmf: PMFConfig = PMFConfig()


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    """Load configuration from a YAML file."""
    if not config_path.exists():
        return {}
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config or {}
    except yaml.YAMLError as e:
        logger.warning("Failed to parse config file %s: %s", config_path, e)
        return {}
    except OSError as e:
        logger.warning("Failed to read config file %s: %s", config_path, e)
        return {}


def merge_configs(
    cli_args: dict[str, Any], file_config: dict[str, Any]
) -> dict[str, Any]:
    """Merge CLI arguments with file configuration. Non-None CLI values win."""
    merged = {**file_config}
    for key, value in cli_args.items():
        if value is not None:
            merged[key] = value
    return merged


def get_settings(
    cli_args: Optional[dict[str, Any]] = None,
    config_path: Optional[Path | str] = None,
) -> DeepScriptConfig:
    """Get settings by merging CLI args and config file."""
    explicit = config_path is not None
    if config_path is not None:
        config_path = Path(config_path)
    else:
        config_path = Path(".deepscript.yaml")

    if explicit and not config_path.exists():
        logger.warning("Config file not found: %s — using defaults", config_path)

    file_config = load_yaml_config(config_path)
    if cli_args:
        merged = merge_configs(cli_args, file_config)
    else:
        merged = file_config
    return DeepScriptConfig(**merged)
