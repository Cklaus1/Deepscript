"""Base analyzer protocol for DeepScript."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from deepscript.llm.provider import LLMProvider

logger = logging.getLogger(__name__)

# Maximum words to send to LLM (controls cost)
LLM_MAX_WORDS = 4000


@dataclass
class AnalysisResult:
    """Structured output from any analyzer."""

    call_type: str
    sections: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict, omitting sections with None values."""
        result = {"call_type": self.call_type}
        for key, value in self.sections.items():
            if value is not None:
                result[key] = value
        return result


class BaseAnalyzer(ABC):
    """Protocol for call-type-specific analyzers.

    Provides shared utilities for LLM calls and keyword matching
    to reduce repetition across analyzer implementations.
    """

    def __init__(self, llm: Optional["LLMProvider"] = None) -> None:
        self.llm = llm

    @property
    @abstractmethod
    def supported_types(self) -> list[str]:
        """Call types this analyzer handles."""
        ...

    @abstractmethod
    def analyze(self, transcript: dict[str, Any]) -> AnalysisResult:
        """Run analysis on a transcript and return structured sections."""
        ...

    # --- Shared utilities ---

    def llm_analyze(
        self, text: str, template: str, **kwargs: str
    ) -> dict[str, Any] | None:
        """Send text to LLM using a prompt template. Returns parsed JSON or None.

        Handles truncation, prompt rendering, JSON parsing, and error handling.
        """
        if not self.llm:
            return None
        truncated = self._truncate(text)
        try:
            prompt = self.llm.render_prompt(template, transcript=truncated, **kwargs)
            return self.llm.complete_json(prompt)
        except Exception as e:
            logger.warning("LLM analysis failed (%s): %s", template, e)
            return None

    @staticmethod
    def _truncate(text: str, max_words: int = LLM_MAX_WORDS) -> str:
        """Truncate text to max_words for LLM cost control."""
        words = text.split()
        if len(words) <= max_words:
            return text
        return " ".join(words[:max_words]) + "\n[...truncated...]"

    @staticmethod
    def score_keywords(
        text: str, keyword_groups: dict[str, list[str]]
    ) -> dict[str, list[str]]:
        """Score text against keyword groups. Returns {group: [matched_keywords]}."""
        text_lower = text.lower()
        return {
            name: [kw for kw in keywords if kw in text_lower]
            for name, keywords in keyword_groups.items()
        }

    @staticmethod
    def detect_keywords(text: str, keywords: list[str]) -> list[str]:
        """Return matching keywords from text."""
        text_lower = text.lower()
        return [kw for kw in keywords if kw in text_lower]

    # --- Combined analysis (1 LLM call instead of 3-5) ---

    def analyze_combined(
        self, transcript: dict[str, Any], call_type: str
    ) -> AnalysisResult | None:
        """Single-LLM-call analysis using combined prompt.

        Subclasses override `_combined_prompt_extras()` to add type-specific sections.
        Returns None if LLM unavailable or fails (caller falls back to multi-call).
        """
        if not self.llm:
            return None

        text = self._truncate(transcript.get("text", ""))
        instructions, schema = self._combined_prompt_extras(call_type)

        try:
            prompt = self.llm.render_prompt(
                "combined_analysis",
                call_type=call_type,
                type_specific_instructions=instructions,
                type_specific_schema=schema,
                transcript=text,
            )
            result = self.llm.complete_json(prompt)
            if result and isinstance(result, dict):
                return AnalysisResult(call_type=call_type, sections=result)
        except Exception as e:
            logger.warning("Combined analysis failed, falling back to multi-call: %s", e)

        return None

    def _combined_prompt_extras(self, call_type: str) -> tuple[str, str]:
        """Return (type_specific_instructions, type_specific_schema) for combined prompt.

        Override in subclasses to add type-specific analysis sections.
        Default returns empty strings (business meeting analysis only).
        """
        return ("", "")
