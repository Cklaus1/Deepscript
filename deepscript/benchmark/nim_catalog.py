"""NVIDIA NIM model catalog — fetch, filter, and categorize models."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"

# Model ID patterns to EXCLUDE (not suitable for transcript analysis)
EXCLUDE_PATTERNS = [
    "embed", "nv-embed", "arctic-embed", "embedqa", "embedcode",
    "guard", "nemoguard", "safety", "content-safety",
    "reward", "rerank",
    "nvclip", "streampetr", "deplot", "paligemma", "kosmos-2", "neva",
    "nemoretriever-parse", "nemotron-parse",
    "gliner-pii",
    "audio", "whisper", "tts", "speech",
    "sdxl", "stable-diffusion", "imagen", "flux",
    "video", "cosmos",
    "steer",
]

# Known high-quality models for text analysis (manually curated)
TIER_HINTS: dict[str, int] = {
    "meta/llama-3.1-405b-instruct": 1,
    "nvidia/llama-3.1-nemotron-ultra-253b-v1": 1,
    "deepseek-ai/deepseek-r1": 1,
    "mistralai/mistral-large-3-675b-instruct-2512": 1,
    "qwen/qwen3.5-397b-a17b": 1,
    "meta/llama-3.3-70b-instruct": 2,
    "meta/llama-3.1-70b-instruct": 2,
    "nvidia/llama-3.1-nemotron-70b-instruct": 2,
    "nvidia/llama-3.3-nemotron-super-49b-v1.5": 2,
    "mistralai/mixtral-8x22b-instruct-v0.1": 2,
    "meta/llama-3.1-8b-instruct": 3,
    "meta/llama-3.2-3b-instruct": 3,
    "mistralai/mistral-7b-instruct-v0.3": 3,
    "microsoft/phi-4": 3,
}


@dataclass
class NIMModel:
    """A model from the NIM catalog."""

    id: str
    owned_by: str
    created: int = 0
    suitable_for_analysis: bool = True
    tier_hint: int = 0  # 0=unknown, 1=top, 2=strong, 3=fast/small


def fetch_nim_models() -> list[NIMModel]:
    """Fetch all models from NVIDIA NIM API (no auth required)."""
    try:
        import urllib.request

        req = urllib.request.Request(f"{NIM_BASE_URL}/models")
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())

        models = []
        for m in data.get("data", []):
            model_id = m.get("id", "")
            owned_by = m.get("owned_by", "")
            created = m.get("created", 0)

            # Check if suitable for transcript analysis
            suitable = not any(pat in model_id.lower() for pat in EXCLUDE_PATTERNS)

            tier = TIER_HINTS.get(model_id, 0)

            models.append(NIMModel(
                id=model_id,
                owned_by=owned_by,
                created=created,
                suitable_for_analysis=suitable,
                tier_hint=tier,
            ))

        return models

    except Exception as e:
        logger.error("Failed to fetch NIM models: %s", e)
        return []


def filter_chat_models(models: list[NIMModel]) -> list[NIMModel]:
    """Filter to models suitable for transcript analysis."""
    return [m for m in models if m.suitable_for_analysis]


def categorize_models(models: list[NIMModel]) -> dict[str, list[NIMModel]]:
    """Categorize models by owner for organized display."""
    by_owner: dict[str, list[NIMModel]] = {}
    for m in models:
        owner = m.owned_by or "unknown"
        by_owner.setdefault(owner, []).append(m)
    return dict(sorted(by_owner.items()))
