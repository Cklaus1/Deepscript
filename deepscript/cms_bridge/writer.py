"""CMS episode writer — appends JSONL to BTask CMS store."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from deepscript.cms_bridge.episode import CallEpisode

logger = logging.getLogger(__name__)


def write_episode(episode: CallEpisode, store_path: str) -> Path:
    """Write a CallEpisode to the CMS store as JSONL.

    Episodes are appended to: {store_path}/episodes/coding/{task_type}.jsonl

    Returns the path to the JSONL file.
    """
    store = Path(store_path)
    episodes_dir = store / "episodes" / "coding"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize task_type for filename — prevent path traversal
    import re
    task_type = re.sub(r"[^a-zA-Z0-9_\-]", "_", episode.task_type)
    if not task_type:
        task_type = "unknown"
    jsonl_path = episodes_dir / f"{task_type}.jsonl"

    # Verify resolved path is under episodes_dir
    resolved = jsonl_path.resolve()
    expected = episodes_dir.resolve()
    if not str(resolved).startswith(str(expected)):
        raise ValueError(f"Path traversal blocked: {episode.task_type}")

    cms_dict = episode.to_cms_dict()
    line = json.dumps(cms_dict, ensure_ascii=False, default=str)

    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")

    logger.info("Episode %s written to %s", episode.episode_id, jsonl_path)
    return jsonl_path
