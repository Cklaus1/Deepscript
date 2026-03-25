"""Processing manifest — tracks analyzed files to support batch --new-only mode."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = ".deepscript-manifest.json"


@dataclass
class ManifestEntry:
    file_path: str
    file_hash: str
    status: str  # "completed" | "failed" | "skipped"
    analyzed_at: str = ""
    call_type: str = ""
    file_size: int = 0    # Cached for fast change detection
    file_mtime: float = 0.0  # Cached mtime for fast change detection


@dataclass
class ProcessingManifest:
    """Tracks which files have been analyzed."""

    entries: dict[str, ManifestEntry] = field(default_factory=dict)
    version: str = "1.1"

    def is_processed(self, file_path: Path) -> bool:
        """Check if a file has already been successfully processed.

        Uses mtime+size as a fast check; only hashes if metadata changed.
        """
        entry = self.entries.get(str(file_path))
        if not entry or entry.status != "completed":
            return False

        try:
            stat = file_path.stat()
        except OSError:
            return False

        # Fast path: mtime and size unchanged → skip hashing
        if entry.file_mtime == stat.st_mtime and entry.file_size == stat.st_size:
            return True

        # Slow path: metadata changed, verify hash
        file_hash = get_file_hash(file_path)
        return entry.file_hash == file_hash

    def record(
        self, file_path: Path, status: str, call_type: str = ""
    ) -> None:
        """Record a processing result."""
        try:
            stat = file_path.stat()
            size = stat.st_size
            mtime = stat.st_mtime
        except OSError:
            size = 0
            mtime = 0.0

        self.entries[str(file_path)] = ManifestEntry(
            file_path=str(file_path),
            file_hash=get_file_hash(file_path),
            status=status,
            analyzed_at=datetime.now(timezone.utc).isoformat(),
            call_type=call_type,
            file_size=size,
            file_mtime=mtime,
        )

    def save(self, manifest_path: Path) -> None:
        """Save manifest to disk atomically."""
        data = {
            "version": self.version,
            "entries": {k: asdict(v) for k, v in self.entries.items()},
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = manifest_path.with_suffix(f".tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            tmp.replace(manifest_path)
        finally:
            if tmp.exists():
                tmp.unlink()

    @classmethod
    def load(cls, manifest_path: Path) -> "ProcessingManifest":
        """Load manifest from disk. Tolerant of schema changes."""
        if not manifest_path.exists():
            return cls()
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            entries: dict[str, ManifestEntry] = {}
            for k, v in data.get("entries", {}).items():
                try:
                    entries[k] = ManifestEntry(**v)
                except (TypeError, ValueError) as e:
                    # Handle schema evolution — old entries may lack new fields
                    logger.debug("Skipped manifest entry %s: %s", k, e)
                    # Try with just required fields
                    try:
                        entries[k] = ManifestEntry(
                            file_path=v.get("file_path", k),
                            file_hash=v.get("file_hash", ""),
                            status=v.get("status", "unknown"),
                            analyzed_at=v.get("analyzed_at", ""),
                            call_type=v.get("call_type", ""),
                        )
                    except Exception:
                        pass

            return cls(entries=entries, version=data.get("version", "1.0"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load manifest %s: %s", manifest_path, e)
            return cls()


def get_file_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]
