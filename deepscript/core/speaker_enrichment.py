"""Speaker enrichment — map diarization metadata back to transcript segments.

Fixes the gap where AudioScript resolves speakers but doesn't always write
labels onto individual segments. DeepScript reconstructs the mapping.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def enrich_speakers(transcript: dict[str, Any]) -> dict[str, Any]:
    """Enrich transcript segments with speaker labels from diarization metadata.

    Handles three cases:
    1. Segments already have 'speaker' field → no-op
    2. Diarization metadata exists → map speakers to segments by time alignment
    3. No diarization → leave as-is (segments default to "Unknown")

    Returns the transcript dict with enriched segments (mutated in-place).
    """
    segments = transcript.get("segments", [])
    if not segments:
        return transcript

    # Check if segments already have speaker labels
    has_speakers = any(s.get("speaker") for s in segments[:5])
    if has_speakers:
        return transcript

    # Try to enrich from diarization metadata
    diar = transcript.get("diarization", {})
    speakers_resolved = diar.get("speakers_resolved", [])
    if not speakers_resolved:
        return transcript

    # Build label → display name mapping
    label_map: dict[str, dict[str, Any]] = {}
    for sr in speakers_resolved:
        label = sr.get("local_label", "")
        label_map[label] = {
            "speaker": sr.get("display_name") or sr.get("speaker_cluster_id") or label,
            "speaker_cluster_id": sr.get("speaker_cluster_id", ""),
            "speaker_confidence": sr.get("confidence", 0.0),
        }

    # Check if diarization has segment-level timing (some pipelines include it)
    diar_segments = diar.get("segments", [])
    if diar_segments:
        _map_from_diarization_segments(segments, diar_segments, label_map)
    else:
        # No timing data — distribute speakers across segments heuristically
        _distribute_speakers_heuristic(segments, speakers_resolved, label_map)

    logger.info("Enriched %d segments with %d speakers", len(segments), len(label_map))
    return transcript


def _map_from_diarization_segments(
    segments: list[dict[str, Any]],
    diar_segments: list[dict[str, Any]],
    label_map: dict[str, dict[str, Any]],
) -> None:
    """Map speakers to transcript segments using diarization timeline."""
    for seg in segments:
        seg_mid = (seg.get("start", 0) + seg.get("end", 0)) / 2
        best_speaker = ""
        for ds in diar_segments:
            if ds.get("start", 0) <= seg_mid <= ds.get("end", 0):
                best_speaker = ds.get("speaker", ds.get("label", ""))
                break

        if best_speaker and best_speaker in label_map:
            seg.update(label_map[best_speaker])
        elif best_speaker:
            seg["speaker"] = best_speaker


def _distribute_speakers_heuristic(
    segments: list[dict[str, Any]],
    speakers_resolved: list[dict[str, Any]],
    label_map: dict[str, dict[str, Any]],
) -> None:
    """Distribute speakers across segments when no timing data exists.

    Heuristic: Assign speakers round-robin based on pause patterns.
    A gap >2s between segments suggests a speaker change.
    """
    if not speakers_resolved:
        return

    labels = [sr.get("local_label", "") for sr in speakers_resolved]
    if not labels:
        return

    current_idx = 0
    prev_end = 0.0

    for seg in segments:
        start = seg.get("start", 0)

        # Detect speaker change: gap >2s suggests different speaker
        if start - prev_end > 2.0 and len(labels) > 1:
            current_idx = (current_idx + 1) % len(labels)

        label = labels[current_idx]
        if label in label_map:
            seg.update(label_map[label])
        else:
            seg["speaker"] = label

        prev_end = seg.get("end", 0)


def load_speaker_db(db_path: str | Path) -> dict[str, Any] | None:
    """Load AudioScript's speaker identity database for cross-call context.

    Returns the speaker DB dict or None if not found.
    """
    import json

    path = Path(db_path)
    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            db = json.load(f)
        identities = db.get("identities", {})
        logger.info("Loaded speaker DB: %d identities", len(identities))
        return db
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load speaker DB %s: %s", db_path, e)
        return None


def get_speaker_context(
    transcript: dict[str, Any],
    speaker_db: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build speaker context for analysis enrichment.

    Combines per-segment speaker labels with cross-call identity data.

    Returns dict with:
        speakers: list of {cluster_id, display_name, status, total_calls, ...}
        is_returning: bool (any speaker seen in previous calls)
    """
    diar = transcript.get("diarization", {})
    speakers_resolved = diar.get("speakers_resolved", [])

    speakers: list[dict[str, Any]] = []
    is_returning = False

    for sr in speakers_resolved:
        cluster_id = sr.get("speaker_cluster_id", "")
        info: dict[str, Any] = {
            "cluster_id": cluster_id,
            "display_name": sr.get("display_name"),
            "status": sr.get("status", "unknown"),
            "confidence": sr.get("confidence", 0.0),
            "is_new": sr.get("is_new", True),
        }

        # Enrich from speaker DB if available
        if speaker_db and cluster_id:
            identity = speaker_db.get("identities", {}).get(cluster_id, {})
            if identity:
                info["canonical_name"] = identity.get("canonical_name")
                info["total_calls"] = identity.get("total_calls", 0)
                info["total_speaking_seconds"] = identity.get("total_speaking_seconds", 0)
                info["first_seen"] = identity.get("first_seen")
                info["last_seen"] = identity.get("last_seen")
                info["co_speakers"] = identity.get("typical_co_speakers", [])
                if identity.get("total_calls", 0) > 1:
                    is_returning = True

        speakers.append(info)

    return {
        "speakers": speakers,
        "is_returning": is_returning,
        "total_speakers": len(speakers),
    }
