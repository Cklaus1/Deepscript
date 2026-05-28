"""Post-transcription name correction dictionary.

Whisper frequently mangles proper nouns — especially non-English names.
This module applies corrections to transcript text and speaker name evidence
before identification runs.

Corrections can come from:
1. A YAML/text file in the transcript directory (name_corrections.txt)
2. Speaker DB entries with confirmed names
3. Contact list names
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Common Whisper mistranscriptions → correct names
# Format: (pattern, replacement, is_regex)
_BUILTIN_CORRECTIONS: list[tuple[str, str, bool]] = []


def load_corrections(
    transcript_dir: str | Path | None = None,
    speaker_db_path: str | Path | None = None,
    contacts: list[dict] | None = None,
) -> dict[str, str]:
    """Build a name correction dictionary from all available sources.

    Returns: {mangled_text_lower: correct_name}
    """
    corrections: dict[str, str] = {}

    # 1. Load from name_corrections.txt in transcript directory
    if transcript_dir:
        corrections.update(_load_corrections_file(Path(transcript_dir)))

    # 2. Build corrections from speaker DB — names that Whisper might mangle
    if speaker_db_path:
        corrections.update(_build_db_corrections(Path(speaker_db_path)))

    # 3. Build corrections from contacts — common mistranscriptions
    if contacts:
        corrections.update(_build_contact_corrections(contacts))

    if corrections:
        logger.info("Name corrections loaded: %d entries", len(corrections))
    return corrections


def _load_corrections_file(transcript_dir: Path) -> dict[str, str]:
    """Load name_corrections.txt — one correction per line.

    Format: mangled text -> correct name
    Example:
        a news grover -> Anuj Grover
        a news -> Anuj
        anews -> Anuj
    """
    corrections: dict[str, str] = {}

    for name in ["name_corrections.txt", ".name_corrections"]:
        path = transcript_dir / name
        if path.exists():
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "->" in line:
                        mangled, correct = line.split("->", 1)
                        corrections[mangled.strip().lower()] = correct.strip()
                    elif "=>" in line:
                        mangled, correct = line.split("=>", 1)
                        corrections[mangled.strip().lower()] = correct.strip()
            logger.info("Loaded %d corrections from %s", len(corrections), path)
            break

    return corrections


def _build_db_corrections(speaker_db_path: Path) -> dict[str, str]:
    """Build corrections from speaker DB names.

    For confirmed names, generate common Whisper mistranscription patterns.
    """
    import json

    if not speaker_db_path.exists():
        return {}

    corrections: dict[str, str] = {}
    try:
        with open(speaker_db_path) as f:
            db = json.load(f)

        for cid, data in db.get("identities", {}).items():
            # Only use high-confidence names
            conf = data.get("deepscript_confidence", 0)
            status = data.get("deepscript_status", "")
            if conf < 0.85 and status not in ("user_confirmed", "deepscript_confident"):
                continue

            name = data.get("canonical_name", "")
            if not name or len(name) < 3:
                continue

            # Generate common Whisper mistranscription patterns for this name
            for variant in _generate_whisper_variants(name):
                if variant.lower() != name.lower():
                    corrections[variant.lower()] = name

    except Exception as e:
        logger.debug("Failed to build DB corrections: %s", e)

    return corrections


def _generate_whisper_variants(name: str) -> list[str]:
    """Generate common ways Whisper might mistranscribe a name.

    E.g., "Anuj" → ["a nooj", "a news", "anews", "a new"]
    """
    variants = []
    first = name.split()[0].lower()

    # Space insertion: "Anuj" → "a nuj", "a new j"
    if len(first) >= 4:
        variants.append(f"{first[0]} {first[1:]}")
        variants.append(f"{first[:2]} {first[2:]}")

    # Common phonetic confusions
    phonetic_map = {
        "uj": ["ews", "ooj", "uge", "ooge"],
        "aj": ["age", "aje"],
        "sh": ["ch", "s"],
        "th": ["d", "t"],
        "v": ["b", "w"],
        "z": ["s"],
    }
    for orig, replacements in phonetic_map.items():
        if orig in first:
            for rep in replacements:
                variants.append(first.replace(orig, rep))

    return variants


def _build_contact_corrections(contacts: list[dict]) -> dict[str, str]:
    """Build corrections from contact names — only non-English names likely to be mangled.

    Conservative: only generates corrections for names with phonetic patterns
    that Whisper commonly mangles, and only for the first name (not full name variants).
    """
    corrections: dict[str, str] = {}

    for c in contacts:
        name = c.get("displayName", "")
        if not name or len(name) < 4:
            continue

        first = name.split()[0]
        if len(first) < 4:
            continue

        # Only generate corrections for names that Whisper is likely to mangle
        if _is_likely_mangled(first):
            for variant in _generate_whisper_variants(first):
                if variant.lower() != first.lower() and len(variant) >= 3:
                    corrections[variant.lower()] = first

    return corrections


def _is_likely_mangled(name: str) -> bool:
    """Check if a name has phonetic patterns Whisper commonly mangles."""
    name_lower = name.lower()
    # Non-English phonetic patterns
    patterns = ["uj", "aj", "bh", "dh", "gh", "jh", "kh", "ph", "sh", "th",
                "aa", "ee", "oo", "ii", "uu", "nj", "nk", "ng"]
    return any(p in name_lower for p in patterns)


def compile_corrections(corrections: dict[str, str]) -> tuple[re.Pattern | None, dict[str, str]]:
    """Compile all corrections into a single regex for fast matching.

    Returns (compiled_pattern, mangled_to_correct_map) or (None, {}).
    """
    if not corrections:
        return None, {}

    # Sort by length descending so longer matches take priority
    sorted_items = sorted(corrections.items(), key=lambda x: -len(x[0]))
    # Build alternation pattern
    escaped = [re.escape(k) for k, _ in sorted_items]
    pattern = re.compile("|".join(escaped), re.IGNORECASE)
    return pattern, corrections


def apply_corrections(
    text: str,
    corrections: dict[str, str],
    compiled: tuple[re.Pattern | None, dict[str, str]] | None = None,
) -> str:
    """Apply name corrections to transcript text.

    Uses pre-compiled regex if provided, otherwise compiles on the fly.
    """
    if not corrections or not text:
        return text

    if compiled and compiled[0]:
        pattern, lookup = compiled
    else:
        pattern, lookup = compile_corrections(corrections)

    if not pattern:
        return text

    def _replace(m: re.Match) -> str:
        return lookup.get(m.group(0).lower(), m.group(0))

    return pattern.sub(_replace, text)


def apply_corrections_to_segments(
    segments: list[dict],
    corrections: dict[str, str],
) -> int:
    """Apply name corrections to all segment text in-place.

    Returns number of corrections made.
    """
    if not corrections:
        return 0

    # Pre-compile once for all segments
    compiled = compile_corrections(corrections)
    if not compiled[0]:
        return 0

    count = 0
    for seg in segments:
        text = seg.get("text", "")
        if text:
            corrected = apply_corrections(text, corrections, compiled=compiled)
            if corrected != text:
                seg["text"] = corrected
                count += 1

    return count
