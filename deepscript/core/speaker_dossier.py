"""Speaker dossier — fact sheet + audio clip extraction for unknown speakers.

Generates a dossier for each unknown speaker containing:
1. 5-10 facts about the person (LLM-derived from their speech)
2. A 10-second audio clip of something distinctive they said
3. Their label, co-speakers, and call history

Used to help humans identify unknown speakers by listening to their voice
and reading a profile of what they talk about.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CLIP_DURATION = 10  # seconds
AUDIO_SOURCE_DIR = Path("/mnt/c/Users/cklau/OneDrive/Documents/Sound Recordings")


@dataclass
class SpeakerDossier:
    """Dossier for an unknown speaker."""
    cluster_id: str
    label: str
    facts: list[str]
    clip_path: str | None = None
    clip_transcript: str = ""
    clip_source: str = ""  # Which recording the clip is from
    clip_timestamp: str = ""  # MM:SS
    co_speakers: list[str] = field(default_factory=list)
    total_calls: int = 0
    category: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "label": self.label,
            "facts": self.facts,
            "clip_path": self.clip_path,
            "clip_transcript": self.clip_transcript,
            "clip_source": self.clip_source,
            "clip_timestamp": self.clip_timestamp,
            "co_speakers": self.co_speakers,
            "total_calls": self.total_calls,
            "category": self.category,
        }


def generate_dossiers(
    profiles: dict[str, Any],
    transcripts: list[tuple[Path, dict]],
    llm: Any,
    output_dir: str | Path = "CRM/Speaker-Clips",
    audio_source: str | Path | None = None,
    clip_duration: int = CLIP_DURATION,
) -> list[SpeakerDossier]:
    """Generate dossiers for all unidentified speakers.

    Args:
        profiles: Speaker profiles from identify_speakers().
        transcripts: List of (file_path, transcript_dict).
        llm: LLMProvider for fact extraction.
        output_dir: Where to save audio clips and dossier files.
        audio_source: Directory containing original audio files (.m4a).
        clip_duration: Duration of audio clips in seconds.

    Returns: List of generated dossiers.
    """
    from deepscript.core.speaker_panel import _gather_speech_samples, _format_speech_samples

    if audio_source is None:
        audio_source = AUDIO_SOURCE_DIR
    audio_source = Path(audio_source)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find speakers that need dossiers (no real name, just labels or unknown)
    targets = []
    for cid, profile in profiles.items():
        # Include speakers with LLM labels or no name
        is_labeled = any(e.source == "llm_label" for e in profile.evidence)
        has_real_name = profile.likely_name and not is_labeled and profile.name_confidence >= 0.60
        if not has_real_name:
            samples = _gather_speech_samples(cid, transcripts)
            total_segs = sum(len(s["segments"]) for s in samples)
            if total_segs >= 3:
                targets.append((cid, profile, samples))

    if not targets:
        logger.info("No speakers need dossiers")
        return []

    logger.info("Generating dossiers for %d speakers", len(targets))

    # Load fact prompt
    fact_prompt = _load_fact_prompt()

    dossiers: list[SpeakerDossier] = []
    for cid, profile, samples in targets:
        # Co-speaker names
        co_names = []
        for co_cid in sorted(profile.co_speakers, key=lambda c: -profile.co_speakers[c])[:5]:
            if co_cid in profiles and profiles[co_cid].likely_name:
                co_names.append(profiles[co_cid].likely_name.split(" (")[0])

        # Generate facts via LLM
        facts = _extract_facts(cid, profile, samples, co_names, llm, fact_prompt)

        # Find best clip segment
        best_seg, best_call = _find_best_clip_segment(cid, transcripts, clip_duration)

        # Extract audio clip
        clip_path = None
        clip_text = ""
        clip_source = ""
        clip_ts = ""
        if best_seg and best_call:
            clip_path = _extract_clip(
                cid, best_seg, best_call, audio_source, output_path, clip_duration
            )
            clip_text = best_seg.get("text", "")
            clip_source = best_call.stem
            start_sec = best_seg.get("start", 0)
            clip_ts = f"{int(start_sec)//60:02d}:{int(start_sec)%60:02d}"

        label = profile.likely_name or cid
        category = ""
        for e in profile.evidence:
            if e.source == "llm_label":
                category = e.detail.split("(")[1].split(")")[0] if "(" in e.detail else ""
                break

        dossier = SpeakerDossier(
            cluster_id=cid,
            label=label,
            facts=facts,
            clip_path=str(clip_path) if clip_path else None,
            clip_transcript=clip_text,
            clip_source=clip_source,
            clip_timestamp=clip_ts,
            co_speakers=co_names,
            total_calls=profile.total_calls,
            category=category,
        )
        dossiers.append(dossier)

        # Save individual dossier as markdown
        _save_dossier_md(dossier, output_path)

    # Save summary
    _save_summary(dossiers, output_path)

    logger.info("Generated %d dossiers in %s", len(dossiers), output_path)
    return dossiers


def _load_fact_prompt() -> str:
    return """Based on this speaker's conversation excerpts, list 5-10 FACTS about this person.

Focus on concrete, identifying details:
- Where they live or travel
- Their profession or expertise
- Companies, schools, or organizations they mention
- People they reference by name
- Specific projects or deals they discuss
- Hobbies, interests, personal details
- Speech patterns or catchphrases

## Speaker's segments:

{speech_samples}

Usually appears with: {co_speaker_names}

Respond with ONLY a JSON object:
{{
  "facts": ["fact 1", "fact 2", "fact 3", "fact 4", "fact 5"]
}}

Each fact should be 1 sentence. Be specific — "Lives in Florida" not "Mentions a state"."""


def _extract_facts(
    cid: str,
    profile: Any,
    samples: list[dict],
    co_names: list[str],
    llm: Any,
    prompt_template: str,
) -> list[str]:
    """Extract 5-10 facts about a speaker via LLM."""
    from deepscript.core.speaker_panel import _format_speech_samples

    prompt = prompt_template.format(
        speech_samples=_format_speech_samples(samples),
        co_speaker_names=", ".join(co_names) if co_names else "(unknown)",
    )

    response = llm.complete(prompt=prompt, system="Respond only with valid JSON.", max_tokens=800)
    if not response:
        return []

    try:
        text = response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            text = text.rsplit("```", 1)[0]
        data = json.loads(text)
        return data.get("facts", [])[:10]
    except (json.JSONDecodeError, KeyError):
        return []


def _find_best_clip_segment(
    cid: str,
    transcripts: list[tuple[Path, dict]],
    clip_duration: int,
) -> tuple[dict | None, Path | None]:
    """Find the best segment for a voice clip.

    Picks the segment where the speaker:
    1. Speaks continuously for close to clip_duration seconds
    2. Says something distinctive (not just "yeah" or filler)
    3. Has high audio confidence
    """
    best_seg = None
    best_call = None
    best_score = 0

    for file_path, transcript in transcripts:
        segments = transcript.get("segments", [])
        for seg in segments:
            if seg.get("speaker_cluster_id") != cid:
                continue

            text = seg.get("text", "").strip()
            duration = seg.get("end", 0) - seg.get("start", 0)
            confidence = seg.get("confidence", 0)
            words = text.split()

            # Skip very short or filler segments
            if len(words) < 5 or duration < 2:
                continue

            # Score: prefer longer, higher confidence, more words
            score = 0
            score += min(duration / clip_duration, 1.0) * 40  # Duration fit
            score += confidence * 30  # Audio quality
            score += min(len(words) / 15, 1.0) * 20  # Content richness
            # Bonus for distinctive content (proper nouns, numbers)
            if any(w[0].isupper() for w in words[1:] if w):
                score += 10

            if score > best_score:
                best_score = score
                best_seg = seg
                best_call = file_path

    return best_seg, best_call


def _extract_clip(
    cid: str,
    segment: dict,
    transcript_path: Path,
    audio_source: Path,
    output_dir: Path,
    clip_duration: int,
) -> Path | None:
    """Extract an audio clip using ffmpeg.

    Finds the original audio file and extracts clip_duration seconds
    starting at the segment's timestamp.
    """
    # Find the original audio file
    # Transcript is "Recording (120).json" → audio is "Recording (120).m4a"
    audio_name = transcript_path.stem + ".m4a"
    audio_path = audio_source / audio_name

    if not audio_path.exists():
        # Try other extensions
        for ext in [".wav", ".mp3", ".ogg", ".flac"]:
            alt = audio_source / (transcript_path.stem + ext)
            if alt.exists():
                audio_path = alt
                break
        else:
            logger.debug("Audio file not found: %s", audio_path)
            return None

    start_time = max(0, segment.get("start", 0) - 0.5)  # Small buffer before
    output_file = output_dir / f"{cid}_{transcript_path.stem[:20]}.mp3"

    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", str(start_time),
                "-i", str(audio_path),
                "-t", str(clip_duration),
                "-q:a", "2",  # Good quality MP3
                "-map", "a",  # Audio only
                str(output_file),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0 and output_file.exists():
            logger.debug("Extracted clip: %s", output_file)
            return output_file
        else:
            logger.warning("ffmpeg failed for %s: %s", cid, result.stderr[:200])
            return None
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning("Clip extraction failed: %s", e)
        return None


def _save_dossier_md(dossier: SpeakerDossier, output_dir: Path) -> None:
    """Save a speaker dossier as a markdown file."""
    import re
    safe_name = re.sub(r"[^a-zA-Z0-9 _\-]", "", dossier.label).strip()[:40]
    if not safe_name:
        safe_name = dossier.cluster_id

    lines = [
        f"# {dossier.label}",
        f"",
        f"**Cluster:** `{dossier.cluster_id}`",
        f"**Calls:** {dossier.total_calls}",
        f"**Usually with:** {', '.join(dossier.co_speakers) if dossier.co_speakers else '(unknown)'}",
        f"",
    ]

    if dossier.clip_path:
        lines.append(f"## Voice Clip")
        lines.append(f"")
        lines.append(f"**File:** [{Path(dossier.clip_path).name}]({dossier.clip_path})")
        lines.append(f"**From:** {dossier.clip_source} at {dossier.clip_timestamp}")
        lines.append(f"**Says:** \"{dossier.clip_transcript[:100]}\"")
        lines.append(f"")

    if dossier.facts:
        lines.append(f"## Facts")
        lines.append(f"")
        for i, fact in enumerate(dossier.facts, 1):
            lines.append(f"{i}. {fact}")
        lines.append(f"")

    path = output_dir / f"{safe_name}.md"
    with open(path, "w") as f:
        f.write("\n".join(lines))


def _save_summary(dossiers: list[SpeakerDossier], output_dir: Path) -> None:
    """Save a summary index of all dossiers."""
    lines = [
        "# Unknown Speaker Dossiers",
        "",
        "Listen to each clip and identify whose voice it is.",
        "",
        f"| # | Label | Calls | Clip | Co-speakers |",
        f"|---|-------|-------|------|-------------|",
    ]

    for i, d in enumerate(dossiers, 1):
        clip = f"[clip]({Path(d.clip_path).name})" if d.clip_path else "no clip"
        co = ", ".join(d.co_speakers[:3])
        lines.append(f"| {i} | {d.label} | {d.total_calls} | {clip} | {co} |")

    lines.append("")

    path = output_dir / "_Index.md"
    with open(path, "w") as f:
        f.write("\n".join(lines))
