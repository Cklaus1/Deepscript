"""Communication metrics — rule-based analysis of speaking patterns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SpeakerStats:
    """Per-speaker communication metrics."""

    speaker: str
    word_count: int
    segment_count: int
    question_count: int
    talk_ratio: float
    avg_words_per_turn: float
    longest_monologue_words: int


@dataclass
class CommunicationMetrics:
    """Aggregate communication metrics for a transcript."""

    total_speakers: int
    total_words: int
    total_segments: int
    total_questions: int
    speakers: list[SpeakerStats]
    speaking_balance: float  # 0.0 = one person talks, 1.0 = perfectly balanced
    speaker_switches_per_segment: float


def analyze_communication(transcript: dict[str, Any]) -> CommunicationMetrics:
    """Compute communication metrics from transcript segments.

    Args:
        transcript: Transcript dict with "segments" containing speaker labels.

    Returns:
        CommunicationMetrics with per-speaker and aggregate stats.
    """
    segments = transcript.get("segments", [])
    if not segments:
        return CommunicationMetrics(
            total_speakers=0,
            total_words=0,
            total_segments=0,
            total_questions=0,
            speakers=[],
            speaking_balance=0.0,
            speaker_switches_per_segment=0.0,
        )

    # Accumulate per-speaker stats
    speaker_words: dict[str, int] = {}
    speaker_segments: dict[str, int] = {}
    speaker_questions: dict[str, int] = {}
    speaker_longest_mono: dict[str, int] = {}

    # Track monologues (consecutive segments by same speaker)
    current_speaker = None
    current_mono_words = 0
    switches = 0

    for seg in segments:
        speaker = seg.get("speaker", "Unknown")
        text = seg.get("text", "").strip()
        words = len(text.split()) if text else 0

        speaker_words[speaker] = speaker_words.get(speaker, 0) + words
        speaker_segments[speaker] = speaker_segments.get(speaker, 0) + 1

        if text.rstrip().endswith("?"):
            speaker_questions[speaker] = speaker_questions.get(speaker, 0) + 1

        # Monologue tracking
        if speaker == current_speaker:
            current_mono_words += words
        else:
            # End previous monologue
            if current_speaker is not None:
                prev_max = speaker_longest_mono.get(current_speaker, 0)
                speaker_longest_mono[current_speaker] = max(prev_max, current_mono_words)
                switches += 1
            current_speaker = speaker
            current_mono_words = words

    # Close final monologue
    if current_speaker is not None:
        prev_max = speaker_longest_mono.get(current_speaker, 0)
        speaker_longest_mono[current_speaker] = max(prev_max, current_mono_words)

    total_words = sum(speaker_words.values())
    total_segments = len(segments)
    total_questions = sum(speaker_questions.values())

    # Build per-speaker stats
    speakers_list: list[SpeakerStats] = []
    for spk in sorted(speaker_words.keys()):
        wc = speaker_words[spk]
        sc = speaker_segments[spk]
        speakers_list.append(
            SpeakerStats(
                speaker=spk,
                word_count=wc,
                segment_count=sc,
                question_count=speaker_questions.get(spk, 0),
                talk_ratio=round(wc / total_words, 3) if total_words > 0 else 0.0,
                avg_words_per_turn=round(wc / sc, 1) if sc > 0 else 0.0,
                longest_monologue_words=speaker_longest_mono.get(spk, 0),
            )
        )

    # Speaking balance: 1 - Gini coefficient approximation
    # For N speakers with equal share, balance = 1.0
    n_speakers = len(speakers_list)
    if n_speakers <= 1 or total_words == 0:
        balance = 0.0 if n_speakers == 0 else 1.0
    else:
        ratios = [s.talk_ratio for s in speakers_list]
        # Gini: mean absolute difference / (2 * mean)
        mean_ratio = 1.0 / n_speakers
        mad = sum(abs(r - mean_ratio) for r in ratios) / n_speakers
        gini = mad / (2 * mean_ratio) if mean_ratio > 0 else 0.0
        balance = round(1.0 - gini, 3)

    switches_per_seg = round(switches / total_segments, 3) if total_segments > 0 else 0.0

    return CommunicationMetrics(
        total_speakers=n_speakers,
        total_words=total_words,
        total_segments=total_segments,
        total_questions=total_questions,
        speakers=speakers_list,
        speaking_balance=balance,
        speaker_switches_per_segment=switches_per_seg,
    )
