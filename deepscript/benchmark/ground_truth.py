"""Ground truth annotations for benchmark evaluation.

Each fixture has annotated correct answers for precision/recall scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GroundTruth:
    """Expected correct outputs for a benchmark transcript."""

    call_type: str
    accepted_types: list[str] = field(default_factory=list)  # Alternative valid classifications
    action_items: list[str] = field(default_factory=list)
    action_assignees: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    pain_points: list[str] = field(default_factory=list)
    key_speakers: list[str] = field(default_factory=list)
    expected_summary_keywords: list[str] = field(default_factory=list)
    buying_signals: list[str] = field(default_factory=list)
    risk_signals: list[str] = field(default_factory=list)


# Ground truth for each fixture transcript
GROUND_TRUTHS: dict[str, GroundTruth] = {
    "sample_transcript": GroundTruth(
        call_type="business-meeting",
        action_items=[
            "draft the mobile app requirements by Friday",
            "set up a meeting with the design team",
            "follow up with the design team about brand guidelines",
        ],
        action_assignees=["Sarah", "Bob"],
        decisions=[
            "prioritize the mobile app redesign",
            "increase the marketing budget by 20% for Q2",
        ],
        key_speakers=["Alice", "Bob", "Sarah"],
        expected_summary_keywords=["quarterly", "review", "Q1", "revenue", "mobile", "roadmap"],
    ),
    "sales_transcript": GroundTruth(
        call_type="sales-call",
        accepted_types=["sales-discovery", "sales-demo"],  # Ambiguous — has discovery + demo elements
        action_items=[
            "send over a proposal document",
            "schedule a follow-up with VP of Operations",
        ],
        action_assignees=["Sales Rep"],
        decisions=[],
        pain_points=["spreadsheets", "things fall through the cracks"],
        key_speakers=["Sales Rep", "Prospect"],
        buying_signals=[
            "when we implement",
            "within our budget",
            "next Tuesday",
        ],
        risk_signals=[
            "need to check with our VP",
            "also looking at Asana and Monday.com",
            "let me get back to you",
        ],
    ),
    "discovery_transcript": GroundTruth(
        call_type="discovery-call",
        accepted_types=["customer-discovery"],
        action_items=[
            "pilot with Sarah and Mike next month",
            "email intro to Dave",
        ],
        pain_points=[
            "handoff between sales and CS breaks down",
            "CS can't find notes or they're incomplete",
            "customer asked same questions again",
            "20 hours a week reconstructing deal context",
        ],
        key_speakers=["Interviewer", "Customer"],
        expected_summary_keywords=["onboarding", "handoff", "sales", "CS", "context"],
    ),
    "pmf_transcript": GroundTruth(
        call_type="pmf-call",
        accepted_types=["customer-discovery", "discovery-call"],
        expected_summary_keywords=["dashboard", "team", "workflow", "Monday"],
        buying_signals=[
            "every Monday our team starts with the dashboard",
            "built our reporting process around it",
            "very disappointed",
        ],
        risk_signals=[
            "still use the spreadsheet for deep-dive analysis",
        ],
        key_speakers=["Interviewer", "Customer"],
    ),
    "interview_transcript": GroundTruth(
        call_type="interview-behavioral",
        accepted_types=["interview-technical"],  # Has both behavioral and technical elements
        key_speakers=["Interviewer", "Candidate"],
        expected_summary_keywords=["team", "conflict", "architecture", "testing", "Kubernetes"],
    ),
    "real_product_discussion": GroundTruth(
        call_type="discovery-call",
        accepted_types=["business-meeting", "sales-discovery", "pmf-call"],
        expected_summary_keywords=["audio", "transcription", "speakers", "software", "recording"],
        action_items=[],  # Casual discussion, few explicit actions
        key_speakers=[],  # Speakers are cluster IDs
    ),
    "real_meeting": GroundTruth(
        call_type="business-meeting",
        accepted_types=["advisory-call", "one-on-one"],
        expected_summary_keywords=["Georgia", "university", "research", "alliance", "faculty"],
        action_items=[],
        key_speakers=[],
    ),
}


def compute_precision_recall(
    extracted: list[str],
    ground_truth: list[str],
    fuzzy_threshold: float = 0.5,
) -> dict[str, float]:
    """Compute precision and recall for extracted items vs ground truth.

    Uses fuzzy substring matching — an extracted item matches if it shares
    enough words with a ground truth item.

    Returns: {"precision": 0-1, "recall": 0-1, "f1": 0-1, "matches": N}
    """
    if not extracted and not ground_truth:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "matches": 0}
    if not extracted:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "matches": 0}
    if not ground_truth:
        # No ground truth — can't evaluate precision meaningfully
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0, "matches": 0}

    matches = 0
    matched_gt: set[int] = set()

    for ext in extracted:
        ext_lower = ext.lower()
        ext_words = set(ext_lower.split())

        for i, gt in enumerate(ground_truth):
            if i in matched_gt:
                continue
            gt_lower = gt.lower()
            gt_words = set(gt_lower.split())

            # Fuzzy match: check word overlap
            if gt_words and ext_words:
                overlap = len(ext_words & gt_words) / max(len(gt_words), 1)
                # Also check substring containment
                if overlap >= fuzzy_threshold or gt_lower in ext_lower or ext_lower in gt_lower:
                    matches += 1
                    matched_gt.add(i)
                    break

    precision = matches / len(extracted) if extracted else 0
    recall = matches / len(ground_truth) if ground_truth else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "matches": matches,
    }


def verify_against_transcript(
    extracted_items: list[str],
    transcript_text: str,
    min_overlap_words: int = 2,
) -> dict[str, Any]:
    """Check if extracted items actually appear in the transcript (hallucination detection).

    Returns: {"grounded": N, "hallucinated": N, "grounding_rate": 0-1, "hallucinated_items": [...]}
    """
    text_lower = transcript_text.lower()
    text_words = set(text_lower.split())
    grounded = 0
    hallucinated_items: list[str] = []

    for item in extracted_items:
        item_lower = item.lower()
        item_words = set(item_lower.split())

        # Check if enough words from the item appear in the transcript
        overlap = len(item_words & text_words)
        # Also check substring
        if overlap >= min_overlap_words or item_lower in text_lower:
            grounded += 1
        else:
            hallucinated_items.append(item)

    total = len(extracted_items)
    return {
        "grounded": grounded,
        "hallucinated": len(hallucinated_items),
        "grounding_rate": round(grounded / total, 3) if total > 0 else 1.0,
        "hallucinated_items": hallucinated_items,
    }
