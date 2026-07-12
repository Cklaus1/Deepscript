"""Tests for speaker identification signals."""

from deepscript.core.speaker_signals import (
    extract_cross_speaker_references,
    extract_names_from_title,
    detect_self_identification,
    compute_speaking_durations,
    filter_low_confidence_segments,
    compute_name_frequencies,
    find_elimination_candidates,
    match_title_names_to_speakers,
    compute_speaker_role_scores,
    match_roles_to_contacts,
    match_duration_to_calendar,
    extract_company_mentions,
)


# --- Cross-Speaker References ---

def test_extract_cross_speaker_references_basic():
    """Detect when Speaker A addresses Speaker B by name."""
    segments = [
        {"text": "Hey Graham, what do you think?", "speaker_cluster_id": "spk_00", "id": 0},
        {"text": "I think we should go ahead.", "speaker_cluster_id": "spk_01", "id": 1},
    ]
    results = extract_cross_speaker_references(segments)
    assert len(results) == 1
    assert results[0]["name"] == "Graham"
    assert results[0]["addressed_cluster"] == "spk_01"


def test_extract_cross_speaker_references_no_match():
    """No references when only one speaker."""
    segments = [
        {"text": "Hello world", "speaker_cluster_id": "spk_00", "id": 0},
    ]
    assert extract_cross_speaker_references(segments) == []


def test_extract_cross_speaker_references_filters_not_names():
    """Common words are filtered out."""
    segments = [
        {"text": "Hey right, what do you think?", "speaker_cluster_id": "spk_00", "id": 0},
        {"text": "I agree.", "speaker_cluster_id": "spk_01", "id": 1},
    ]
    results = extract_cross_speaker_references(segments)
    assert len(results) == 0


def test_extract_cross_speaker_references_thanks():
    """'Thanks Patrick' pattern detected."""
    segments = [
        {"text": "Thanks Patrick, that helps a lot.", "speaker_cluster_id": "spk_00", "id": 0},
        {"text": "You're welcome.", "speaker_cluster_id": "spk_01", "id": 1},
    ]
    results = extract_cross_speaker_references(segments)
    assert len(results) == 1
    assert results[0]["name"] == "Patrick"


# --- Self Identification ---

def test_detect_self_identification_basic():
    """Detect 'Hi, I'm John' pattern."""
    segments = [
        {"text": "Hi, I'm John Smith", "speaker_cluster_id": "spk_00", "id": 0},
        {"text": "Hello John, nice to meet you.", "speaker_cluster_id": "spk_01", "id": 1},
    ]
    results = detect_self_identification(segments)
    assert len(results) == 1
    assert results[0]["name"] == "John Smith"
    assert results[0]["cluster_id"] == "spk_00"


def test_detect_self_identification_my_name_is():
    """Detect 'My name is John' pattern."""
    segments = [
        {"text": "My name is Sarah Johnson", "speaker_cluster_id": "spk_00", "id": 0},
    ]
    results = detect_self_identification(segments)
    assert len(results) == 1
    assert results[0]["name"] == "Sarah Johnson"


def test_detect_self_identification_this_is():
    """Detect 'This is John' pattern."""
    segments = [
        {"text": "This is John Smith calling about the meeting", "speaker_cluster_id": "spk_00", "id": 0},
    ]
    results = detect_self_identification(segments)
    assert len(results) == 1
    assert results[0]["name"] == "John Smith"


def test_detect_self_identification_no_match():
    """No self-ID in normal conversation."""
    segments = [
        {"text": "Hey, how are you doing today?", "speaker_cluster_id": "spk_00", "id": 0},
        {"text": "I'm doing great, thanks.", "speaker_cluster_id": "spk_01", "id": 1},
    ]
    assert detect_self_identification(segments) == []


# --- Meeting Title Name Parsing ---

def test_extract_names_from_title_and():
    """Extract names from 'Chris and Graham' title."""
    names = extract_names_from_title("Chris and Graham catch-up")
    assert "Chris" in names
    assert "Graham" in names


def test_extract_names_from_title_slash():
    """Extract names from 'Chris/Anuj' title."""
    names = extract_names_from_title("Chris/Anuj sync")
    assert "Chris" in names
    assert "Anuj" in names


def test_extract_names_from_title_full_names():
    """Extract full names from 'Jen Whitlow and Chris Klaus'."""
    names = extract_names_from_title("Jen Whitlow and Chris Klaus")
    assert "Jen Whitlow" in names
    assert "Chris Klaus" in names


def test_extract_names_from_title_no_names():
    """Return empty for titles without names."""
    names = extract_names_from_title("Weekly team standup")
    assert names == []


def test_match_title_names_to_speakers():
    """Match title names to unidentified speakers."""
    transcript = {
        "diarization": {
            "speakers_resolved": [
                {"speaker_cluster_id": "spk_00", "display_name": "Chris"},
                {"speaker_cluster_id": "spk_01", "display_name": None},
            ],
        },
    }
    profiles = {
        "spk_00": type("P", (), {"likely_name": "Chris Klaus"})(),
        "spk_01": type("P", (), {"likely_name": ""})(),
    }
    event = {"subject": "Chris and Graham catch-up"}
    results = match_title_names_to_speakers(
        transcript, event, profiles, account_owner_name="chris klaus"
    )
    # When multiple title names match, both are added as weaker evidence
    assert len(results) >= 1
    names = [r["name"] for r in results]
    assert "Graham" in names


# --- Speaking Durations ---

def test_compute_speaking_durations():
    """Compute duration per speaker cluster."""
    segments = [
        {"speaker_cluster_id": "spk_00", "start": 0, "end": 30},
        {"speaker_cluster_id": "spk_01", "start": 0, "end": 10},
        {"speaker_cluster_id": "spk_00", "start": 30, "end": 60},
    ]
    durations = compute_speaking_durations(segments)
    assert durations["spk_00"] == 60.0
    assert durations["spk_01"] == 10.0


# --- Low Confidence Filtering ---

def test_filter_low_confidence_segments():
    """Remove low-confidence segments."""
    segments = [
        {"confidence": 0.9, "no_speech_prob": 0.1},
        {"confidence": 0.3, "no_speech_prob": 0.1},  # Low confidence
        {"confidence": 0.9, "no_speech_prob": 0.9},  # High no-speech
        {"confidence": 0.7, "no_speech_prob": 0.5},
    ]
    filtered = filter_low_confidence_segments(segments)
    assert len(filtered) == 2


def test_filter_low_confidence_segments_all_pass():
    """All segments pass when quality is high."""
    segments = [
        {"confidence": 0.95, "no_speech_prob": 0.05},
        {"confidence": 0.90, "no_speech_prob": 0.10},
    ]
    filtered = filter_low_confidence_segments(segments)
    assert len(filtered) == 2


# --- Name Frequencies ---

def test_compute_name_frequencies():
    """Count name mentions per speaker."""
    segments = [
        {"speaker_cluster_id": "spk_00", "text": "Graham said Graham is great"},
        {"speaker_cluster_id": "spk_01", "text": "I agree with Graham"},
    ]
    freqs = compute_name_frequencies(segments)
    assert "spk_00" in freqs
    assert freqs["spk_00"]["Graham"] == 2


# --- Role Correlation ---

def test_compute_speaker_role_scores():
    """Compute role affinity from vocabulary."""
    segments = [
        {"speaker_cluster_id": "spk_00", "text": "The trust and estate liability statute requires a fiduciary"},
        {"speaker_cluster_id": "spk_00", "text": "We need to discuss probate and counsel"},
    ] * 20  # Need enough text for scoring
    scores = compute_speaker_role_scores(segments)
    assert "spk_00" in scores
    assert "attorney" in scores["spk_00"]


def test_match_roles_to_contacts():
    """Match speaker roles to contact job titles."""
    role_scores = {
        "spk_00": {"attorney": 0.8, "legal": 0.6},
    }
    contacts = [
        {"displayName": "Jane Doe", "jobTitle": "Estate Attorney", "emailAddresses": [{"address": "jane@example.com"}]},
    ]
    evidence = match_roles_to_contacts(role_scores, contacts)
    assert len(evidence) >= 1
    assert evidence[0]["name"] == "Jane Doe"


# --- Duration to Calendar ---

def test_match_duration_to_calendar():
    """Correlate speaking duration with calendar attendees."""
    durations = {"spk_00": 300.0, "spk_01": 150.0}
    event = {
        "organizer": {"emailAddress": {"name": "Chris Klaus"}},
        "attendees": [
            {"emailAddress": {"name": "Graham"}},
            {"emailAddress": {"name": "Chris Klaus"}},
        ],
    }
    evidence = match_duration_to_calendar(durations, event, account_owner_cluster="spk_00")
    assert len(evidence) >= 1
    assert evidence[0]["name"] == "Graham"


# --- Company Mentions ---

def test_extract_company_mentions():
    """Find company names in speaker segments."""
    segments = [
        {"speaker_cluster_id": "spk_00", "text": "Acme Corp is a great company"},
        {"speaker_cluster_id": "spk_01", "text": "I work at TechStart"},
    ]
    contacts = [
        {"companyName": "Acme Corp"},
        {"companyName": "TechStart"},
    ]
    result = extract_company_mentions(segments, contacts)
    assert "spk_00" in result
    assert "spk_01" in result


# --- Elimination ---

def test_find_elimination_candidates():
    """If N-1 identified, the Nth must be the remaining attendee."""
    profiles = {
        "spk_00": type("P", (), {"likely_name": "Chris Klaus", "evidence": []})(),
        "spk_01": type("P", (), {"likely_name": "", "evidence": []})(),
    }
    transcript = {
        "diarization": {
            "speakers_resolved": [
                {"speaker_cluster_id": "spk_00"},
                {"speaker_cluster_id": "spk_01"},
            ],
        },
    }
    event = {
        "attendees": [
            {"emailAddress": {"name": "Chris Klaus"}},
            {"emailAddress": {"name": "Graham"}},
        ],
    }
    results = find_elimination_candidates(
        transcript, profiles, event=event, account_owner_name="chris klaus"
    )
    assert len(results) == 1
    assert results[0]["name"] == "Graham"
    assert results[0]["cluster_id"] == "spk_01"


def test_find_elimination_candidates_no_unknowns():
    """No elimination when all speakers identified."""
    profiles = {
        "spk_00": type("P", (), {"likely_name": "Chris Klaus", "evidence": []})(),
        "spk_01": type("P", (), {"likely_name": "Graham", "evidence": []})(),
    }
    transcript = {
        "diarization": {
            "speakers_resolved": [
                {"speaker_cluster_id": "spk_00"},
                {"speaker_cluster_id": "spk_01"},
            ],
        },
    }
    results = find_elimination_candidates(transcript, profiles)
    assert results == []


