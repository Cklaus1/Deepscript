"""Advanced speaker identification signals.

Extracts identity evidence from transcript segments that the base pipeline misses:
1. Cross-speaker name references (segment-level attribution)
2. Role-to-speaker correlation (topic content vs contact job titles)
3. Company/org mention extraction
4. Name frequency confidence boosting
5. Segment confidence filtering
6. Process of elimination (N-1 identified → Nth is known)
7. Meeting title name parsing ("Anuj and Chris" → names both speakers)
8. Self-identification detection ("Hi, I'm [name]" in opening segments)
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# --- 1. Cross-Speaker Name References ---

# Patterns where Speaker A addresses Speaker B by name
_ADDRESS_PATTERNS = [
    # "Hey Graham", "Hi Lori", "Thanks Patrick"
    re.compile(r'\b(?:hey|hi|hello|thanks|thank you|bye|goodbye)\s*[,]?\s*([A-Z][a-z]{2,})', re.IGNORECASE),
    # "what do you think, Graham?"
    re.compile(r'what do you (?:think|say)[,]?\s*([A-Z][a-z]{2,})\s*\?', re.IGNORECASE),
    # "you still on, Graham?" / "you still there, Graham?"
    re.compile(r'you (?:still|there)\s+(?:on|there)[,]?\s*([A-Z][a-z]{2,})', re.IGNORECASE),
    # "right, Graham?" / "okay Graham"
    re.compile(r'(?:right|okay|ok|so)\s*[,]\s*([A-Z][a-z]{2,})\s*\??', re.IGNORECASE),
    # "Graham, what" / "Graham, can you" (name at start followed by comma)
    re.compile(r'^([A-Z][a-z]{2,})\s*[,]\s*(?:what|can|could|do|did|are|is|how|tell|let)', re.IGNORECASE),
]

# Common words that look like names but aren't
_NOT_NAMES = frozenset({
    'the', 'and', 'but', 'this', 'that', 'what', 'when', 'where', 'who', 'how',
    'yeah', 'yes', 'just', 'like', 'well', 'right', 'okay', 'sure', 'let', 'can',
    'will', 'one', 'all', 'some', 'also', 'then', 'now', 'here', 'there', 'been',
    'have', 'has', 'had', 'was', 'were', 'are', 'his', 'her', 'its', 'our', 'may',
    'should', 'would', 'could', 'actually', 'really', 'basically', 'honestly',
    'again', 'maybe', 'exactly', 'absolutely', 'definitely', 'probably',
    'you', 'hey', 'sir', 'man', 'dude', 'bro', 'guys', 'everybody', 'everyone',
    'god', 'lord', 'wow', 'sorry', 'please', 'great', 'good', 'alright',
})


def extract_cross_speaker_references(
    segments: list[dict],
) -> list[dict]:
    """Find cases where Speaker A addresses Speaker B by name.

    When Speaker A says "Hey Graham" and the next different speaker is B,
    "Graham" is very likely B's name.

    Returns list of:
        {"name": str, "addressed_cluster": str, "speaker_cluster": str,
         "segment_id": int, "confidence": float, "text": str}
    """
    results = []
    if len(segments) < 2:
        return results

    for i in range(len(segments) - 1):
        seg = segments[i]
        text = seg.get("text", "")
        spk_a = seg.get("speaker_cluster_id", "")

        if not spk_a or not text:
            continue

        # Find the next segment from a DIFFERENT speaker
        spk_b = None
        for j in range(i + 1, min(i + 4, len(segments))):  # Look ahead up to 3 segments
            next_spk = segments[j].get("speaker_cluster_id", "")
            if next_spk and next_spk != spk_a:
                spk_b = next_spk
                break

        if not spk_b:
            continue

        # Check if this segment addresses someone by name
        for pattern in _ADDRESS_PATTERNS:
            m = pattern.search(text)
            if m:
                name = m.group(1).strip()
                if name.lower() in _NOT_NAMES or len(name) < 2:
                    continue

                # Validate: must be a real capitalized name, not a word fragment
                if not name[0].isupper() or not name.isalpha():
                    continue

                results.append({
                    "name": name,
                    "addressed_cluster": spk_b,
                    "speaker_cluster": spk_a,
                    "segment_id": seg.get("id", i),
                    "confidence": 0.90,
                    "text": text[:100],
                })
                break  # One match per segment

    # Deduplicate: same name → same cluster across multiple segments strengthens
    return results


# --- 2. Relationship Graph ---

@dataclass
class Relationship:
    """A relationship between the speaker and another person."""
    name: str
    relation: str  # wife, husband, dad, mom, son, daughter, assistant, boss, etc.
    cluster_id: str = ""  # If we can map to a cluster
    mentions: int = 1

_RELATIONSHIP_PATTERNS = [
    # "my wife Ingrid", "my husband John", "my son Will"
    re.compile(r'\bmy\s+(wife|husband|partner|girlfriend|boyfriend|fiancée?|spouse)\s+([A-Z][a-z]+)', re.IGNORECASE),
    re.compile(r'\bmy\s+(son|daughter|kid|child|brother|sister)\s+([A-Z][a-z]+)', re.IGNORECASE),
    re.compile(r'\bmy\s+(mom|dad|mother|father|parent)\s+([A-Z][a-z]+)', re.IGNORECASE),
    re.compile(r'\bmy\s+(assistant|secretary|associate|partner|boss|manager|attorney|lawyer|accountant|advisor)\s+([A-Z][a-z]+)', re.IGNORECASE),
    # "Thanks, dad" / "love you, mom" (at end of segment — identifies OTHER speaker)
    re.compile(r'(?:thanks|thank you|love you|bye|goodbye)\s*[,]?\s*(dad|mom|mother|father)\b', re.IGNORECASE),
    # "my sister Carrie said", "my brother Mike thinks"
    re.compile(r'\bmy\s+(sister|brother)\s+([A-Z][a-z]+)', re.IGNORECASE),
]


def extract_relationships(
    segments: list[dict],
    speaker_cluster_id: str | None = None,
) -> list[Relationship]:
    """Extract relationship signals from transcript text.

    Returns relationships mentioned by the specified speaker (or all speakers).
    """
    relationships: dict[str, Relationship] = {}

    for seg in segments:
        if speaker_cluster_id and seg.get("speaker_cluster_id") != speaker_cluster_id:
            continue

        text = seg.get("text", "")
        for pattern in _RELATIONSHIP_PATTERNS:
            for m in pattern.finditer(text):
                groups = m.groups()
                if len(groups) == 2:
                    relation, name = groups
                elif len(groups) == 1:
                    relation = groups[0]
                    name = relation  # "dad", "mom" — the relation IS the identifier
                else:
                    continue

                key = f"{relation.lower()}:{name.lower()}"
                if key in relationships:
                    relationships[key].mentions += 1
                else:
                    relationships[key] = Relationship(
                        name=name,
                        relation=relation.lower(),
                    )

    return list(relationships.values())


def build_relationship_graph(
    transcripts: list[tuple[Path, dict]],
) -> dict[str, list[Relationship]]:
    """Build a relationship graph across all transcripts.

    Returns: {cluster_id: [Relationship, ...]}
    """
    graph: dict[str, list[Relationship]] = defaultdict(list)

    for _, transcript in transcripts:
        segments = transcript.get("segments", [])
        diar = transcript.get("diarization", {})
        resolved = diar.get("speakers_resolved", [])

        # Get all cluster IDs in this call
        cluster_ids = set()
        for sr in resolved:
            cid = sr.get("speaker_cluster_id", "")
            if cid:
                cluster_ids.add(cid)

        for cid in cluster_ids:
            rels = extract_relationships(segments, speaker_cluster_id=cid)
            for rel in rels:
                # Check if relationship name matches another speaker in this call
                for other_cid in cluster_ids:
                    if other_cid == cid:
                        continue
                    # Will be matched downstream
                    rel.cluster_id = ""
                graph[cid].extend(rels)

    # Deduplicate per cluster
    for cid in graph:
        seen = {}
        for rel in graph[cid]:
            key = f"{rel.relation}:{rel.name.lower()}"
            if key in seen:
                seen[key].mentions += rel.mentions
            else:
                seen[key] = rel
        graph[cid] = list(seen.values())

    return graph


# --- 3. Role-to-Speaker Correlation ---

# Keywords that suggest professional roles
_ROLE_KEYWORDS: dict[str, list[str]] = {
    "attorney": ["trust", "estate", "legal", "liability", "statute", "fiduciary", "probate", "counsel"],
    "accountant": ["tax", "deduction", "audit", "irs", "filing", "return", "depreciation", "amortization"],
    "financial_advisor": ["portfolio", "investment", "asset allocation", "diversification", "yield", "fund"],
    "sales": ["pipeline", "quota", "deal", "prospect", "close", "commission", "territory", "objection"],
    "engineer": ["deploy", "api", "database", "architecture", "refactor", "sprint", "bug", "release"],
    "recruiter": ["candidate", "interview", "offer", "compensation", "hiring", "role", "position"],
    "executive": ["strategy", "board", "stakeholder", "revenue", "growth", "market", "compete"],
    "support": ["ticket", "issue", "resolution", "escalation", "sla", "customer satisfaction"],
    "insurance": ["policy", "premium", "coverage", "claim", "underwriting", "deductible", "liability"],
    "real_estate": ["property", "lease", "tenant", "mortgage", "appraisal", "zoning", "title"],
}


def compute_speaker_role_scores(
    segments: list[dict],
) -> dict[str, dict[str, float]]:
    """Compute role affinity scores for each speaker based on their vocabulary.

    Returns: {cluster_id: {"attorney": 0.73, "accountant": 0.12, ...}}
    """
    speaker_texts: dict[str, list[str]] = defaultdict(list)

    for seg in segments:
        cid = seg.get("speaker_cluster_id", "")
        text = seg.get("text", "")
        if cid and text:
            speaker_texts[cid].append(text.lower())

    results: dict[str, dict[str, float]] = {}
    for cid, texts in speaker_texts.items():
        combined = " ".join(texts)
        word_count = len(combined.split())
        if word_count < 20:
            continue

        role_scores: dict[str, float] = {}
        for role, keywords in _ROLE_KEYWORDS.items():
            hits = sum(combined.count(kw) for kw in keywords)
            # Normalize by word count
            score = min(1.0, hits / max(word_count * 0.01, 1))
            if score > 0.1:
                role_scores[role] = round(score, 3)

        if role_scores:
            results[cid] = role_scores

    return results


def match_roles_to_contacts(
    role_scores: dict[str, dict[str, float]],
    contacts: list[dict],
    calendar_attendees: list[str] | None = None,
) -> list[dict]:
    """Match speaker role profiles to contact job titles.

    Returns evidence entries for speakers whose topic profile matches a contact's role.
    """
    # Build contact role index
    contact_roles: list[dict] = []
    for c in contacts:
        title = (c.get("jobTitle") or "").lower()
        name = c.get("displayName") or ""
        if not title or not name:
            continue

        # Map contact title to role keywords
        matched_roles = set()
        for role, keywords in _ROLE_KEYWORDS.items():
            if any(kw in title for kw in [role] + keywords[:3]):
                matched_roles.add(role)
        # Common title patterns
        if any(w in title for w in ["attorney", "lawyer", "counsel", "legal"]):
            matched_roles.add("attorney")
        if any(w in title for w in ["cpa", "accountant", "tax", "audit"]):
            matched_roles.add("accountant")
        if any(w in title for w in ["advisor", "wealth", "financial planner"]):
            matched_roles.add("financial_advisor")
        if any(w in title for w in ["ceo", "president", "vp", "director", "chief"]):
            matched_roles.add("executive")
        if any(w in title for w in ["insurance", "underwriter", "broker"]):
            matched_roles.add("insurance")
        if any(w in title for w in ["real estate", "property", "realtor"]):
            matched_roles.add("real_estate")

        if matched_roles:
            contact_roles.append({"name": name, "title": title, "roles": matched_roles})

    # Match speakers to contacts by role overlap
    evidence = []
    for cid, scores in role_scores.items():
        speaker_roles = set(scores.keys())
        for contact in contact_roles:
            overlap = speaker_roles & contact["roles"]
            if overlap:
                # Filter by calendar attendees if available
                if calendar_attendees:
                    name_lower = contact["name"].lower()
                    if not any(name_lower in att.lower() or att.lower() in name_lower for att in calendar_attendees):
                        continue

                overlap_score = len(overlap) / max(len(speaker_roles), len(contact["roles"]))
                # Require strong overlap AND high role scores to avoid noise
                avg_role_score = sum(scores.get(r, 0) for r in overlap) / len(overlap)
                if overlap_score >= 0.5 and avg_role_score >= 0.3:
                    evidence.append({
                        "cluster_id": cid,
                        "name": contact["name"],
                        "confidence": min(0.60, 0.40 + overlap_score * 0.25),
                        "detail": f"Role match: speaker topics [{', '.join(overlap)}] match {contact['name']} ({contact['title']})",
                        "source": "role_correlation",
                    })

    return evidence


# --- 4. Speaking Duration × Calendar Ordering ---

def compute_speaking_durations(
    segments: list[dict],
) -> dict[str, float]:
    """Compute total speaking duration per cluster in a transcript.

    Returns: {cluster_id: seconds}
    """
    durations: dict[str, float] = defaultdict(float)
    for seg in segments:
        cid = seg.get("speaker_cluster_id", "")
        if cid:
            durations[cid] += seg.get("end", 0) - seg.get("start", 0)
    return dict(sorted(durations.items(), key=lambda x: -x[1]))


def match_duration_to_calendar(
    durations: dict[str, float],
    event: dict,
    account_owner_cluster: str | None = None,
) -> list[dict]:
    """Correlate speaking duration ranking with calendar attendee ordering.

    Heuristic: organizer speaks most (if they're the account owner).
    The next highest speaker is likely the primary external attendee.
    """
    if not event or not durations:
        return []

    organizer = event.get("organizer", {}).get("emailAddress", {}).get("name", "")
    attendees = []
    for att in event.get("attendees", []):
        name = att.get("emailAddress", {}).get("name", "")
        if name and name.lower() != organizer.lower():
            attendees.append(name)

    if not attendees:
        return []

    # Rank speakers by duration (excluding account owner)
    ranked = [(cid, dur) for cid, dur in durations.items()
              if cid != account_owner_cluster]

    evidence = []
    # The top non-owner speaker is likely the primary attendee
    if ranked and attendees:
        top_speaker = ranked[0][0]
        primary_attendee = attendees[0]
        evidence.append({
            "cluster_id": top_speaker,
            "name": primary_attendee,
            "confidence": 0.45,
            "detail": f"Duration heuristic: top speaker ({ranked[0][1]:.0f}s) matches primary attendee {primary_attendee}",
            "source": "duration_calendar",
        })

    return evidence


# --- 5. Company/Org Mention Extraction ---

def extract_company_mentions(
    segments: list[dict],
    contacts: list[dict],
) -> dict[str, list[str]]:
    """Find company names from contacts mentioned in each speaker's segments.

    Returns: {cluster_id: ["Company A", "Company B"]}
    """
    # Build company name set from contacts — skip very short or common words
    _common_words = {"the", "inc", "llc", "corp", "company", "group", "services", "solutions"}
    companies: dict[str, str] = {}  # lowercase → original case
    for c in contacts:
        company = c.get("companyName") or ""
        if company and len(company) > 4 and company.lower() not in _common_words:
            companies[company.lower()] = company

    if not companies:
        return {}

    # Scan per-speaker segments
    speaker_companies: dict[str, Counter] = defaultdict(Counter)
    for seg in segments:
        cid = seg.get("speaker_cluster_id", "")
        text = (seg.get("text") or "").lower()
        if not cid or not text:
            continue

        for company_lower, company_orig in companies.items():
            if company_lower in text:
                speaker_companies[cid][company_orig] += 1

    return {cid: list(counts.keys()) for cid, counts in speaker_companies.items()}


# --- 6. Segment Confidence Filtering ---

def filter_low_confidence_segments(
    segments: list[dict],
    min_confidence: float = 0.40,
    max_no_speech: float = 0.80,
) -> list[dict]:
    """Filter out noise segments that would pollute speaker identification.

    Returns segments that pass quality thresholds.
    """
    filtered = []
    removed = 0
    for seg in segments:
        conf = seg.get("confidence", 1.0)
        nsp = seg.get("no_speech_prob", 0.0)
        if conf >= min_confidence and nsp <= max_no_speech:
            filtered.append(seg)
        else:
            removed += 1

    if removed:
        logger.debug("Filtered %d/%d low-confidence segments", removed, len(segments))
    return filtered


# --- 7. Name Frequency Boosting ---

def compute_name_frequencies(
    segments: list[dict],
) -> dict[str, dict[str, int]]:
    """Count how many times each name is mentioned per speaker.

    When other speakers mention "Graham" 12 times vs "Tom" once,
    Graham is a stronger name candidate.

    Returns: {cluster_id: {"Graham": 12, "Tom": 1}}
    """
    # First, collect all names mentioned per-cluster-by-others
    name_mentions: dict[str, Counter] = defaultdict(Counter)

    # Find all capitalized words that appear repeatedly (likely names)
    all_text = " ".join(seg.get("text", "") for seg in segments)
    # Extract capitalized words that appear 2+ times
    candidates = Counter(re.findall(r'\b([A-Z][a-z]{2,})\b', all_text))
    name_candidates = {name for name, count in candidates.items()
                       if count >= 2 and name.lower() not in _NOT_NAMES}

    if not name_candidates:
        return {}

    for seg in segments:
        cid = seg.get("speaker_cluster_id", "")
        text = seg.get("text", "")
        if not cid or not text:
            continue

        for name in name_candidates:
            count = text.count(name)
            if count > 0:
                # This speaker mentions this name — attribute to OTHER speakers
                # (we'll resolve which cluster later)
                name_mentions[cid][name] += count

    return dict(name_mentions)


# --- 6. Process of Elimination ---

def find_elimination_candidates(
    transcript: dict,
    profiles: dict[str, Any],
    event: dict | None = None,
    account_owner_name: str = "",
) -> list[dict]:
    """If N-1 of N speakers are identified, the Nth must be the remaining attendee.

    In a 3-person call where Chris and Anuj are identified, and the calendar
    shows Chris, Anuj, and Graham — the unknown speaker IS Graham (0.90 confidence).

    Also works without calendar: if a 2-person call has 1 identified speaker
    and the meeting title says "Chris and Graham", the unknown is Graham.
    """
    diar = transcript.get("diarization", {})
    resolved = diar.get("speakers_resolved", [])
    cids = [sr.get("speaker_cluster_id", "") for sr in resolved if sr.get("speaker_cluster_id")]

    if len(cids) < 2:
        return []

    # Split into identified and unidentified
    identified_names = set()
    unidentified = []
    for cid in cids:
        if cid in profiles and profiles[cid].likely_name:
            identified_names.add(profiles[cid].likely_name.lower().split(" (")[0])
        else:
            unidentified.append(cid)

    # Need exactly 1 unknown for elimination to work
    if len(unidentified) != 1:
        return []

    unknown_cid = unidentified[0]

    # Get candidate names from calendar event attendees
    candidate_names = []
    if event:
        for att in event.get("attendees", []):
            name = att.get("emailAddress", {}).get("name", "")
            if not name:
                continue
            name_lower = name.lower()
            # Skip account owner and already-identified speakers
            if account_owner_name and name_lower == account_owner_name:
                continue
            if any(name_lower.startswith(idn) or idn.startswith(name_lower) for idn in identified_names):
                continue
            candidate_names.append(name)

    if len(candidate_names) == 1:
        # Perfect elimination — exactly one unmatched attendee
        return [{
            "cluster_id": unknown_cid,
            "name": candidate_names[0],
            "confidence": 0.88,
            "detail": f"Process of elimination: only unidentified speaker, only unmatched attendee",
            "source": "elimination",
        }]
    elif len(candidate_names) > 1:
        # Multiple candidates — still useful at lower confidence
        return [{
            "cluster_id": unknown_cid,
            "name": candidate_names[0],
            "confidence": 0.55,
            "detail": f"Elimination candidate: 1 of {len(candidate_names)} unmatched attendees",
            "source": "elimination",
        }]

    return []


# --- 7. Meeting Title Name Parsing ---

# Patterns for extracting names from meeting titles
_TITLE_NAME_PATTERNS = [
    # "Aidan Pratt and Chris Klaus" → ["Aidan Pratt", "Chris Klaus"]
    re.compile(r'^([A-Z][a-z]+ [A-Z][a-z]+)\s+and\s+([A-Z][a-z]+ [A-Z][a-z]+)$'),
    # "Chris/Anuj catch-up" → ["Chris", "Anuj"]
    re.compile(r'^([A-Z][a-z]+)\s*/\s*([A-Z][a-z]+)'),
    # "team feedback (RD/CK)" → initials, harder to match
    # "Jen Whitlow and Chris Klaus" → full names
    re.compile(r'([A-Z][a-z]+ [A-Z][a-z]+)\s+and\s+([A-Z][a-z]+ [A-Z][a-z]+)'),
    # "X and Y" with single names
    re.compile(r'^([A-Z][a-z]{2,})\s+and\s+([A-Z][a-z]{2,})(?:\s|$)'),
    # "X / Y" with single names
    re.compile(r'^([A-Z][a-z]{2,})\s*/\s*([A-Z][a-z]{2,})(?:\s|$)'),
    # "Name1, Name2, and Name3 Meeting"
    re.compile(r'^([A-Z][a-z]+(?:\s[A-Z][a-z]+)?),\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)?),?\s+and\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)'),
]


def extract_names_from_title(title: str) -> list[str]:
    """Extract participant names from a calendar event title.

    Returns list of names found in the title.
    """
    if not title:
        return []

    names = []
    for pattern in _TITLE_NAME_PATTERNS:
        m = pattern.search(title)
        if m:
            for g in m.groups():
                if g and g.lower() not in _NOT_NAMES and len(g) >= 3:
                    names.append(g)
            if names:
                break

    return names


def match_title_names_to_speakers(
    transcript: dict,
    event: dict,
    profiles: dict[str, Any],
    account_owner_name: str = "",
) -> list[dict]:
    """Match names from meeting title to speakers in the transcript.

    "Anuj and Chris catch-up" + 2-speaker call → names both speakers.
    """
    title = event.get("subject", "")
    names = extract_names_from_title(title)
    if not names:
        return []

    # Filter out account owner
    names = [n for n in names if not (account_owner_name and n.lower().startswith(account_owner_name))]
    if not names:
        return []

    diar = transcript.get("diarization", {})
    resolved = diar.get("speakers_resolved", [])
    cids = [sr.get("speaker_cluster_id", "") for sr in resolved if sr.get("speaker_cluster_id")]

    # Find unidentified speakers
    unidentified = [cid for cid in cids
                    if cid in profiles and not profiles[cid].likely_name]

    evidence = []
    # If there's exactly 1 unidentified and 1 title name (after removing owner), match them
    if len(unidentified) == 1 and len(names) == 1:
        evidence.append({
            "cluster_id": unidentified[0],
            "name": names[0],
            "confidence": 0.85,
            "detail": f"Meeting title: \"{title}\"",
            "source": "title_parsing",
        })
    else:
        # Multiple names/speakers — add as weaker evidence
        for name in names:
            for cid in unidentified:
                evidence.append({
                    "cluster_id": cid,
                    "name": name,
                    "confidence": 0.50,
                    "detail": f"Meeting title name: \"{title}\"",
                    "source": "title_parsing",
                })

    return evidence


# --- 8. Self-Identification Detection ---

_SELF_ID_PATTERNS = [
    # "Hi, I'm John" / "Hey, I'm Sarah from Acme"
    re.compile(r"\b(?:hi|hey|hello)[,.]?\s+(?:i'm|i am|this is)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)", re.IGNORECASE),
    # "My name is John Smith"
    re.compile(r"\bmy name is\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)", re.IGNORECASE),
    # "This is John" / "This is John Smith calling"
    re.compile(r"\bthis is\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\b", re.IGNORECASE),
    # "It's John" / "Hey it's Sarah"
    re.compile(r"\b(?:hey\s+)?it'?s\s+([A-Z][a-z]+)\b", re.IGNORECASE),
    # "Speaking is John Smith"
    re.compile(r"\bspeaking is\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)", re.IGNORECASE),
    # "John here" / "John Smith here"
    re.compile(r"^([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\s+here\b", re.IGNORECASE),
]


def detect_self_identification(
    segments: list[dict],
    max_segments: int = 5,
) -> list[dict]:
    """Detect self-identification in the opening segments of a recording.

    "Hi, I'm Anuj" in segment 2 → the speaker of segment 2 is Anuj.

    Args:
        segments: Transcript segments.
        max_segments: Only check first N segments (intros happen early).

    Returns evidence entries.
    """
    evidence = []

    for seg in segments[:max_segments]:
        text = seg.get("text", "")
        cid = seg.get("speaker_cluster_id", "")
        if not text or not cid:
            continue

        for pattern in _SELF_ID_PATTERNS:
            m = pattern.search(text)
            if m:
                name = m.group(1).strip()
                if name.lower() in _NOT_NAMES or len(name) < 3:
                    continue
                if not name[0].isupper():
                    continue

                evidence.append({
                    "cluster_id": cid,
                    "name": name,
                    "confidence": 0.88,
                    "detail": f"Self-ID in opening: \"{text[:60]}\"",
                    "source": "self_identification",
                })
                break

    return evidence
