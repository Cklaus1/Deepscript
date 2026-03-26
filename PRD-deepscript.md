# DeepScript — Transcript Intelligence Engine

**Version:** 0.1 (Concept)
**Date:** 2026-03-24
**Status:** Proposal
**Relationship to AudioScript:** Separate package, optional dependency. AudioScript transcribes; DeepScript analyzes.

---

## 1. Why a Separate Tool

AudioScript is an **audio → text + identity** pipeline (faster-whisper, diarization, speaker identity, calendar join, output formatting).
DeepScript is a **text → intelligence** engine (classification, coaching, insights, relationship analysis, deep identity reasoning).

| Concern | AudioScript | DeepScript |
|---------|-------------|------------|
| Input | Audio files | Any transcript (AudioScript, Zoom, Otter, manual) |
| Speaker identity | Cluster linking, embedding match, calendar lookup, review queue | LLM inference, relationship graphs, importance ranking |
| Calendar | Event lookup, attendee extraction (deterministic) | Pattern reasoning across recurring series |
| Dependencies | torch, faster-whisper, pyannote (~10GB) | LLM APIs, regex, nltk (~50MB) |
| Iteration speed | Slow (model changes are rare) | Fast (new prompts, frameworks weekly) |
| Install profile | Heavy (GPU-dependent) | Light (API keys or local LLM) |
| Core value | Accurate transcription | Actionable intelligence |

```
pip install audioscript                    # Transcription + speaker identity + sync
pip install deepscript                     # Analysis + deep identity reasoning
pip install audioscript[deepscript]        # Full pipeline
```

### Speaker Identity Split (see PRD-speaker-identity.md)

```
AudioScript (fast operational layer)        DeepScript (deeper reasoning layer)
┌─────────────────────────────┐           ┌─────────────────────────────────┐
│ Diarization (pyannote)       │           │ LLM identity inference          │
│ Embedding extraction         │           │ "discusses budgets → finance"   │
│ Cosine similarity matching   │           │                                 │
│ Stable spk_xxxx cluster IDs  │──────────→│ Cross-call behavioral analysis  │
│ Cross-call stitching         │  speaker  │ "appears in all Tuesday syncs"  │
│ Calendar event lookup        │  cluster  │                                 │
│ Attendee → candidate gen     │    IDs    │ Importance ranking              │
│ Regex name extraction        │           │ Relationship/role graphs        │
│ Confidence scoring           │           │ Ambiguous-speaker workflows     │
│ Review queue generation      │           │ Cross-project intelligence      │
└─────────────────────────────┘           └─────────────────────────────────┘
```

**Principle:** AudioScript links voices reliably. DeepScript figures out who they probably are and whether it matters.

### Cross-Call Speaker Intelligence (DeepScript Feature)

DeepScript identifies speakers across multiple calls by combining evidence from multiple sources:

**Evidence sources (strongest → weakest):**

| Source | Signal | Confidence |
|--------|--------|-----------|
| AudioScript voice embedding | Same voice = same person (deterministic) | 0.95+ |
| AudioScript LLM name extraction | "Chris addresses this speaker as 'Kim'" | 0.85 |
| Calendar event attendees | Meeting at 2pm with "Kim Anderson, Attorney" | 0.80 |
| Contact list match | "Kim" → "Kim Anderson, Anderson Law Group" | 0.75 |
| Email thread participants | Emailed "Adam" about this meeting | 0.70 |
| Cross-call topic analysis | This voice always discusses estate planning | 0.50 |
| Cross-call co-speaker patterns | This voice always appears with Chris | 0.40 |

**CLI:**
```bash
# Identify speakers across all calls
deepscript speakers identify --transcripts ./batch10-test/ \
  --speaker-db ./speaker_identities.json \
  --calendar ms365 \
  --contacts ms365

# View speaker profile
deepscript speakers profile spk_d7b4714c

# Output:
# spk_d7b4714c:
#   Likely name: Adam Schwartz (confidence: 0.92)
#   Evidence:
#     - LLM: "Chris addresses as 'Adam'" (3 calls)
#     - Calendar: "Adam Schwartz, Schwartz Wealth" (2 meetings)
#     - Contacts: "Adam Schwartz" in Outlook
#     - Topics: wealth management, portfolio review (5 calls)
#   Role: Financial advisor
#   Total calls: 8
#   Total speaking time: 4.2 hours
#   Co-speakers: Chris Klaus (8 calls), Kim Anderson (3 calls)
```

**Config:**
```yaml
speakers:
  cross_call: true
  sources:
    speaker_db: ./speaker_identities.json
    calendar: ms365       # ms365 | google | none
    contacts: ms365       # ms365 | google | none
    email: ms365          # ms365 | google | none
  auto_name_threshold: 0.80  # Auto-assign name at this confidence
  llm_inference: true         # Use LLM to infer identity from topics/context
```

**Integration:** AudioScript calls DeepScript as a post-processing step (if installed):
```yaml
# .audioscript.yaml
sync:
  deepscript:
    enabled: true
    config: .deepscript.yaml    # DeepScript's own config
```

---

## 2. DeepScript Modules

### Core (always available)

| Module | Purpose | LLM Required? |
|--------|---------|---------------|
| `classifier` | Classify transcript type (business-meeting, sales-call, discovery-call, family, voice-memo, etc.) | No (keyword + heuristic), LLM enhances |
| `topic_segmenter` | Break transcript into topic sections with index/TOC | Basic: pause + keyword shift. Good: LLM |
| `communication` | Speaking balance, questions, interruptions, engagement, talk ratio | No |
| `tagger` | Classification + topics → properties/tags for MiNotes/Obsidian | No |

### Classification-Specific Analyzers

Classification determines which analyzer runs. Each produces structured sections.

#### Business Meeting Analyzer
| Section | LLM? | Description |
|---------|------|-------------|
| Summary | Basic: extractive. Good: LLM abstractive | Meeting overview |
| Action Items | Basic: regex. Good: LLM with assignee+deadline | `- [ ]` format |
| Key Decisions | Basic: regex. Good: LLM | What was decided |
| Questions Raised | Basic: `?` extraction. Good: LLM unanswered detection | Open items |
| Follow-ups | Extracted from actions as `[[wikilinks]]` | MiNotes links |
| Attendees | From diarization + speaker DB | Participant list with stats |
| Communication Insights | Rule-based Tier 1 metrics | Speaking balance, engagement |

#### Sales Call Analyzer (`sales-call`)
Everything from Business Meeting, plus:

| Section | LLM? | Description |
|---------|------|-------------|
| Call Phases | Basic: keyword clusters. Good: LLM | Intro → Discovery → Demo → Objections → Pricing → Close |
| Methodology Score | LLM | MEDDIC, BANT, SPIN, or Challenger scorecard |
| Buying Signals | LLM | Ownership language, implementation questions, multi-threading |
| Risk Signals | Basic: hedge words. Good: LLM | Stall language, competitor mentions, vague next steps |
| Missed Opportunities | LLM | Moments where close was possible but not attempted |
| Competitor Analysis | Basic: name detection. Good: LLM | How competitors were handled |
| Next Steps Quality | LLM | Specific (date+action+owner) vs vague ("I'll follow up") |

#### Discovery Call Analyzer (`discovery-call`)
Everything from Business Meeting, plus:

| Section | LLM? | Description |
|---------|------|-------------|
| Call Phases | LLM | Context → Current State → Pain → Impact → Workarounds → Wrap-up |
| Mom Test Score | Basic: hypothetical detection. Good: LLM | Compliance with Mom Test rules |
| JTBD Extraction | LLM | "When [situation], I want [motivation], so I can [outcome]" |
| Pain Points | LLM | Extracted with severity (frequency + impact + emotion), ranked |
| Hidden Opportunities | LLM | Normalized pain, unstated needs, workflow gaps |
| Commitment Signals | LLM | Time, reputation, money commitments (real validation) |
| Compliment Traps | LLM | Enthusiasm without commitment (false positives) |

#### Relationship Analyzer (`family`, `partner`, `personal`)
Different from all above — focuses on communication health, not productivity.

| Section | LLM? | Description |
|---------|------|-------------|
| Listening Balance | No | Talk ratio, question ratio per person |
| Emotional Tone | LLM | Warmth, tension, neutrality arc across the call |
| Gottman Indicators | LLM | 4 Horsemen detection (criticism, contempt, defensiveness, stonewalling) |
| Appreciation Ratio | Basic: sentiment. Good: LLM | Positive:negative interaction ratio (healthy: 5:1) |
| Bids for Connection | LLM | Emotional bids detected + response (turn toward/away/against) |
| Repair Attempts | LLM | De-escalation attempts during tension |
| NVC Patterns | LLM | Observations vs judgments, feelings vs thoughts, needs vs strategies |
| We/I Language | No | Collaborative vs separate framing |
| Validation Moments | Basic: keyword. Good: LLM | "I understand", "that makes sense", acknowledgments |
| Engagement Signals | No | Word count per turn (opening up vs one-word answers) |
| Topic Interest Map | Basic: word count per topic. Good: LLM | What topics generate energy? |
| Growth Suggestions | LLM | Personalized, gentle coaching for stronger communication |

**Sensitivity note:** Relationship analysis output should be:
- Private by default (not exported to shared MiNotes unless explicitly configured)
- Framed positively ("opportunity to grow" not "you failed at")
- Backed by research citations (Gottman, NVC)
- Optional per-call (`--relationship-insights` flag, not automatic)

---

## 3. Topic Segmentation (All Call Types)

Every long call benefits from a **Topics Index** — a table of contents with timestamps.

### Output Format

```markdown
## Topics Index

| # | Topic | Time | Duration | Speakers |
|---|-------|------|----------|----------|
| 1 | [Portfolio Review](#topic-1-portfolio-review) | 00:00 | 12m | Dad, Advisor |
| 2 | [Real Estate Holdings](#topic-2-real-estate-holdings) | 12:15 | 8m | Dad, Mom, Advisor |
| 3 | [Trust Fund Updates](#topic-3-trust-fund-updates) | 20:30 | 15m | Mom, Advisor |

---

## Topic 1: Portfolio Review
**00:00 — 12:15** | Dad, Advisor

### Advisor — 00:00
Let's start with the quarterly numbers...
```

### Detection Methods

**Rule-based (no LLM):**
- Long pause (>5 seconds) + speaker pattern change
- Keyword cluster shift (finance words → real estate words → legal words)
- Explicit transition language ("let's move on to", "next topic", "now regarding")
- New speaker entering the conversation

**LLM-based (accurate):**
- Send transcript in chunks → "identify topic boundaries and name each topic"
- Returns: `[{topic: "Portfolio Review", start: 0.0, end: 735.0}, ...]`

**Hybrid (recommended):**
- Rule-based detects candidate boundaries (pause + keyword shift)
- LLM names and refines the topics
- Cheap: only the boundary regions go to LLM, not the full transcript

### Config

```yaml
# .deepscript.yaml
topics:
  enabled: true
  min_topic_duration: 60        # Seconds — don't create topics shorter than this
  max_topics: 20                # Cap to avoid over-segmentation
  method: hybrid                # rule | llm | hybrid
  index: true                   # Generate Topics Index section
```

---

## 4. Relationship-Specific Analysis Detail

### For Partner Calls

```markdown
## Relationship Insights

### Communication Health: 7.5/10

### Appreciation Ratio: 4.2:1
- Positive interactions: 21 (gratitude, agreement, humor, affection)
- Negative interactions: 5 (criticism, frustration, dismissal)
- Target: 5:1 (Gottman research) — close but room to grow

### Gottman Four Horsemen Check
- Criticism: 1 instance (22:15) — "You always forget to..."
  → *Reframe opportunity: "I feel frustrated when X happens because I need Y"*
- Contempt: None detected ✅
- Defensiveness: 1 instance (22:30) — "Well you never..."
  → *Try: "You're right, I could do better at that"*
- Stonewalling: None detected ✅

### Bids for Connection
| Bid | Response | Quality |
|-----|----------|---------|
| "I had such a weird day today" (03:15) | Asked follow-up questions | ✅ Turned toward |
| "Look what I found for the trip" (18:30) | "Mm-hmm" (continued own topic) | ⚠️ Turned away |
| "I miss you" (35:00) | "I miss you too, tell me more about..." | ✅ Turned toward |

**Turning-toward rate: 67%** (healthy couples: 86%)

### Listening Balance
- You: 55% | Partner: 45% — balanced ✅
- You asked 12 questions, partner asked 8

### Growth Suggestions
- When your partner shares something about their day, try reflecting back before responding with your own experience
- The "you always" pattern at 22:15 could be reframed: share how you feel + what you need, without generalizing
- You turned away from a bid at 18:30 — small moments of attention build trust over time
```

### For Parent-Child Calls

```markdown
## Connection Insights

### Engagement Level
- Son's average response length: 23 words (up from 15 last call ✅)
- Longest response: 45 words (about his coding project at 12:30)
- One-word answers: 4 (down from 8 last call ✅)

### What Sparked Energy
| Topic | Son's Engagement | Your Approach |
|-------|-----------------|---------------|
| School grades | Low (short answers) | Direct questions ❌ |
| His coding project | High (volunteered details) | Asked "show me" ✅ |
| Weekend plans | Medium | Offered ideas together ✅ |
| Chores | Low (defensive) | Instruction-style ❌ |

### What Worked
- Asking "can you show me?" about his project → longest, most animated response
- "What do you think we should do this weekend?" → collaborative, not directive
- Laughed together about the dog (28:00) — shared humor = connection

### Growth Suggestions
- Topics with high engagement (coding, games) are bridges — spend more time there before transitioning to harder topics
- Try "What was the best part of your day?" instead of "How was school?" — specific > general
- When he gives a short answer, try silence (wait 5 seconds) — he may fill the space
- The chores conversation went better when framed as "we need to figure this out together" vs "you need to do X"
```

### For Family Group Calls

```markdown
## Family Dynamics

### Participation
| Person | Talk % | Topics Initiated | Questions Asked |
|--------|--------|-----------------|-----------------|
| Dad    | 35%    | 4               | 6               |
| Mom    | 30%    | 3               | 12              |
| Sister | 20%    | 2               | 4               |
| You    | 15%    | 1               | 2               |

### Communication Patterns
- Mom asks the most questions (connector role)
- Dad initiates most topics (agenda-setter role)
- You spoke least — consider sharing more or being invited in
- Sister and you rarely spoke directly to each other (mostly through parents)

### Warmth Indicators
- Shared laughter: 5 moments
- Gratitude expressed: Dad→Mom (1), Mom→Sister (2)
- Future planning: 3 family activity plans discussed (all initiated by Mom)
```

---

## 5. Architecture

```
deepscript/
├── core/
│   ├── classifier.py           # Transcript type classification
│   ├── topic_segmenter.py      # Topic boundary detection + TOC
│   ├── communication.py        # Speaking balance, talk ratio, questions, engagement
│   └── tagger.py               # Properties/tags for MiNotes/Obsidian
├── analyzers/
│   ├── business.py             # Summary, actions, decisions, questions
│   ├── sales.py                # Phases, MEDDIC/BANT/SPIN, signals, close coaching
│   ├── discovery.py            # Mom Test, JTBD, pain points, hidden opportunities
│   └── relationship.py         # Gottman, NVC, appreciation ratio, bids, growth
├── llm/
│   ├── provider.py             # Claude/OpenAI adapter
│   ├── prompts/                # Prompt templates per analysis type
│   │   ├── classify.txt
│   │   ├── summarize.txt
│   │   ├── sales_score.txt
│   │   ├── discovery_score.txt
│   │   ├── relationship.txt
│   │   └── topic_segment.txt
│   └── cost_tracker.py         # Token usage tracking + budget enforcement
├── formatters/
│   ├── markdown.py             # Extended markdown with all sections
│   └── json.py                 # Structured JSON output
├── cli/
│   └── main.py                 # `deepscript analyze transcript.json`
└── config.py                   # DeepScript config (.deepscript.yaml)
```

### Input Format

DeepScript accepts any transcript as JSON. AudioScript outputs enriched transcripts with stable speaker identities:

```json
{
  "text": "full text...",
  "language": "en",
  "backend": "faster-whisper",
  "segments": [
    {
      "start": 0.0, "end": 2.5, "text": "Hello everyone",
      "speaker": "Alice",
      "speaker_cluster_id": "spk_a91f",
      "speaker_confidence": 0.94
    },
    {
      "start": 2.5, "end": 5.0, "text": "Thanks for joining",
      "speaker": "spk_31bc",
      "speaker_cluster_id": "spk_31bc",
      "speaker_confidence": 0.0
    }
  ],
  "diarization": {
    "num_speakers": 2,
    "speakers_resolved": [
      {
        "local_label": "SPEAKER_00",
        "speaker_cluster_id": "spk_a91f",
        "display_name": "Alice",
        "status": "confirmed",
        "confidence": 0.94,
        "source": "db_match",
        "is_new": false
      },
      {
        "local_label": "SPEAKER_01",
        "speaker_cluster_id": "spk_31bc",
        "display_name": null,
        "status": "unknown",
        "confidence": 1.0,
        "source": "auto_cluster",
        "is_new": true
      }
    ]
  },
  "metadata": {
    "audio": {"duration_seconds": 134.7, "creation_time": "2026-03-25T16:13:54Z"},
    "file": {"name": "Recording (203).m4a", "size_human": "3.2 MB"}
  }
}
```

**Key:** `speaker_cluster_id` is stable across calls. DeepScript uses it to:
- Track the same person across multiple transcripts
- Build per-speaker behavioral profiles
- Correlate with calendar attendees for identity inference
- Generate cross-call relationship graphs

Also works with non-AudioScript transcripts (Zoom, Otter) — those just won't have `speaker_cluster_id`.

### CLI

```bash
# Standalone
deepscript analyze transcript.json --output analysis.md
deepscript analyze transcript.json --type sales-call --methodology meddic
deepscript analyze transcript.json --type discovery-call --framework mom_test
deepscript analyze transcript.json --type family --relationship-insights

# With AudioScript pipeline
audioscript sync --deepscript   # Runs DeepScript after each transcription
```

---

## 6. Privacy & Sensitivity

### Relationship Analysis Safeguards
- **Opt-in only:** Never runs automatically. Requires explicit `--relationship-insights` or config flag
- **Private by default:** Output saved locally, NOT exported to MiNotes unless `relationship.export: true`
- **Positive framing:** "Growth opportunity" not "failure". "Consider trying" not "you should"
- **Research-backed:** Every suggestion cites the framework (Gottman, NVC)
- **No judgment scoring on relationships:** No "your relationship is 6/10". Instead: specific patterns + specific suggestions
- **Consent:** Ideally both parties consent to analysis. Config option: `relationship.consent_note: true` adds a note to output

### LLM Privacy
- Transcripts sent to LLM API contain personal conversations
- Config: `llm.provider: local` uses local LLM (e.g., Ollama) for maximum privacy
- Config: `llm.redact_names: true` replaces names with Speaker A/B before sending to API
- All API calls logged locally for audit

---

## 7. Complete Analyzer Taxonomy

### Classification → Analyzer Matrix

DeepScript detects the call type and runs the appropriate analyzer. 50+ call types organized into tiers.

#### Tier 1 — Build First (highest value, largest market)

| Classification | Analyzer | Key Sections | Primary User |
|---------------|----------|-------------|-------------|
| `sales-discovery` | SalesDiscoveryAnalyzer | MEDDIC/BANT/SPIN score, pain extraction, next steps quality | Sales teams |
| `sales-demo` | SalesDemoAnalyzer | Phase detection, pain-to-demo alignment, buying signals, close opportunities | Sales teams |
| `sales-negotiation` | NegotiationAnalyzer | Concession tracking, commitment language, anchoring analysis, close probability | Sales leaders |
| `customer-discovery` | DiscoveryAnalyzer | Mom Test score, JTBD extraction, pain severity, hidden opportunities | Founders |
| `business-meeting` | BusinessAnalyzer | Summary, actions, decisions, questions, communication insights | Everyone |
| `support-escalation` | SupportAnalyzer | Issue classification, emotion trajectory, resolution detection, empathy score | Support teams |
| `qbr` | QBRAnalyzer | Customer health score, expansion signals, churn risk, value realization | CS teams |
| `interview-behavioral` | BehavioralInterviewAnalyzer | STAR completeness, competency mapping, evidence strength, bias detection | Recruiting |
| `interview-technical` | TechnicalInterviewAnalyzer | Problem-solving approach, hints needed, communication clarity, depth score | Engineering hiring |

#### Tier 2 — Build Next (strong value, founder-critical)

| Classification | Analyzer | Key Sections | Primary User |
|---------------|----------|-------------|-------------|
| `investor-pitch` | PitchAnalyzer | Investor interest signals, questions asked (reveal concerns), objection handling, next step secured, pitch improvement areas | Founders |
| `pmf-call` | PMFAnalyzer | 8-dimension PMF score, Ellis classification, anti-PMF flags, feature request quality, Vohra segment analysis (cross-call) | Founders |
| `recruiter-screen` | RecruiterScreenAnalyzer | Qualification match, red flags, compensation alignment, culture fit signals, advancement recommendation | Recruiting |
| `reference-check` | ReferenceCheckAnalyzer | Reference enthusiasm (glowing vs faint praise), specificity, consistency with candidate claims, rehire conviction | Recruiting |
| `vendor-evaluation` | VendorAnalyzer | Requirement coverage score, pricing comparison, risk flags, decision criteria tracking, vendor ranking | IT/Procurement |
| `customer-onboarding` | OnboardingAnalyzer | Progress milestones, confusion detection, risk flags, time-to-value tracking | CS teams |
| `one-on-one` | OneOnOneAnalyzer | Topic coverage (career vs status), coaching quality, engagement signals, blocker tracking, burnout indicators | Managers |
| `cofounder-alignment` | CofounderAnalyzer | Vision alignment, decision dynamics, conflict patterns, role clarity, power balance | Founders |
| `advisory-call` | AdvisoryAnalyzer | Advice extraction, intro/connection offers (easily forgotten!), action items, advisor engagement | Founders |
| `board-meeting` | BoardAnalyzer | Decisions, concerns raised, help requests, governance compliance, board confidence | Founders/Execs |
| `cold-call` | ColdCallAnalyzer | Opener effectiveness, objection handling, meeting booked Y/N, time-to-pitch | SDRs |
| `performance-review` | PerformanceReviewAnalyzer | Evidence-based feedback score, goal clarity, employee receptiveness, development plan, legal compliance (documented concerns) | Managers/HR |
| `standup` | StandupAnalyzer | Blocker detection, who's stuck, sprint progress signals, time discipline (<15 min), action items | Eng teams |
| `all-hands` | AllHandsAnalyzer | Key announcements, employee questions (themed), unanswered questions, sentiment, leadership clarity | Leadership |
| `classroom` | ClassroomAnalyzer | Student engagement (questions), concept coverage, pacing, comprehension checks, study notes, terminology | Educators |
| `offer-negotiation` | OfferNegotiationAnalyzer | Candidate priorities (cash vs equity vs title), competing offer signals, acceptance probability, commitment language | Recruiting |

#### Tier 3 — Specialized (high-value niches)

| Classification | Analyzer | Key Sections | Primary User |
|---------------|----------|-------------|-------------|
| `churn-save` | ChurnAnalyzer | True churn reason, save likelihood, concession effectiveness, competitor switching to | CS teams |
| `renewal-expansion` | RenewalAnalyzer | Renewal likelihood, expansion opportunities, churn risk, stakeholder changes | Account managers |
| `investor-update` | InvestorUpdateAnalyzer | Metrics discussed, board confidence, strategic decisions, help requests, governance | Founders |
| `earnings-call` | EarningsAnalyzer | Financial metrics extraction, guidance changes, analyst concerns, management sentiment | Finance |
| `due-diligence` | DueDiligenceAnalyzer | Risk factors, information completeness, team assessment, market validation | Investors |
| `medical-appointment` | MedicalAnalyzer | Diagnosis, treatment plan, medications, follow-up instructions, questions to ask | Patients |
| `legal-consultation` | LegalAnalyzer | Legal assessment, risk exposure, action items, deadlines, document requests | Clients |
| `therapy-session` | TherapyAnalyzer | Goal progress, emotional patterns, coping strategies, homework tracking | Clients (private) |
| `coaching-session` | CoachingAnalyzer | GROW model tracking, commitment follow-through, insight moments, self-awareness growth | Coachees |
| `sprint-retro` | RetroAnalyzer | What went well/didn't/change, recurring themes, action item follow-through, psych safety | Eng teams |
| `postmortem` | PostmortemAnalyzer | Timeline, root cause, contributing factors, blamelessness score, prevention actions | SRE/Eng |
| `family` | RelationshipAnalyzer | Gottman indicators, appreciation ratio, participation equity, warmth signals | Personal |
| `partner` | RelationshipAnalyzer | Bids for connection, NVC patterns, 4 Horsemen check, growth suggestions | Personal |
| `lecture` | LectureAnalyzer | Key concepts, topic outline, Q&A extraction, study notes, terminology glossary | Students |
| `podcast` | PodcastAnalyzer | Show notes, key quotes, guest bio, topic timestamps, episode summary | Creators |
| `fundraising-donor` | FundraisingAnalyzer | Donor interest, ask amount, commitment signals, follow-up actions | Nonprofits |
| `voice-memo` | SimpleAnalyzer | Quick summary, action items if any | Everyone |

#### Founder-Critical Analyzers (your use case)

These are the ones that directly help founders build companies:

| Call Type | Why Founders Need This |
|-----------|----------------------|
| **Customer discovery** | Validate problem-market fit without self-deception (Mom Test catches false validation) |
| **Investor pitch** | Know which concerns investors have before the follow-up — improve the pitch iteratively |
| **Sales calls** | Founders sell first — MEDDIC scoring teaches you what you're missing before you hire a sales team |
| **Co-founder alignment** | Catch misalignment before it becomes a crisis. Track decision patterns |
| **Advisory/mentor calls** | Capture introductions (often mentioned casually, then forgotten), track which advice you acted on |
| **Board meetings** | Document decisions and concerns with full evidence trail |
| **Recruiting interviews** | Score candidates consistently, detect bias in your own interviewing |
| **Vendor evaluation** | Compare 5 vendor demos systematically instead of going with "gut feel" |
| **Customer onboarding** | Find where customers get confused — that's your product feedback |
| **QBRs** | Detect churn risk before the customer tells you they're leaving |

#### Founder-Critical: PMF Analyzer (`pmf-call`)

**This is whitespace.** No existing tool measures Product-Market Fit from call transcripts. Gong tracks sales metrics, Dovetail tags research themes — but nobody has built the Sean Ellis / Rahul Vohra PMF engine on top of conversation analysis.

**Classification trigger:** Customer calls where the product is discussed — onboarding calls, feedback calls, QBRs, customer discovery post-launch. Auto-detected or manual: `--type pmf-call`.

**PMF Score (0-10, composite of 8 dimensions):**

```markdown
## Product-Market Fit Score: 6.8/10

### Dimension Scores
| Dimension | Score | Evidence |
|-----------|-------|----------|
| Emotional Intensity | 7/10 | "We love the dashboard" — specific, unprompted at 08:15 |
| Workflow Integration | 8/10 | Described 3 workflows, product is central to Monday team review |
| Referral / Evangelism | 5/10 | "I mentioned it to a colleague" — vague, no outcome |
| Switching Cost | 7/10 | "We've built our reporting process around it" |
| Feature Request Quality | 8/10 | All requests are incremental (extend existing), not fundamental |
| Willingness to Pay | 6/10 | No price complaints, but no unsolicited ROI statements |
| Urgency | 5/10 | No expansion asks, no timeline pressure |
| Unprompted Praise | 7/10 | 3 positive statements before any question |

### PMF Stage: Solution Validation → Approaching PMF
### Ellis Classification: "Somewhat Disappointed" (would miss it, but could adapt)

### Strongest Signals
- "Every Monday our team starts with the dashboard" — habitual use ✅
- "We've built our reporting process around it" — switching cost ✅
- Feature requests are all incremental (export formats, filters) — good sign ✅

### Anti-PMF Flags
- "We're also still using [spreadsheet] for the detailed analysis" — not fully replacing alternatives ⚠️
- No organic referral evidence — product isn't remarkable enough to tell others yet ⚠️
- When asked about ROI, gave vague answer — hasn't internalized value quantification ⚠️

### Key Quotes (Signal-Rich)
1. "We love the dashboard — it saves us the Monday morning scramble" (Emotional + Workflow, 08:15)
2. "I mentioned it to Sarah on the ops team" (Weak referral, 22:30)
3. "If you could add export to PDF, that would be great" (Incremental request, 28:00)
4. "We still use the spreadsheet for deep-dive analysis" (Anti-PMF: alternative still active, 15:45)
```

**Cross-Call PMF Dashboard (Vohra Engine):**

When 20+ PMF calls are in CMS, the distiller generates:

```markdown
# PMF Dashboard — Generated from 24 customer calls

## Ellis Distribution
- Very Disappointed (strong PMF): 29% (7/24) — target: 40%+
- Somewhat Disappointed: 50% (12/24)
- Not Disappointed: 21% (5/24)

## PMF Score Trend
- Month 1 (8 calls): avg 4.2
- Month 2 (9 calls): avg 5.8
- Month 3 (7 calls): avg 6.5
→ Trending positive ✅ but below 40% "very disappointed" threshold

## Segment Analysis (Vohra Engine)
### "Very Disappointed" Segment Profile:
- Company size: 20-100 employees (100%)
- Role: Operations/RevOps managers (86%)
- Use case: Weekly team reporting (100%)
- Common phrase: "can't imagine going back to spreadsheets"
- They ALL describe the Monday morning workflow
→ **This is your PMF segment. Build for them.**

### "Somewhat Disappointed" Segment:
- Company size: mixed
- Top blocker: "Still need spreadsheets for detailed analysis"
- Top request: Advanced filtering + export (incremental ✅)
→ **Build advanced analytics to convert these to "very disappointed"**

### "Not Disappointed" Segment:
- Signed up but rarely active
- Describe product generically, can't articulate specific value
→ **Don't build for these users. They're not your market.**

## Top Value Proposition (from strong-PMF users):
"Eliminates the Monday morning scramble by giving the team a single view of what matters"

## PMF Blockers (from medium-PMF users):
1. Can't do deep-dive analysis (still need spreadsheets) — 8/12 mention
2. No export/sharing capability — 5/12 mention
3. Missing integration with [specific tool] — 4/12 mention

## Feature Priority Matrix
| Feature | Requested by Strong PMF | Requested by Weak PMF | Build? |
|---------|------------------------|----------------------|--------|
| PDF export | 4/7 (57%) | 1/5 (20%) | ✅ Yes — power users want it |
| Advanced filters | 5/7 (71%) | 2/5 (40%) | ✅ Yes — top PMF blocker |
| Mobile app | 0/7 (0%) | 3/5 (60%) | ❌ No — weak users want it, strong users don't |
| Slack integration | 3/7 (43%) | 0/5 (0%) | ✅ Yes — strong users extending workflow |
```

**Detection signals (rule-based + LLM):**

| Signal Type | Rule-Based Detection | LLM Enhancement |
|------------|---------------------|-----------------|
| **Dependency language** | "can't live without", "critical", "essential" | Distinguish genuine vs. polite |
| **Evangelism** | "told", "recommended", "showed" + person name | Detect outcome of referral |
| **Workflow depth** | Count features/workflows mentioned | Assess centrality vs peripheralness |
| **Switching cost** | "built around", "integrated", "migrated" | Assess actual lock-in strength |
| **Feature request quality** | Classify as incremental vs fundamental | Map to product strategy |
| **WTP signals** | Price/cost mentions, ROI language | Detect price anchoring, value framing |
| **Anti-PMF patterns** | "nice to have", "alternative", "also using" | Detect politeness masking disinterest |
| **Ellis tier** | Composite of above | Full tier classification with confidence |

**Config:**
```yaml
# .deepscript.yaml
pmf:
  enabled: true
  ellis_threshold: 0.40        # Target % of "very disappointed" for PMF
  min_calls_for_dashboard: 10  # Minimum calls before generating PMF dashboard
  segment_by: [company_size, role, use_case]  # Vohra segmentation dimensions
  track_trend: true            # PMF score over time
```

### Cross-Cutting Capabilities (Apply to All Types)

| Capability | Rule-Based? | Description |
|-----------|-------------|-------------|
| Topic segmentation + index | Hybrid | Break any long call into sections with TOC |
| Action item extraction | Hybrid | `- [ ]` with assignee + deadline when detectable |
| Sentiment arc | LLM | Positive/negative trajectory across the call |
| Talk ratio / speaking balance | Rule-based | Per-speaker stats |
| Hedge word tracking | Rule-based | Low-priority awareness metric (extreme outliers only) |
| Question quality scoring | LLM | Open vs closed, leading vs neutral, hypothetical vs specific |
| Commitment language detection | LLM | "We will" vs "We might" vs "We'd like to" |
| Follow-up scheduling | Hybrid | Next meeting, deadline, follow-up action detection |
| Cross-call pattern analysis | LLM | Trends across multiple calls (same person, same deal, same team) |
| Compliance verification | Hybrid | Did required topics get covered? (regulatory, governance) |
| Risk flag detection | LLM | Generic concern/risk pattern detection |

---

## 8. BTask CMS Integration — Automatic Playbook Generation

### Why CMS (Cognitive Memory System)

BTask's CMS at `/root/projects/BTask/packages/cms/` is an episodic/semantic memory engine with quality gating. It already has the infrastructure DeepScript needs:

| CMS Capability | DeepScript Use |
|---------------|----------------|
| **Episodic memory** (JSONL) | Each call analysis → episode with scores, findings, outcomes |
| **Semantic memory** (playbooks/) | Distilled patterns across calls → auto-generated playbooks |
| **Dead-ends** | Failed approaches: "never discuss pricing before minute 20" |
| **Gating** (Elo ranking) | Quality control — only proven patterns become playbooks |
| **Distiller** | Automatically extracts patterns from 50 episodes → promotes to playbook |
| **MCP tools** | `cms_write_episode`, `cms_get_working_memory` for agent integration |

### Data Flow

```
AudioScript                  DeepScript                    CMS
┌──────────┐               ┌──────────────┐            ┌──────────────┐
│ 219 audio│──transcript──→│ Classify     │            │              │
│ files    │   (JSON)      │ Analyze      │──episode──→│ Episodes     │
│          │               │ Score        │  (JSONL)   │ (per call)   │
└──────────┘               │ Extract      │            │              │
                           └──────────────┘            │  Distiller   │
                                                       │  ↓           │
                                                       │  Playbooks   │
                                                       │  (markdown)  │
                                                       │              │
                                                       │  Dead-Ends   │
                                                       │  (JSONL)     │
                                                       └──────────────┘
                                                            ↓
                                                       Working Memory
                                                       (assembled for
                                                        next call prep)
```

### Episode Schema (extends CMS Episode)

```python
@dataclass
class CallEpisode:
    # Standard CMS fields
    episode_id: str                     # ep_call_xxx
    timestamp: str                      # ISO 8601
    mode: str = "call_analysis"

    # Call-specific context
    call_type: str                      # "sales-discovery", "investor-pitch", etc.
    source_file: str                    # Audio file path
    duration_seconds: float
    speakers: list[dict]               # [{cluster_id, display_name, status, confidence}]
    speaker_cluster_ids: list[str]     # ["spk_a91f", "spk_31bc"] — stable across calls
    classification_confidence: float

    # Analysis outcome
    methodology_score: dict | None      # MEDDIC: {metrics: 2, econ_buyer: 1, ...}
    pain_points: list[dict] | None      # [{pain, severity, frequency, evidence}]
    action_items: list[dict] | None     # [{text, assignee, deadline}]
    buying_signals: list[str] | None
    risk_signals: list[str] | None
    communication_metrics: dict         # {talk_ratio, questions, monologue_max, ...}

    # Outcome tracking (filled in later)
    deal_outcome: str | None = None     # "won" | "lost" | "stalled" | None
    outcome_score: float | None = None  # 0-1, set when deal closes

    # CMS gating
    findings: list[str]                 # Extracted learnings
    confidence: float                   # Analysis confidence
    gate_decision: str | None = None    # "promote" | "hold" | "discard"
```

### Playbook Generation Pipeline

```
Step 1: Accumulate Episodes
  - 50 sales discovery calls analyzed → 50 CallEpisodes in CMS

Step 2: CMS Distiller Runs
  - Groups episodes by call_type
  - Extracts patterns:
    - "In 78% of won deals, pain point X was mentioned"
    - "Average talk ratio in won deals: 42% vs lost: 65%"
    - "Implication questions correlate with 2x close rate"

Step 3: Gating
  - Elo-rank patterns by outcome correlation
  - Only promote patterns with sufficient sample size (N>10)
  - Dead-end patterns demoted

Step 4: Playbook Assembly
  - Versioned markdown written to store/semantic/playbooks/
  - Sections: What Works, What Fails, Template, Benchmarks
  - Auto-updated as new calls come in

Step 5: Working Memory
  - Before a sales call, agent queries CMS:
    cms_get_working_memory(task_type="sales-discovery")
  - Returns: relevant playbook sections + dead-ends to avoid
  - Could be used for: call prep notes, real-time coaching, post-call review
```

### Playbook Types Auto-Generated

| Playbook | Built From | Key Sections |
|----------|-----------|-------------|
| **Sales Discovery Playbook** | 50+ sales-discovery episodes | Best questions, pain points that close, objection responses ranked by outcome, talk ratio benchmarks |
| **Customer Discovery Playbook** | 30+ customer-discovery episodes | Best question patterns (Mom Test compliant), most common pain points, JTBD templates, false validation traps |
| **Pitch Playbook** | 20+ investor-pitch episodes | Questions investors ask (= concerns), what resonated most, objection handling, next-step progression rates |
| **Interview Playbook** | 40+ interview episodes | Best behavioral questions by competency, STAR completion rates, bias patterns detected, calibration data |
| **Objection Handling Playbook** | Extracted from all sales calls | Every objection → best response (from won deals) + worst response (from lost deals) |
| **Onboarding Playbook** | 20+ onboarding episodes | Common confusion points, time-to-value benchmarks, risk signal patterns |
| **Meeting Effectiveness Playbook** | 100+ business-meeting episodes | Best practices by meeting type, time allocation benchmarks, decision velocity norms |
| **Communication Coaching Playbook** | All calls with insights | Per-person trend data, talk ratio improvement, question quality, pause discipline |
| **PMF Playbook** | 20+ pmf-call episodes | Ellis distribution trend, segment profiles, top value prop (in users' words), PMF blockers ranked, feature priority matrix, anti-PMF pattern frequency |
| **Customer Discovery Playbook** | 30+ discovery episodes | Best question patterns, common pain points across interviews, JTBD templates, false validation traps, insight saturation tracker |

### Integration Config

```yaml
# .deepscript.yaml
cms:
  enabled: true
  store_path: /root/projects/BTask/packages/cms/store
  episode_mode: call_analysis

  # Playbook generation
  playbooks:
    enabled: true
    min_episodes: 10          # Minimum calls before generating playbook
    auto_update: true         # Regenerate when new episodes arrive
    outcome_tracking: true    # Track won/lost to improve playbooks over time

  # Dead-end tracking
  dead_ends:
    enabled: true             # Log failed patterns
    threshold: 3              # Mark as dead-end after 3 failures

  # Working memory integration
  working_memory:
    enabled: true             # Provide playbook context for call prep
    token_budget: 4000        # Max tokens for working memory assembly
```

### MCP Tool Extensions

```
# New MCP tools for DeepScript + CMS
deepscript_analyze_call:      Analyze a transcript → structured CallEpisode
deepscript_write_episode:     Write CallEpisode to CMS episodic memory
deepscript_get_playbook:      Retrieve auto-generated playbook by type
deepscript_prep_call:         Assemble working memory for upcoming call
deepscript_track_outcome:     Record deal outcome (won/lost) for playbook improvement
deepscript_query_patterns:    Query distilled patterns across calls
```

### Cross-Project Architecture

```
/root/projects/
├── audioscript/          # Audio → Text (transcription)
│   └── sync engine       # Watch folders, auto-transcribe
│
├── deepscript/           # Text → Intelligence (analysis)  [NEW REPO]
│   ├── analyzers/        # 41 call-type analyzers
│   ├── core/             # Classification, topics, communication
│   └── cms_bridge/       # CMS integration layer
│
├── BTask/
│   └── packages/
│       └── cms/          # Episodic → Semantic memory
│           └── store/
│               ├── episodes/call_analysis/   # DeepScript episodes land here
│               ├── semantic/playbooks/       # Auto-generated playbooks
│               └── gating/dead-ends.jsonl    # Failed patterns
│
└── MiNotes/              # Knowledge management (UI + search)
    └── Transcripts/      # Transcript pages with tags + properties
```

---

## 9. Config (.deepscript.yaml)

```yaml
# Classification
classify: true
custom_classifications:
  standup:
    keywords: [standup, blocker, yesterday, today, sprint]
  board-meeting:
    keywords: [board, directors, governance, fiduciary, resolution, quorum]

# Topic segmentation
topics:
  enabled: true
  method: hybrid            # rule | llm | hybrid
  min_duration: 60
  index: true

# Communication insights (data-backed metrics, rule-based, free)
communication:
  enabled: true
  speaking_balance: true    # Talk ratio, questions, monologue length, WPM
  engagement: true          # Speaker switches/min, silence %, participation Gini

# Business meeting analysis
business:
  summary: true
  action_items: true
  decisions: true
  questions: true

# Sales call analysis
sales:
  enabled: true
  methodology: meddic       # meddic | bant | spin | challenger
  competitors: [Gong, Chorus, Fireflies]
  close_coaching: true
  phase_detection: true

# Customer discovery analysis
discovery:
  enabled: true
  framework: mom_test        # mom_test | jtbd | problem_solution
  pain_extraction: true
  insight_mapping: true

# Relationship analysis (opt-in, private)
relationship:
  enabled: false             # Must be explicitly enabled
  frameworks: [gottman, nvc]
  appreciation_ratio: true
  bids_for_connection: true
  growth_suggestions: true
  export_to_minotes: false   # Private by default
  consent_note: true

# LLM configuration
llm:
  provider: claude           # claude | openai | ollama | none
  model: claude-sonnet-4-6   # Model for analysis
  # api_key: set via ANTHROPIC_API_KEY env var
  redact_names: false        # Replace names before sending to LLM
  budget_per_month: 50.00    # USD limit
  cost_tracking: true

# Output
output:
  format: markdown           # markdown | json
  sections: all              # all | [summary, actions, communication]

# Calendar context enrichment (via ms365-cli or gwscli)
calendar:
  enabled: true
  provider: ms365              # ms365 | google | none
  enrich_classification: true  # Use meeting metadata to boost classifier
  match_by: time_proximity     # Match transcript to calendar event by recording timestamp
  time_window: 30              # Minutes tolerance for matching

# Notification routing (post-analysis)
notifications:
  enabled: false
  channels:
    - type: teams              # via ms365-cli
      command: "ms365 chat send {chat_id} --body '{summary}'"
      on: [sales-call, business-meeting]
    - type: email              # via ms365-cli
      command: "ms365 mail send --to {user} --subject 'Call Analysis: {title}' --body '{summary}'"
      on: [investor-pitch, qbr]
    - type: google-chat        # via gwscli
      command: "gws chat +send --space {space} --text '{summary}'"
      on: [business-meeting]

# Scheduled batch processing (via bflow scheduler)
schedule:
  enabled: true
  type: cron
  cron_expression: "*/30 * * * *"  # Every 30 minutes
  scan_dir: "./transcripts"
  process_new: true                # Only unanalyzed files
```

---

## 10. BFlow Integration

DeepScript is managed by BFlow as a **concierge skill** — a SKILL.md file with CLI command reference. BFlow handles skill discovery, scheduling, and composition with other tools. DeepScript does **not** use complex YAML workflow orchestration.

### 10.1 DeepScript BFlow Skill

Installed at `bflow/concierge/skills/deepscript/SKILL.md`:

```yaml
---
name: deepscript
description: "Transcript intelligence — analyze calls for insights, action items, PMF scoring, relationship health, sales methodology. Use when user asks to analyze a transcript, meeting recording, or call."
metadata:
  bflow:
    emoji: "🔬"
    mode: "document"
    requires:
      bins: ["deepscript"]
      env: []  # ANTHROPIC_API_KEY optional (degrades to rule-based)
---
```

```markdown
# DeepScript — Transcript Intelligence

Analyze call transcripts for structured insights. Works with any transcript source (AudioScript, Zoom, Otter, manual).

## When to Use

- "Analyze this call recording", "What were the action items from that meeting?"
- "Score this sales call", "How did the discovery call go?"
- "Check our PMF signals from customer calls"
- After AudioScript transcription completes

## Quick Reference

| Operation | Command |
|-----------|---------|
| Analyze transcript | `deepscript analyze transcript.json -o markdown` |
| Force call type | `deepscript analyze transcript.json --type sales-call` |
| Sales with methodology | `deepscript analyze transcript.json --type sales-call --methodology meddic` |
| Discovery call | `deepscript analyze transcript.json --type discovery-call --framework mom_test` |
| Relationship insights | `deepscript analyze transcript.json --type family --relationship-insights` |
| Batch analyze | `deepscript analyze ./transcripts/ --recursive` |
| Classify only | `deepscript classify transcript.json` |
| Topic segmentation | `deepscript topics transcript.json` |

## Execution

All commands output structured JSON by default:
\```python
execute_skill("deepscript", "analyze transcript.json -o json")
\```

## Global Flags

- `--type <call-type>` — Override auto-classification
- `--tier <1|2|3>` — Analyzer tier (1=core, 2=extended, 3=specialized)
- `-o json|markdown` — Output format
- `--no-llm` — Rule-based only (no API calls)
- `--calendar` — Enrich with calendar context before analysis
- `--notify` — Send results via configured notification channels
```

### 10.2 Calendar Context Enrichment

Before analysis, DeepScript can pull calendar context via **ms365-cli** or **gwscli** to enrich the classifier and speaker identification.

**What calendar data provides:**

| Calendar Field | DeepScript Use |
|---------------|----------------|
| Meeting subject | Classification boost ("Sales Demo" → `sales-demo`, "1:1 with Sarah" → `one-on-one`) |
| Attendees | Speaker identification hints (match attendee names to diarized speakers) |
| Organizer | Identify who called the meeting (context for talk ratio analysis) |
| Recurring series | Cross-call pattern detection ("Every Tuesday sync" → track trends) |
| Meeting body/agenda | Pre-seed topic segmentation, identify expected discussion items |

**Calendar matching:** DeepScript matches a transcript to a calendar event by comparing the recording's creation timestamp against calendar events within a configurable time window (default: 30 minutes).

```bash
# ms365-cli: find calendar event near recording time
ms365 calendar view --start "2026-03-25T15:30:00" --end "2026-03-25T17:00:00" -o json

# gwscli: same lookup via Google Calendar
gws calendar events list --params '{"timeMin":"2026-03-25T15:30:00Z","timeMax":"2026-03-25T17:00:00Z","singleEvents":true}' --format json
```

**Enriched input:** When a calendar event is matched, DeepScript injects metadata into the analysis context:

```json
{
  "calendar_context": {
    "subject": "Q1 Pipeline Review",
    "organizer": "alice@company.com",
    "attendees": ["bob@company.com", "carol@company.com"],
    "recurring": true,
    "series_name": "Weekly Pipeline Review",
    "location": "Teams Meeting",
    "body_preview": "Agenda: 1) New leads 2) Stalled deals 3) Forecast update"
  }
}
```

This enrichment is **optional** — DeepScript works without it, but classification accuracy and speaker resolution improve significantly when calendar context is available.

### 10.3 Notification Routing

After analysis, DeepScript can push results to Teams, email, or Google Chat via the same CLI tools BFlow already uses.

**Flow:**
1. Analysis completes → structured output (JSON/markdown)
2. DeepScript formats a summary appropriate for the notification channel
3. Notification sent via CLI command (ms365-cli or gwscli)

**Routing rules** are configured per call type in `.deepscript.yaml` (see section 9). Examples:

```bash
# Post sales call analysis to Teams chat
ms365 chat send <chatId> --body "📊 Sales Call Analysis: Pipeline Review
Score: MEDDIC 7/10 | 3 buying signals, 1 risk flag
Action items: 3 (see full analysis)" -o json

# Email investor pitch analysis
ms365 mail send --to founder@company.com \
  --subject "Pitch Analysis: Acme Ventures" \
  --body "$(deepscript analyze pitch.json -o markdown)" -o json

# Post to Google Chat space
gws chat +send --space spaces/ABC123 \
  --text "Meeting summary posted: 4 action items, 2 decisions"
```

**Summary formatting** adapts to channel:
- **Teams/Chat**: Short summary (1-3 lines) with key metrics, link to full analysis
- **Email**: Full markdown analysis as email body
- **MiNotes**: Full structured output with frontmatter (existing behavior)

### 10.4 End-to-End Pipeline

BFlow composes DeepScript with other tools through sequential CLI invocations. There is no YAML workflow graph — each step is a standalone command.

```
┌─────────────────────────────────────────────────────────────────┐
│  BFlow Orchestration (skill composition, not YAML nodes)        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. AudioScript sync              (audio → transcript.json)    │
│     audioscript sync --watch                                    │
│         ↓ new transcript                                        │
│  2. Calendar lookup [optional]    (enrich with meeting context) │
│     ms365 calendar view ... -o json                             │
│     gws calendar events list ...                                │
│         ↓ calendar_context.json                                 │
│  3. DeepScript analyze            (transcript → intelligence)   │
│     deepscript analyze transcript.json --calendar               │
│         ↓ analysis.json / analysis.md                           │
│  4. CMS episode write [optional]  (store in BTask CMS)         │
│     deepscript cms-write analysis.json                          │
│         ↓ episode stored                                        │
│  5. Notification [optional]       (push to Teams/Chat/email)    │
│     ms365 chat send ... / gws chat +send ...                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**AudioScript integration point:** AudioScript's sync engine can invoke DeepScript as a post-processing step when `sync.deepscript.enabled: true` in `.audioscript.yaml`. This triggers steps 2-5 automatically after each transcription.

### 10.5 Scheduled Batch Processing

BFlow's scheduler can trigger DeepScript to scan for and analyze new transcripts on a cron schedule. This complements AudioScript's real-time sync — batch mode catches anything missed or processes transcripts from non-AudioScript sources.

```bash
# Cron entry (generated by bflow scheduler)
*/30 * * * * deepscript analyze ./transcripts/ --recursive --new-only --calendar --notify
```

**`--new-only`** checks the processing manifest (same pattern as AudioScript) to skip already-analyzed files. Combined with `--calendar` and `--notify`, a single scheduled command handles the full pipeline.
