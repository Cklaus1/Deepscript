# GTMScript — Go-To-Market Intelligence Engine

**Version:** 0.1 (Concept)
**Date:** 2026-03-26
**Status:** Proposal
**Relationship:** Separate package. AudioScript transcribes → DeepScript analyzes → GTMScript tracks pipeline + deals + PMF.

---

## 1. Why a Separate Tool

| Concern | DeepScript | GTMScript |
|---------|-----------|-----------|
| Input | Any transcript | DeepScript analyses, emails, CRM data |
| Output | Structured intelligence per call | Pipeline state, deal health, PMF dashboard, coaching |
| State | Stateless (analyze and forget) | Stateful (tracks companies, deals, contacts over time) |
| Entity model | None (just transcripts) | Companies, Contacts, Deals, ICPs |
| Time horizon | Single call | Months of relationship across many calls |
| Core value | "What happened in this call?" | "Is this deal going to close? Is this our PMF segment?" |

```
pip install audioscript                    # Audio → Text
pip install deepscript                     # Text → Intelligence
pip install gtmscript                      # Intelligence → Pipeline + PMF
pip install audioscript[deepscript,gtm]    # Full pipeline
```

### Integration Architecture

```
AudioScript                DeepScript                 GTMScript
┌──────────┐             ┌──────────────┐          ┌───────────────────┐
│ Record   │──transcript→│ Classify     │─analysis→│ Link to Company   │
│ Transcribe│   (JSON)   │ Analyze      │  (JSON)  │ Update Deal State │
│          │             │ Score        │          │ Track PMF         │
└──────────┘             └──────────────┘          │ Coach Rep         │
                                                   │ Generate Follow-up│
Email (ms365/gws)──────────────────────────────────│ Ghost Detection   │
                                                   │ Pipeline View     │
Calendar (ms365/gws)───────────────────────────────│ Meeting Context   │
                                                   └───────────────────┘
                                                          ↓
                                                   MiNotes (UI layer)
                                                   BTask CMS (memory)
```

---

## 2. Entity Model

### ICP (Ideal Customer Profile)

```yaml
icp:
  name: "Series A SaaS Ops Teams"
  criteria:
    company_size: [20, 100]
    roles: ["VP Ops", "RevOps Manager", "Head of CS"]
    industries: ["SaaS", "Fintech", "Martech"]
    pain_signals: ["spreadsheet chaos", "manual reporting", "tool sprawl"]
    disqualifiers: ["enterprise (>500)", "pre-revenue", "no ops team"]
  channels: ["LinkedIn", "conferences", "warm intros", "Product Hunt"]
  target_count: 1000
```

### Company

```yaml
company:
  id: "comp_acme123"
  name: "Acme Corp"
  icp_match: "Series A SaaS Ops Teams"
  icp_score: 0.85  # How well they match ICP criteria

  # Firmographics
  size: 50
  industry: "SaaS"
  website: "acme.com"
  location: "San Francisco"

  # Relationship state
  stage: "validation"  # cold → outreach → discovery → validation → pilot → loi → deal → customer | dead
  stage_changed_at: "2026-03-20"
  first_contact: "2026-02-15"
  last_contact: "2026-03-25"

  # Contacts
  contacts:
    - id: "cont_sarah"
      name: "Sarah Chen"
      role: "VP Operations"
      is_champion: true
      is_decision_maker: false
      sentiment: "positive"
      last_spoke: "2026-03-25"

    - id: "cont_bob"
      name: "Bob Kim"
      role: "CEO"
      is_champion: false
      is_decision_maker: true
      sentiment: "neutral"
      last_spoke: "2026-03-10"

  # Intelligence (auto-populated from DeepScript)
  pain_map:
    - pain: "Spreadsheet-based reporting breaks every Monday"
      severity: "high"
      frequency: "weekly"
      source: "call_2026-03-15"
      workaround: "Manual copy-paste from 3 tools"

    - pain: "New hires take 3 weeks to learn the reporting process"
      severity: "medium"
      source: "call_2026-03-20"

  pmf_signals:
    current_score: 6.8
    ellis_classification: "somewhat_disappointed"
    trend: [4.2, 5.5, 6.8]  # Across 3 product calls
    strongest: "workflow integration"
    blocker: "still using spreadsheets for deep analysis"

  buying_signals:
    - signal: "Asked about enterprise pricing unprompted"
      call: "call_2026-03-25"
      strength: "strong"
    - signal: "Mentioned Q2 budget cycle"
      call: "call_2026-03-20"

  risk_signals:
    - signal: "CEO hasn't been on a call yet"
      severity: "high"
    - signal: "Mentioned evaluating 2 competitors"
      severity: "medium"

  # Deal
  deal:
    value: null  # Unknown until pricing discussed
    timeline: "Q2 2026"
    decision_process: "VP Ops recommends → CEO approves"
    competitors: ["Asana", "Monday.com"]
    next_step: "Schedule demo with CEO"
    next_step_date: "2026-03-28"
    next_step_owner: "us"

  # Calls
  calls:
    - id: "call_2026-03-15"
      type: "discovery-call"
      deepscript_episode: "ep_abc123"
      summary: "Identified spreadsheet pain, Sarah is champion"
      pain_points_found: 2
      commitment_signals: ["agreed to follow-up demo"]

    - id: "call_2026-03-25"
      type: "sales-call"
      deepscript_episode: "ep_def456"
      summary: "Demo went well, pricing discussed, needs CEO buy-in"
      buying_signals: 3
      risk_signals: 1

  # Emails
  email_threads:
    - thread_id: "thread_xyz"
      subject: "Re: Acme × [Product] — Next Steps"
      last_reply: "2026-03-26"
      last_reply_by: "them"
      status: "active"
      sentiment: "positive"

  # Health score (computed)
  health:
    score: 7.2
    factors:
      champion_identified: true
      decision_maker_engaged: false  # CEO hasn't been on a call
      pain_confirmed: true
      timeline_exists: true
      budget_discussed: true
      competitors_active: true
      multi_threaded: false  # Only talking to Sarah
      next_step_specific: true
    risks: ["No CEO engagement", "Competitor evaluation active"]
    recommendation: "Get CEO on next call. Prepare competitive battlecard for Asana."
```

### Deal

```yaml
deal:
  id: "deal_acme_q2"
  company: "comp_acme123"
  stage: "validation"
  value: null
  probability: 0.45  # Computed from signals
  expected_close: "2026-06"
  owner: "founder"

  # MEDDIC across all calls (aggregated)
  meddic:
    metrics: 2      # Mentioned "save 20 hours/week"
    economic_buyer: 1  # CEO identified but not engaged
    decision_criteria: 2  # Integration + pricing discussed
    decision_process: 2  # VP recommends → CEO approves
    identify_pain: 3    # Strong pain confirmed in 2 calls
    champion: 3         # Sarah actively pushing internally
    total: 13/18
    gaps: ["Economic buyer not engaged"]

  # Timeline
  events:
    - date: "2026-02-15"
      type: "outreach"
      note: "Cold LinkedIn message to Sarah"
    - date: "2026-03-01"
      type: "reply"
      note: "Sarah responded, interested"
    - date: "2026-03-15"
      type: "discovery-call"
      duration: 30
      outcome: "Pain confirmed, agreed to demo"
    - date: "2026-03-25"
      type: "demo"
      duration: 45
      outcome: "Positive, needs CEO buy-in"
    - date: "2026-03-28"
      type: "scheduled"
      note: "Demo with CEO"
```

---

## 3. Pipeline Stages

```
cold              Not yet contacted. In target list.
outreach          Contacted (email/LinkedIn). Waiting for response.
discovery         Had first conversation. Exploring pain.
validation        Pain confirmed. Testing product fit.
pilot             Using product (trial/POC).
negotiation       Pricing/terms discussion.
loi               Letter of intent or verbal commit.
deal              Signed. Customer.
dead              Not a fit. Documented why.
```

### Stage Transition Rules

| From → To | Trigger | Auto-detection |
|-----------|---------|---------------|
| cold → outreach | First email/message sent | Email detected via ms365/gws |
| outreach → discovery | First call completed | DeepScript analysis received |
| discovery → validation | Pain score ≥ 7, commitment signal detected | DeepScript signals |
| validation → pilot | Product access granted | Manual or webhook |
| pilot → negotiation | Pricing discussed on call | DeepScript buying signals |
| negotiation → loi | Verbal commit or LOI signed | Manual |
| loi → deal | Contract signed | Manual |
| any → dead | No response 30 days, or explicit "not interested" | Ghost detection / DeepScript |

---

## 4. Cross-Call Intelligence

### Pain Map (Aggregated Across Calls)

For each company, GTMScript maintains a **living pain map** — every pain point mentioned across all calls, deduplicated and ranked:

```markdown
## Acme Corp — Pain Map

| Pain | Severity | Frequency | First Mentioned | Confirmed In | Workaround |
|------|----------|-----------|-----------------|-------------|------------|
| Spreadsheet reporting breaks | High | Weekly | Call 1 (Mar 15) | Call 2, Call 3 | Manual copy-paste |
| New hire onboarding too slow | Medium | Per hire | Call 2 (Mar 20) | — | Shadow for 3 weeks |
| No single source of truth | High | Daily | Call 2 (Mar 20) | Call 3 | Slack channels |
```

### PMF Tracking (Per Company Over Time)

```markdown
## Acme Corp — PMF Trajectory

Call 1 (Mar 15): PMF 4.2 — Discovery, no product usage yet
Call 2 (Mar 20): PMF 5.5 — Started pilot, "this is useful"
Call 3 (Mar 25): PMF 6.8 — "Team loves the dashboard"
Trend: ↑ Improving

Ellis: "Somewhat Disappointed" → approaching "Very Disappointed"
Blocker: "Still need spreadsheets for deep analysis"
```

### PMF Dashboard (Across ALL Companies)

```markdown
## PMF Dashboard — 24 Companies in Validation/Pilot

Ellis Distribution:
- Very Disappointed: 29% (7/24) — target 40%+
- Somewhat Disappointed: 50% (12/24)
- Not Disappointed: 21% (5/24)

PMF Segment (Very Disappointed profile):
- Company size: 20-50 employees (100%)
- Role: Ops Manager (86%)
- Use case: Weekly team reporting (100%)
- Common phrase: "can't imagine going back"

Top Value Prop: "Eliminates Monday morning scramble"

PMF Blockers (Somewhat Disappointed → Very):
1. No deep-dive analysis (8/12 mention)
2. No export/sharing (5/12)
3. Missing [specific tool] integration (4/12)

Feature Priority:
| Feature | Strong PMF Want | Weak PMF Want | Build? |
|---------|----------------|---------------|--------|
| Advanced filters | 71% | 40% | ✅ Yes |
| PDF export | 57% | 20% | ✅ Yes |
| Mobile app | 0% | 60% | ❌ No |
```

---

## 5. Email Intelligence

### Ghost Detection

```yaml
ghost_detection:
  warning_days: 7    # Yellow flag after 7 days no response
  critical_days: 14  # Red flag after 14 days
  dead_days: 30      # Auto-move to dead after 30 days
  check_source: ms365  # ms365 | google
```

Monitor email threads with prospects:
- Track last reply date and who replied last
- Alert when prospect goes silent
- Detect "we're busy" / "circle back next quarter" stall language
- Auto-draft re-engagement email based on last conversation pain points

### Email Analysis

Run DeepScript on email threads:
```bash
# Fetch emails with a prospect and analyze
gtmscript email-intel "Acme Corp"
```

Extracts:
- Commitment language in emails vs calls (are they stronger/weaker in writing?)
- Response time patterns (getting slower = losing interest)
- CC/forwarding patterns (are they multi-threading internally? good sign)
- Attachment patterns (sent a proposal = high intent)

---

## 6. Rep Coaching (For When You Hire Reps)

Track per-rep metrics across all their calls:

```markdown
## Rep: Sarah (SDR)

### Talk Metrics (last 30 days, 45 calls)
- Talk ratio: 62% (target: 40-55%) ⚠️ Talking too much
- Questions per call: 6 avg (target: 11-14) ⚠️ Not asking enough
- Longest monologue: 3.2 min avg (target: <90s) ⚠️ Monologuing
- Filler words: 8 per call (target: <5)

### Methodology Compliance
- MEDDIC avg: 9/18 (target: 14+)
- Weakest: Economic Buyer (0.8/3) — not identifying who signs
- Strongest: Identify Pain (2.5/3)

### Deal Outcomes
- Calls to close: 4.2 avg
- Win rate: 28% (team avg: 32%)
- Biggest gap: Deals stall at "validation" — not converting to pilot

### Coaching Suggestions
1. "You're averaging 6 questions per call. Top performers ask 12. Try preparing 5 open-ended questions before each call."
2. "Your monologues average 3.2 minutes. Prospects check out after 90 seconds. Practice the 'pause and ask' technique."
3. "You identify pain well (2.5/3) but don't ask about the economic buyer (0.8/3). Add 'Who signs off on tools like this?' to your call script."
```

---

## 7. CLI Commands

```bash
# Company management
gtmscript company add "Acme Corp" --icp "Series A SaaS" --size 50 --industry SaaS
gtmscript company list --stage discovery
gtmscript company view "Acme Corp"
gtmscript company move "Acme Corp" --stage validation

# Link DeepScript analysis to company
gtmscript link call.analysis.json --company "Acme Corp"

# Auto-link (match by speaker, calendar, or email)
gtmscript auto-link ./analyses/

# Pipeline
gtmscript pipeline                    # Kanban view
gtmscript pipeline --stage discovery  # Filter
gtmscript pipeline --health red       # At-risk deals

# PMF Dashboard
gtmscript pmf                         # Cross-company PMF
gtmscript pmf --segment "20-50 employees"

# Email intelligence
gtmscript email-intel "Acme Corp"     # Analyze email thread
gtmscript ghosts                       # Show ghost prospects
gtmscript ghosts --re-engage          # Draft re-engagement emails

# ICP management
gtmscript icp define --name "Series A SaaS"
gtmscript icp import targets.csv      # Bulk import
gtmscript icp score                    # Score companies against ICP

# Rep coaching (when you have reps)
gtmscript coaching "Sarah"            # Rep performance report
gtmscript coaching --team             # Team benchmarks

# Deal intelligence
gtmscript deal "Acme Corp"            # Deal health + MEDDIC
gtmscript deal risks                  # All at-risk deals
gtmscript deal forecast               # Pipeline forecast

# Follow-up generation
gtmscript follow-up "Acme Corp"       # Draft follow-up email
gtmscript prep "Acme Corp"            # Call prep from all intelligence
```

---

## 8. Data Storage

### Option A: MiNotes (Recommended for Solo Founder)

Each company = a MiNotes page with YAML frontmatter:
```markdown
---
type: company
stage: validation
icp: "Series A SaaS"
pain_score: 8.2
pmf_score: 6.8
health: 7.2
next_step: "Demo with CEO"
next_step_date: 2026-03-28
last_contact: 2026-03-25
calls: 3
---

# Acme Corp

## Pain Map
...

## PMF Trajectory
...

## Call History
...
```

### Option B: BTask CMS (For Programmatic Access)

```
store/
├── companies/
│   ├── comp_acme123.json
│   └── comp_globex456.json
├── deals/
│   ├── deal_acme_q2.json
│   └── deal_globex_pilot.json
├── contacts/
│   └── ...
├── pipeline/
│   └── pipeline_state.json
└── analytics/
    ├── pmf_dashboard.json
    └── rep_coaching.json
```

### Option C: Both (MiNotes for UI, CMS for Compute)

MiNotes = human-readable pages you browse and edit.
CMS = structured JSON for programmatic queries and dashboards.
Sync between them.

---

## 9. Config (.gtmscript.yaml)

```yaml
# ICP
icp:
  name: "Series A SaaS Ops Teams"
  target_count: 1000
  criteria:
    company_size: [20, 100]
    roles: ["VP Ops", "RevOps Manager"]
    industries: ["SaaS", "Fintech"]

# Pipeline
pipeline:
  stages: [cold, outreach, discovery, validation, pilot, negotiation, loi, deal, dead]
  auto_advance: true  # Auto-move based on signals
  ghost_days: 30       # Days before auto-dead

# Data sources
sources:
  deepscript:
    enabled: true
    auto_link: true    # Auto-link analyses to companies
  email:
    provider: ms365    # ms365 | google
    enabled: true
    ghost_detection: true
  calendar:
    provider: ms365
    enabled: true

# Storage
storage:
  primary: minotes     # minotes | btask-cms | json
  minotes_dir: "CRM/"
  cms_store: "/root/projects/BTask/packages/cms/store"

# Coaching (when you have reps)
coaching:
  enabled: false
  benchmarks:
    talk_ratio: [0.40, 0.55]
    questions_per_call: [11, 14]
    max_monologue_seconds: 90
    meddic_target: 14

# PMF tracking
pmf:
  enabled: true
  ellis_threshold: 0.40
  min_companies: 10
  segment_by: [company_size, role, use_case]
```

---

## 10. Integration Points

### With DeepScript

```python
# After DeepScript analyzes a call:
analysis = deepscript.analyze("call.json")

# GTMScript links it to a company and updates state
gtmscript.link(analysis, company="Acme Corp")
# → Updates pain map
# → Updates PMF score
# → Updates buying/risk signals
# → Checks if stage should advance
# → Generates follow-up draft
```

### With AudioScript

```yaml
# .audioscript.yaml
sync:
  deepscript:
    enabled: true
  gtmscript:
    enabled: true
    auto_link: true  # Try to match transcript to company by speaker/calendar
```

### With ms365-cli / gwscli

```bash
# Email monitoring
gtmscript email-sync  # Fetch recent emails, match to companies, detect ghosts

# Calendar context
gtmscript calendar-sync  # Match upcoming meetings to companies
```

### With BFlow

GTMScript registers as a BFlow skill:
```yaml
---
name: gtmscript
description: "GTM pipeline tracking, deal intelligence, PMF dashboard, ghost detection"
metadata:
  bflow:
    emoji: "🎯"
    requires:
      bins: ["gtmscript"]
---
```

---

## 11. Cross-Project Architecture

```
/root/projects/
├── audioscript/         # Audio → Text
├── deepscript/          # Text → Intelligence
├── gtmscript/           # Intelligence → Pipeline + PMF  [NEW]
│   ├── core/
│   │   ├── company.py         # Company entity + CRUD
│   │   ├── deal.py            # Deal tracking + health
│   │   ├── pipeline.py        # Stage management
│   │   ├── icp.py             # ICP definition + scoring
│   │   └── contact.py         # Contact management
│   ├── intelligence/
│   │   ├── pain_map.py        # Cross-call pain aggregation
│   │   ├── pmf_tracker.py     # Per-company + cross-company PMF
│   │   ├── deal_health.py     # Deal health scoring
│   │   ├── ghost_detector.py  # Email silence detection
│   │   └── coaching.py        # Rep metrics + suggestions
│   ├── integrations/
│   │   ├── deepscript.py      # Consume DeepScript analyses
│   │   ├── email.py           # ms365/gws email intelligence
│   │   ├── calendar.py        # Meeting context
│   │   └── minotes.py         # MiNotes page generation
│   ├── cli/
│   │   └── main.py            # CLI commands
│   └── config.py
│
├── BTask/
│   └── packages/
│       ├── cms/              # Episodic memory
│       └── bflow/            # Orchestration
│
├── ms365-cli/               # Microsoft 365
├── gwscli/                  # Google Workspace
└── MiNotes/                 # Knowledge base (UI layer)
```

---

## 12. Why This Beats HubSpot for Pre-PMF

| Dimension | HubSpot | GTMScript |
|-----------|---------|-----------|
| **Cost** | $45-800/mo/user | Free (self-hosted) |
| **PMF tracking** | None | 8-dimension Ellis scoring, Vohra segmentation |
| **Call intelligence** | Basic (requires Gong add-on) | Full DeepScript analysis built-in |
| **Discovery quality** | None | Mom Test scoring, JTBD extraction |
| **Data ownership** | Their cloud | Your machine |
| **AI/LLM** | Limited | 7 providers, local models, benchmarked |
| **Customization** | Config-based | Code-level, open source |
| **Pipeline** | Feature-rich (overkill) | Minimal (right-sized for discovery) |
| **Relationship analysis** | None | Gottman/NVC (for personal calls too) |
| **Setup time** | Days | Minutes |

**When to switch to HubSpot:** When you have 3+ reps, need email sequences, and are past PMF. GTMScript exports to HubSpot when you outgrow it.
