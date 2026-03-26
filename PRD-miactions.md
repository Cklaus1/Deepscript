# MiActions — Universal Task Intelligence Engine

**Version:** 0.1 (Concept)
**Date:** 2026-03-26
**Status:** Proposal
**Relationship:** Separate package. DeepScript extracts action items → MiActions tracks, deduplicates, prioritizes, and assists completion.

---

## 1. Why a Separate Tool

| Concern | DeepScript | GTMScript | MiActions |
|---------|-----------|-----------|-----------|
| Input | Transcripts | DeepScript analyses | Action items from any source |
| Output | Per-call intelligence | Pipeline + deals + PMF | Task lifecycle + accountability |
| State | Stateless | Company/deal state | Task state across people + time |
| Core value | "What was said?" | "Will this deal close?" | "What needs to happen, by whom, by when?" |

```
AudioScript → DeepScript → MiActions (tasks)
                         → GTMScript (deals)
```

MiActions is the **task layer** that sits alongside GTMScript, not inside it. A task from a customer call goes to both:
- GTMScript: "Acme Corp needs a proposal" (deal context)
- MiActions: "Send proposal to Sarah by Friday" (task execution)

```
pip install miactions
```

---

## 2. Task Model

```yaml
task:
  id: "task_abc123"

  # What
  text: "Send trust reformation legal opinion to Chris"
  category: "document"  # follow-up | intro | document | decision | research | meeting | recurring

  # Who
  suggested_by: "Adam"           # Who said it on the call
  suggested_by_cluster: "spk_d7b4714c"
  owner: "Graham"                # Who's responsible
  owner_cluster: "spk_22f3f702"
  delegated_to: null             # If re-assigned

  # When
  created_at: "2026-03-15"
  due_date: "2026-03-22"         # Explicit or inferred
  due_source: "inferred"         # explicit ("by Friday") | inferred ("next week") | none
  overdue: true
  overdue_days: 4

  # Priority
  priority: "high"               # critical | high | medium | low
  priority_source: "context"     # explicit ("urgent") | context (legal/compliance) | frequency (mentioned 3x)
  commitment_strength: 0.9       # 0-1: "I will" (0.9) vs "we should" (0.3) vs "maybe" (0.1)

  # Status
  status: "open"                 # open | in-progress | blocked | done | verified | cancelled
  blocked_by: null               # Another task or person
  blocked_reason: null

  # Source
  source_call: "Recording (145)"
  source_episode: "ep_abc123"
  source_timestamp: 1245.0       # Seconds into the call
  source_quote: "Graham, can you get your legal opinion on whether judicial reformation alone is sufficient?"

  # Recurrence
  recurring: false
  recurrence_pattern: null       # "quarterly" | "monthly" | "weekly" | null
  next_occurrence: null

  # Deduplication
  duplicate_of: null             # Parent task if this is a duplicate
  reinforced_in: ["Recording (150)", "Recording (158)"]  # Calls where this was mentioned again
  reinforcement_count: 2         # Times mentioned across calls

  # Context
  company: null                  # GTMScript company link (if applicable)
  related_tasks: []              # Other tasks that depend on or relate to this
  tags: ["legal", "trust", "estate"]

  # Resolution
  completed_at: null
  completed_evidence: null       # "Discussed on call Recording (160), Graham confirmed sent"
  verified: false                # Was completion confirmed?
```

---

## 3. Task Extraction Pipeline

### From DeepScript Analysis

DeepScript already extracts action items per call. MiActions enriches them:

```
DeepScript output:
  {"text": "Send trust docs to Chris", "assignee": null, "speaker": "Adam", "deadline": null}

MiActions enrichment:
  1. Owner resolution: "Adam" said it → but it's assigned TO Graham (from context)
  2. Due date inference: Call was about "before the next meeting" → next meeting is Thursday → due Thursday
  3. Priority: Legal/compliance context → high priority
  4. Commitment scoring: "Can you get your legal opinion" = direct request (0.85)
  5. Category: "legal opinion" + "send" = document
  6. Deduplication: Similar task exists from 2 calls ago → link as reinforcement
```

### From Email (via ms365-cli)

```bash
# Scan emails for action items
miactions scan-email --days 7

# Detects:
# - "I'll send the docs by EOD" → task: send docs, owner: me, due: today
# - "Can you review and sign?" → task: review + sign, owner: me, due: inferred
# - "Let's schedule a call next week" → task: schedule call, owner: me, due: next week
```

### From Calendar (via ms365-cli)

```bash
# Pre-meeting task collection
miactions prep "Adam"

# Shows:
# 3 open tasks involving Adam
# 2 overdue items from last meeting
# 1 blocked task waiting on Graham
```

### Manual

```bash
miactions add "Review Acme proposal" --owner me --due friday --priority high
```

---

## 4. Deduplication & Reinforcement

When the same task is mentioned across multiple calls, MiActions:

1. **Detects duplicates** — LLM compares new action item against existing open tasks
2. **Links as reinforcement** — doesn't create a new task, adds to `reinforced_in`
3. **Escalates priority** — task mentioned 3+ times without completion → auto-escalate
4. **Tracks frustration** — if the same request is made with increasing urgency → flag

```
Call 1 (Mar 15): "Graham, can you look into the trust reformation?"
  → task_001 created, owner: Graham, priority: medium

Call 2 (Mar 20): "Graham, did you get a chance to look at the reformation?"
  → task_001 reinforced (count: 2), priority stays medium

Call 3 (Mar 25): "Graham, we really need that legal opinion before the meeting"
  → task_001 reinforced (count: 3), priority auto-escalated to HIGH
  → notification: "Task mentioned 3x without completion"
```

### Completion Detection

MiActions listens for completion signals in subsequent calls:

```
Call 4 (Mar 28): "Graham sent over his opinion yesterday, looks good"
  → task_001 auto-marked as done
  → completed_evidence: "Mentioned as completed in Recording (165)"
  → verified: false (until owner confirms)
```

---

## 5. Commitment Scoring

Not all "action items" are real commitments. MiActions scores them:

| Language | Score | Classification |
|----------|-------|---------------|
| "I will send it by Friday" | 0.95 | Strong commitment |
| "I'll get that to you" | 0.85 | Commitment |
| "Let me look into that" | 0.70 | Soft commitment |
| "We should consider" | 0.40 | Suggestion |
| "It might be worth" | 0.20 | Idea |
| "Someone should probably" | 0.10 | Vague |

Only tasks with commitment score ≥ 0.60 are tracked as real tasks. Below that, they're logged as "ideas" — searchable but not tracked for completion.

---

## 6. Priority System

Priority is computed from multiple signals:

| Signal | Impact | Example |
|--------|--------|---------|
| Explicit urgency language | +2 | "urgent", "ASAP", "critical", "immediately" |
| Legal/compliance context | +2 | Tax deadlines, regulatory filings, legal opinions |
| Financial impact | +1 | "million dollar", "budget", "funding" |
| Deadline proximity | +1 | Due within 48 hours |
| Reinforcement count | +1 per | Mentioned 3+ times |
| Requester seniority | +1 | CEO/board request vs peer suggestion |
| Blocker for others | +1 | Other tasks depend on this |

Scores mapped: 0-1 = low, 2-3 = medium, 4-5 = high, 6+ = critical

---

## 7. Views & Dashboards

### My Tasks
```bash
miactions mine

# Open tasks assigned to me:
#
# CRITICAL (1):
#   ⚠️  File amended California returns [OVERDUE 12 days]
#       Owner: me | Due: Mar 14 | From: Adam (call Mar 10)
#       Reinforced 3x | Blocked by: waiting on AutoCurdman docs
#
# HIGH (3):
#   📋  Send trust reformation opinion to Chris
#       Owner: me → delegated to Graham | Due: Mar 22 | From: Adam
#   📋  Update powers of attorney
#       Owner: me | Due: none | From: Brandon (call Mar 15)
#   📋  Review Acme proposal and send feedback
#       Owner: me | Due: Friday | Manual entry
#
# MEDIUM (5):
#   ...
```

### Person View
```bash
miactions person "Adam"

# Open tasks involving Adam (12):
#
# Adam owes me (4):
#   - Run Fusion investment reports [OVERDUE 5 days]
#   - Send SMA details to group
#   - Schedule Sarah Alfred Goldman call
#   - Confirm Justin's January presentation
#
# I owe Adam (3):
#   - Review bond allocation
#   - Provide full investment list (SAFEs vs equity)
#   - Approve quarterly billing
#
# Adam delegated to others (5):
#   - Laney: Forward tax-inclusive volume data
#   - Graham: Trust reformation legal opinion
#   - Justin: Prepare year-end presentation
#   - Elizabeth: Send Menlo Ventures memo
#   - Steven: Check California refund status
```

### Meeting Prep
```bash
miactions prep "Adam"

# Before your next call with Adam:
#
# ⚠️ OVERDUE (2):
#   - Run Fusion investment reports (was due Mar 20)
#   - Approve quarterly billing (was due Mar 15)
#
# 📋 OPEN (3):
#   - Review bond allocation
#   - Provide investment list
#   - Discuss trust reformation (Graham sent opinion)
#
# ✅ RECENTLY COMPLETED (2):
#   - Updated powers of attorney (done Mar 22)
#   - Sent escrow hold notice (done Mar 20)
#
# 💡 DISCUSSION TOPICS (from recent calls):
#   - QSBS tax strategy follow-up
#   - California amended returns status
#   - New R&D expensing rules impact
```

### Weekly Digest
```bash
miactions digest

# Weekly Task Digest — Mar 20-26
#
# Summary:
#   Completed: 8 | New: 12 | Overdue: 5 | Blocked: 2
#
# 🎯 Completed This Week:
#   ✅ Updated powers of attorney (Adam, Mar 22)
#   ✅ Sent escrow hold notice (Lori, Mar 20)
#   ✅ Filed entity registrations (Laney, Mar 21)
#   ...
#
# ⚠️ Overdue:
#   - File amended CA returns (12 days overdue, owner: me)
#   - Run Fusion reports (5 days, owner: Adam)
#   ...
#
# 📋 Due This Week:
#   - Review Acme proposal (Friday, owner: me)
#   - Schedule Goldman AI call (Thursday, owner: Elizabeth)
#   ...
#
# 🔮 Coming Up:
#   - Quarterly portfolio review (recurring, next: Apr 1)
#   - Tax filing deadline (Apr 15)
```

### Overdue Report
```bash
miactions overdue

# Overdue Tasks (5):
#
# | Task | Owner | Due | Overdue | Mentions | Priority |
# |------|-------|-----|---------|----------|----------|
# | File amended CA returns | me | Mar 14 | 12 days | 3x | CRITICAL |
# | Run Fusion reports | Adam | Mar 20 | 5 days | 2x | HIGH |
# | Send trust docs | Graham | Mar 22 | 4 days | 3x | HIGH |
# | Forward volume data | Laney | Mar 18 | 8 days | 1x | MEDIUM |
# | Check CA refund | Steven | Mar 15 | 11 days | 1x | MEDIUM |
```

---

## 8. AI-Assisted Actions

### Follow-Up Email Drafts

```bash
miactions follow-up "Send trust reformation opinion"

# Generates:
# To: Chris Klaus
# Subject: Trust Reformation — Graham's Legal Opinion
#
# Hi Chris,
#
# Following up from our call on the 15th — Graham completed his
# legal opinion on whether judicial reformation alone is sufficient
# for the trust restructuring.
#
# Key finding: [extracted from call analysis if available]
#
# Let me know if you'd like to discuss before Thursday's meeting.
#
# Best,
# [sender]
```

### Intro Email Drafts

```bash
miactions intro "Connect Adam with Kim"

# Generates:
# To: Adam Fuller, Kim Anderson
# Subject: Introduction — Adam Fuller × Kim Anderson
#
# Hi Adam, Kim,
#
# Wanted to connect you two. Adam is our wealth manager at Schwartz
# and Kim handles estate planning at Anderson Law.
#
# We discussed on our last call that it would be helpful for you
# to coordinate on the trust reformation timeline.
#
# I'll let you take it from here.
#
# Chris
```

### Meeting Agenda Generation

```bash
miactions agenda "Adam"

# Generates agenda from open tasks + recent topics:
#
# ## Meeting with Adam — Suggested Agenda
#
# 1. **Overdue Items** (5 min)
#    - Fusion investment reports (due Mar 20)
#    - Quarterly billing approval (due Mar 15)
#
# 2. **Trust Reformation Update** (10 min)
#    - Graham's legal opinion received
#    - Next steps on judicial vs non-judicial path
#
# 3. **Tax Strategy** (10 min)
#    - California amended returns status
#    - New R&D expensing rules (from Steven's memo)
#
# 4. **Portfolio Review** (10 min)
#    - Q1 performance
#    - Private credit allocation
#
# 5. **Action Items Review** (5 min)
#    - Assign owners and deadlines
```

---

## 9. CLI Commands

```bash
# Task management
miactions mine                              # My open tasks
miactions all                               # All tasks across all people
miactions add "Task text" --owner me --due friday --priority high
miactions done "task_abc123"                # Mark complete
miactions done "Send trust docs"            # Fuzzy match by text
miactions block "task_abc123" --reason "Waiting on Graham"

# Person view
miactions person "Adam"                     # All tasks involving Adam
miactions person "Adam" --overdue           # Just overdue
miactions person "Adam" --owe-me            # What Adam owes me
miactions person "Adam" --i-owe             # What I owe Adam

# Meeting support
miactions prep "Adam"                       # Pre-meeting task review
miactions agenda "Adam"                     # Generate meeting agenda
miactions debrief "Recording (165).json"    # Post-meeting: update tasks from call

# AI actions
miactions follow-up "task text or ID"       # Draft follow-up email
miactions intro "Connect Adam with Kim"     # Draft intro email
miactions remind "Adam" --tasks overdue     # Draft reminder to Adam

# Digests
miactions digest                            # Weekly summary
miactions digest --daily                    # Daily summary
miactions overdue                           # Overdue report

# Import from DeepScript
miactions import ./analysis-output/         # Import action items from analyses
miactions import --call "Recording (145)"   # Import from specific call
miactions scan-email --days 7               # Extract tasks from emails

# Sync
miactions sync                              # Sync task status with MiNotes pages
```

---

## 10. Data Storage

### MiNotes (Source of Truth)

Each task = a MiNotes page or a section within a contact/company page.

**Option A: Inline on contact pages** (recommended for <500 tasks)
```markdown
# Adam Fuller

## Open Tasks (4)
- [ ] Run Fusion investment reports [OVERDUE] #high
- [ ] Send SMA details to group #medium
- [ ] Schedule Sarah Alfred Goldman call #medium
- [ ] Confirm Justin January presentation #low

## Completed Tasks
- [x] Updated powers of attorney (Mar 22)
- [x] Sent escrow hold notice (Mar 20)
```

**Option B: Separate task pages** (for >500 tasks)
```markdown
---
type: task
id: task_abc123
text: "Send trust reformation legal opinion"
owner: Graham
suggested_by: Adam
due: 2026-03-22
priority: high
status: open
commitment: 0.85
source_call: "Recording (145)"
reinforced: 2
company: null
tags: [legal, trust, estate]
---

# Send trust reformation legal opinion

**Owner:** [[Graham]] | **Due:** Mar 22 | **Priority:** HIGH
**Requested by:** [[Adam]] on [[Recording (145)]]
**Reinforced in:** [[Recording (150)]], [[Recording (158)]]

## Context
Graham was asked to provide his legal opinion on whether judicial
reformation alone is sufficient for the trust restructuring...

## Related Tasks
- [[Update powers of attorney]]
- [[File amended California returns]]
```

### BTask CMS (Index)

```
store/
├── tasks/
│   ├── open.jsonl           # Active tasks (append-only log)
│   ├── completed.jsonl      # Done tasks (archive)
│   └── index.json           # Fast lookup by person, priority, due date
└── analytics/
    └── task_velocity.json   # Completion rate, avg days to close
```

---

## 11. Integration Points

### With DeepScript
```python
# After analysis, extract and import tasks
analysis = deepscript.analyze("call.json")
miactions.import_from_analysis(analysis, call_id="Recording (145)")
# → Creates tasks, deduplicates, assigns owners, infers dates
```

### With GTMScript
```python
# Tasks linked to deals
miactions.link_to_company("task_abc123", company="Acme Corp")
# → Task appears on Acme Corp's company page AND owner's task list
```

### With AudioScript
```yaml
# .audioscript.yaml
sync:
  deepscript:
    enabled: true
  miactions:
    enabled: true
    auto_import: true  # Import tasks from every new analysis
```

### With ms365-cli
```bash
# Email task extraction
miactions scan-email --days 7

# Follow-up sending
miactions follow-up "task_abc123" --send  # Draft + send via ms365

# Calendar-based due dates
miactions infer-dates --calendar ms365  # "Before Thursday's meeting" → actual date
```

### With MiNotes
```bash
# Sync tasks to/from MiNotes pages
miactions sync

# Tasks appear as checkboxes on contact pages
# Checking a box in MiNotes → miactions marks it done on next sync
```

---

## 12. Config (.miactions.yaml)

```yaml
# Task extraction
extraction:
  min_commitment: 0.60       # Only track tasks above this commitment score
  auto_import: true           # Auto-import from DeepScript analyses
  scan_email: false           # Auto-scan emails for tasks

# Deduplication
dedup:
  enabled: true
  similarity_threshold: 0.80  # LLM similarity score for dedup
  auto_reinforce: true        # Auto-link duplicates as reinforcements

# Priority
priority:
  auto_escalate_after: 3      # Reinforce count before auto-escalating
  overdue_escalate: true       # Escalate priority when overdue

# Notifications
notifications:
  digest: "weekly"            # daily | weekly | none
  overdue_alert: true
  prep_before_meeting: true   # Show open tasks before scheduled calls

# Due date inference
due_dates:
  calendar_provider: ms365    # ms365 | google | none
  infer_from_context: true    # "by Friday", "next week", "before the meeting"
  default_due_days: 7         # If no date mentioned, default to 7 days

# Storage
storage:
  primary: minotes
  minotes_dir: "Tasks/"
  task_on_contact_pages: true  # Add task sections to contact pages
  cms_store: "/root/projects/BTask/packages/cms/store"

# AI assistance
ai:
  provider: nim               # Same as DeepScript
  draft_emails: true
  draft_intros: true
  generate_agendas: true
```

---

## 13. Architecture

```
/root/projects/
├── audioscript/         # Audio → Text
├── deepscript/          # Text → Intelligence (extracts action items)
├── gtmscript/           # Intelligence → Pipeline + PMF (deal tracking)
├── miactions/           # Intelligence → Tasks + Accountability  [NEW]
│   ├── core/
│   │   ├── task.py            # Task entity + CRUD
│   │   ├── extractor.py       # Extract tasks from DeepScript analyses
│   │   ├── deduplicator.py    # Cross-call dedup + reinforcement
│   │   ├── prioritizer.py     # Priority scoring
│   │   ├── scheduler.py       # Due date inference + overdue detection
│   │   └── completion.py      # Auto-detect completion from calls
│   ├── views/
│   │   ├── my_tasks.py        # Personal task view
│   │   ├── person_view.py     # Per-person task view
│   │   ├── digest.py          # Weekly/daily digest
│   │   ├── overdue.py         # Overdue report
│   │   └── prep.py            # Meeting prep from tasks
│   ├── ai/
│   │   ├── follow_up.py       # Draft follow-up emails
│   │   ├── intro.py           # Draft intro emails
│   │   ├── agenda.py          # Generate meeting agendas
│   │   └── remind.py          # Draft reminder messages
│   ├── integrations/
│   │   ├── deepscript.py      # Import from DeepScript analyses
│   │   ├── email.py           # Scan emails for tasks
│   │   ├── calendar.py        # Due date inference from calendar
│   │   ├── minotes.py         # Sync with MiNotes pages
│   │   └── gtmscript.py       # Link tasks to companies/deals
│   ├── cli/
│   │   └── main.py
│   └── config.py
│
├── BTask/packages/cms/  # Task index + analytics
├── ms365-cli/           # Email + calendar
└── MiNotes/             # Source of truth (UI)
```

---

## 14. Build Priority

### Phase 1 — Foundation (v0.1)
1. Task entity model + CRUD
2. Import from DeepScript analysis output (batch)
3. Deduplication (LLM similarity matching)
4. Owner resolution (match to speaker profiles)
5. Basic CLI (mine, all, add, done, person)
6. MiNotes task sections on contact pages

### Phase 2 — Intelligence (v0.2)
7. Commitment scoring (language analysis)
8. Priority computation (multi-signal)
9. Due date inference from context + calendar
10. Reinforcement tracking (same task across calls)
11. Completion detection from subsequent calls
12. Meeting prep view

### Phase 3 — AI Assistance (v0.3)
13. Follow-up email drafts
14. Intro email drafts
15. Meeting agenda generation
16. Weekly/daily digest
17. Email scanning for tasks

### Phase 4 — Automation (v1.0)
18. Auto-import from AudioScript → DeepScript → MiActions pipeline
19. Recurring task detection + auto-creation
20. Overdue escalation + notifications
21. Blocked task dependency tracking
22. GTMScript integration (tasks on company pages)

---

## 15. Why This Beats Todoist/Asana/Linear

| Dimension | Traditional Task Tools | MiActions |
|-----------|----------------------|-----------|
| **Task creation** | Manual entry | Auto-extracted from calls + emails |
| **Deduplication** | Manual | LLM-powered cross-call matching |
| **Owner detection** | You assign it | Inferred from who said what on the call |
| **Due dates** | You set them | Inferred from "by Friday", "before the meeting" |
| **Priority** | You choose | Computed from urgency, context, reinforcement count |
| **Completion** | You check the box | Auto-detected from subsequent calls |
| **Context** | Title only | Full call transcript, speaker, timestamp, quote |
| **Accountability** | None | Per-person view: "Adam owes me 4 things" |
| **Meeting prep** | Separate tool | "Here's everything open with Adam before your call" |
| **Follow-up** | You write the email | AI drafts it from task context + call history |

**When to switch to Linear/Asana:** When you have a team of 10+ and need sprint boards, epics, and project management. MiActions is for personal/small-team accountability from conversations, not software development project management.
