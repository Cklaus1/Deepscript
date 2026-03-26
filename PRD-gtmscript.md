# GTMScript — Go-To-Market Intelligence Engine

**Version:** 0.2 (Revised)
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
| Entity model | None (just transcripts) | Companies, Contacts, ICPs |
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
│ Transcribe│   (JSON)   │ Analyze      │  (JSON)  │ Update Pipeline   │
│          │             │ Score        │          │ Track PMF         │
└──────────┘             └──────────────┘          │ Generate Follow-up│
                                                   │ Ghost Detection   │
Email (ms365/gws)──────────────────────────────────│ Email Intel       │
Calendar (ms365/gws)───────────────────────────────│ Meeting Context   │
                                                   └───────────────────┘
                                                          ↓
                                                   MiNotes (source of truth)
                                                   BTask CMS (index/compute)
```

---

## 2. Data Storage: Source of Truth

**MiNotes is primary. CMS is index.**

Each company = a MiNotes page with YAML frontmatter. GTMScript reads and writes MiNotes pages AND maintains a JSON index in BTask CMS for fast queries. The index is derived from MiNotes, not the other way around.

```
MiNotes/CRM/                          BTask CMS store/
├── ICP Definition.md                  ├── gtm/
├── Targets/                           │   ├── index.json (derived from MiNotes)
│   ├── Acme Corp.md                   │   ├── pipeline_state.json
│   ├── Globex Inc.md                  │   └── analytics/
│   └── ...                            │       ├── pmf_dashboard.json
├── Pipeline Dashboard.md              │       └── competitive_intel.json
└── PMF Dashboard.md                   └── episodes/ (DeepScript analyses)
```

If there's a conflict, MiNotes wins. The CMS index is rebuilt from MiNotes pages on `gtmscript sync`.

---

## 3. Entity Model

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

### Company (Primary Entity — v1 merges Deal into Company)

```yaml
company:
  id: "comp_acme123"
  name: "Acme Corp"
  icp_match: "Series A SaaS Ops Teams"
  icp_score: 0.85

  # Firmographics
  size: 50
  industry: "SaaS"
  website: "acme.com"
  location: "San Francisco"

  # Relationship state
  stage: "validation"  # cold → outreach → discovery → validation → pilot → negotiation → loi → deal → customer | dead
  stage_changed_at: "2026-03-20"
  first_contact: "2026-02-15"
  last_contact: "2026-03-25"

  # Dead tracking (only when stage=dead)
  dead_reason: null  # no_pain | bad_timing | competitor | budget | no_champion | ghosted | not_icp
  dead_notes: ""
  reactivate_date: null  # "2026-Q3" — when to check back

  # Referral tracking
  referred_by: "Jane Advisor"
  referral_source: "advisor"  # advisor | investor | customer | conference | inbound | cold

  # Revenue estimate
  estimated_arr: 12000  # Rough: size × pricing tier, updated as pricing discussed
  deal_probability: 0.45  # Computed from signals
  deal_timeline: "Q2 2026"
  deal_decision_process: "VP Ops recommends → CEO approves"

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

  # Outreach tracking
  outreach_attempts:
    - date: "2026-02-15"
      channel: "linkedin"
      message_template: "pain-focused-v2"
      response: false
    - date: "2026-02-22"
      channel: "email"
      message_template: "case-study-share"
      response: true
      days_to_response: 7

  # Intelligence (auto-populated from DeepScript)
  pain_map:
    - pain: "Spreadsheet-based reporting breaks every Monday"
      severity: "high"
      frequency: "weekly"
      source: "call_2026-03-15"
      confirmed_in: ["call_2026-03-20", "call_2026-03-25"]
      workaround: "Manual copy-paste from 3 tools"

    - pain: "New hires take 3 weeks to learn the reporting process"
      severity: "medium"
      frequency: "per hire"
      source: "call_2026-03-20"

  pmf_signals:
    current_score: 6.8
    ellis_classification: "somewhat_disappointed"
    trend: [4.2, 5.5, 6.8]
    strongest: "workflow integration"
    blocker: "still using spreadsheets for deep analysis"

  buying_signals:
    - signal: "Asked about enterprise pricing unprompted"
      call: "call_2026-03-25"
      strength: "strong"
    - signal: "Mentioned Q2 budget cycle"
      call: "call_2026-03-20"
      strength: "moderate"

  risk_signals:
    - signal: "CEO hasn't been on a call yet"
      severity: "high"
      type: "missing_stakeholder"
    - signal: "Evaluating Asana and Monday.com"
      severity: "medium"
      type: "competitor"

  competitors: ["Asana", "Monday.com"]

  # MEDDIC (aggregated across all calls)
  meddic:
    metrics: 2        # "Save 20 hours/week"
    economic_buyer: 1  # CEO identified but not engaged
    decision_criteria: 2
    decision_process: 2
    identify_pain: 3
    champion: 3
    total: 13
    max: 18
    gaps: ["Economic buyer not engaged"]

  next_step: "Schedule demo with CEO"
  next_step_date: "2026-03-28"
  next_step_owner: "us"  # us | them

  # Calls (linked from DeepScript)
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
      decision_maker_engaged: false
      pain_confirmed: true
      timeline_exists: true
      budget_discussed: true
      competitors_active: true
      multi_threaded: false
      next_step_specific: true
    risks: ["No CEO engagement", "Competitor evaluation active"]
    recommendation: "Get CEO on next call. Prepare competitive battlecard for Asana."
```

**Note:** No separate Deal entity in v1. Company IS the deal. Split Deal out in v2 when you have enterprise accounts with multiple opportunities.

---

## 4. Pipeline Stages

```
cold              Not yet contacted. In target list.
outreach          Contacted (email/LinkedIn). Waiting for response.
discovery         Had first conversation. Exploring pain.
validation        Pain confirmed. Testing product fit.
pilot             Using product (trial/POC).
negotiation       Pricing/terms discussion.
loi               Letter of intent or verbal commit.
deal              Signed. Customer.
dead              Not a fit. Documented why (dead_reason required).
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
| any → dead | No response 30 days, explicit "not interested", or manual | Ghost detection / DeepScript / manual |

**Dead requires a reason.** `gtmscript company move "Acme" --stage dead --reason competitor --notes "Chose Asana, liked their Jira integration"`

---

## 5. Auto-Linking (How Analyses Connect to Companies)

When DeepScript produces an analysis, GTMScript needs to know which company it belongs to. Matching priority:

1. **Explicit flag** (highest confidence):
   ```bash
   deepscript analyze call.json --company "Acme Corp"
   ```

2. **Speaker cluster ID** — If AudioScript's speaker DB maps `spk_381a53e4` to a contact, and that contact belongs to Acme Corp, auto-link.

3. **Calendar event** — If the call timestamp matches a calendar meeting titled "Acme Demo" (±30 min window), link to Acme.

4. **Email proximity** — If there's an email thread with an Acme contact within 24 hours of the call, likely related.

5. **Prompt user** — If no match found:
   ```
   Could not auto-link call_2026-03-25.json
   Speakers: spk_381a53e4, spk_58cddeaf
   Time: 2026-03-25 15:00 (50 min)

   Possible matches:
     1. Acme Corp (Sarah Chen spoke 3 days ago)
     2. Globex Inc (meeting scheduled today)
     3. [New company]
     4. [Skip]
   ```

---

## 6. Cross-Call Intelligence

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

### Competitive Intelligence (Aggregated Across All Companies)

```markdown
## Competitive Landscape — From 50 Active Deals

| Competitor | Mentioned In | Win Rate | Top Objection | Best Counter |
|-----------|-------------|----------|---------------|-------------|
| Asana | 12/50 (24%) | 40% | "Cheaper" | Automation engine demo |
| Monday.com | 8/50 (16%) | 25% | "Brand recognition" | Migration support offer |
| Spreadsheets | 30/50 (60%) | 80% | "Free" | "20 hrs/week saved" ROI calc |

### Trends
- Asana mentions ↑ 30% this quarter
- Monday.com mentions stable
- "Build in-house" down 50% (good — they're buying not building)
```

---

## 7. Email Intelligence

### Ghost Detection

```yaml
ghost_detection:
  warning_days: 7    # Yellow flag after 7 days no response
  critical_days: 14  # Red flag after 14 days
  dead_days: 30      # Auto-move to dead after 30 days (with reason=ghosted)
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
gtmscript email-intel "Acme Corp"
```

Extracts:
- Commitment language in emails vs calls (are they stronger/weaker in writing?)
- Response time patterns (getting slower = losing interest)
- CC/forwarding patterns (are they multi-threading internally? good sign)
- Attachment patterns (sent a proposal = high intent)

### Outreach Analytics

```bash
gtmscript outreach-stats
```

```markdown
## Outreach Performance

| Channel | Sent | Replied | Rate | Avg Days to Reply |
|---------|------|---------|------|-------------------|
| LinkedIn | 120 | 24 | 20% | 5.2 days |
| Cold email | 200 | 18 | 9% | 8.1 days |
| Warm intro | 30 | 22 | 73% | 1.4 days |
| Conference | 15 | 9 | 60% | 3.0 days |

Best performing template: "pain-focused-v2" (28% reply rate)
Worst performing: "product-demo-invite" (4% reply rate)
```

---

## 8. CLI Commands

```bash
# Company management
gtmscript company add "Acme Corp" --icp "Series A SaaS" --size 50 --industry SaaS
gtmscript company add "Acme Corp" --referred-by "Jane Advisor" --source advisor
gtmscript company list --stage discovery
gtmscript company view "Acme Corp"
gtmscript company move "Acme Corp" --stage validation
gtmscript company move "Acme Corp" --stage dead --reason competitor --notes "Chose Asana"

# Link DeepScript analysis to company
gtmscript link call.analysis.json --company "Acme Corp"

# Auto-link (match by speaker, calendar, or email — prompts on ambiguity)
gtmscript auto-link ./analyses/

# Pipeline
gtmscript pipeline                    # Kanban view
gtmscript pipeline --stage discovery  # Filter
gtmscript pipeline --health red       # At-risk deals

# PMF Dashboard
gtmscript pmf                         # Cross-company PMF
gtmscript pmf --segment "20-50 employees"

# Competitive intelligence
gtmscript competitors                  # Cross-deal competitive view
gtmscript competitors asana            # Asana-specific win/loss

# Email intelligence
gtmscript email-intel "Acme Corp"     # Analyze email thread
gtmscript ghosts                       # Show ghost prospects
gtmscript ghosts --re-engage          # Draft re-engagement emails
gtmscript outreach-stats               # Channel + template performance

# ICP management
gtmscript icp define --name "Series A SaaS"
gtmscript icp import targets.csv      # Bulk import (CSV → MiNotes pages)
gtmscript icp score                    # Score companies against ICP

# Deal intelligence
gtmscript deal "Acme Corp"            # Health + MEDDIC + timeline
gtmscript deal risks                  # All at-risk deals
gtmscript deal forecast               # Weighted pipeline forecast

# Follow-up generation
gtmscript follow-up "Acme Corp"       # Draft follow-up email from last call
gtmscript prep "Acme Corp"            # Call prep from all intelligence

# Sync
gtmscript sync                        # Rebuild CMS index from MiNotes pages

# Export (when you outgrow GTMScript)
gtmscript export hubspot --format csv  # Companies → Accounts, Contacts, Deals
gtmscript export salesforce --format csv
```

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
  auto_advance: true
  ghost_days: 30
  require_dead_reason: true

# Data sources
sources:
  deepscript:
    enabled: true
    auto_link: true
    link_priority: [explicit, speaker_id, calendar, email_proximity, prompt]
  email:
    provider: ms365    # ms365 | google
    enabled: true
    ghost_detection: true
  calendar:
    provider: ms365
    enabled: true

# Storage
storage:
  primary: minotes
  minotes_dir: "CRM/"
  cms_store: "/root/projects/BTask/packages/cms/store"
  sync_on_change: true

# PMF tracking
pmf:
  enabled: true
  ellis_threshold: 0.40
  min_companies: 10
  segment_by: [company_size, role, use_case]

# Revenue
revenue:
  default_pricing_per_seat: 25  # $/user/month for ARR estimation
  currency: "USD"
```

---

## 10. Integration Points

### With DeepScript

```python
# After DeepScript analyzes a call:
analysis = deepscript.analyze("call.json")

# GTMScript links it to a company and updates state
gtmscript.link(analysis, company="Acme Corp")
# → Updates pain map (dedup + rank)
# → Updates PMF score (trend)
# → Updates buying/risk signals
# → Updates MEDDIC scores (aggregated)
# → Checks if stage should advance
# → Updates MiNotes page
# → Rebuilds CMS index
```

### With AudioScript

```yaml
# .audioscript.yaml
sync:
  deepscript:
    enabled: true
  gtmscript:
    enabled: true
    auto_link: true
```

### With ms365-cli / gwscli

```bash
gtmscript email-sync   # Fetch recent emails, match to companies, detect ghosts
gtmscript calendar-sync # Match upcoming meetings to companies, generate prep
```

### With BFlow

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
│   │   ├── pipeline.py        # Stage management + auto-advance
│   │   ├── icp.py             # ICP definition + scoring + import
│   │   └── contact.py         # Contact management
│   ├── intelligence/
│   │   ├── pain_map.py        # Cross-call pain aggregation + dedup
│   │   ├── pmf_tracker.py     # Per-company + cross-company PMF
│   │   ├── deal_health.py     # Health scoring + MEDDIC aggregation
│   │   ├── ghost_detector.py  # Email silence detection
│   │   ├── competitive.py     # Cross-deal competitive intelligence
│   │   └── outreach_stats.py  # Channel + template performance
│   ├── integrations/
│   │   ├── deepscript.py      # Consume DeepScript analyses + auto-link
│   │   ├── email.py           # ms365/gws email intelligence
│   │   ├── calendar.py        # Meeting context
│   │   ├── minotes.py         # MiNotes page read/write
│   │   └── export.py          # HubSpot/Salesforce export
│   ├── generators/
│   │   ├── follow_up.py       # Draft follow-up emails
│   │   ├── call_prep.py       # Call prep from all intelligence
│   │   └── re_engage.py       # Ghost re-engagement drafts
│   ├── cli/
│   │   └── main.py
│   └── config.py
│
├── BTask/
│   └── packages/
│       ├── cms/              # Index + compute
│       └── bflow/            # Orchestration
│
├── ms365-cli/               # Microsoft 365
├── gwscli/                  # Google Workspace
└── MiNotes/                 # Source of truth (UI layer)
```

---

## 12. Build Priority

### Phase 1 — Foundation (v0.1)
1. Company CRUD + MiNotes page generation
2. ICP definition + CSV import (bulk create 1000 company pages)
3. Pipeline stages + manual stage movement
4. DeepScript linking (manual `--company` flag)
5. Ghost detection (email last-reply tracking)

### Phase 2 — Intelligence (v0.2)
6. Auto-linking (speaker ID → calendar → email proximity → prompt)
7. Pain map aggregation (cross-call dedup + ranking)
8. PMF dashboard (cross-company Ellis + Vohra segmentation)
9. Deal health scoring + MEDDIC aggregation
10. Follow-up email generation

### Phase 3 — Scale (v0.3)
11. Competitive intelligence aggregation
12. Outreach analytics (channel + template performance)
13. Call prep generation
14. CRM export (HubSpot/Salesforce CSV)
15. BFlow skill + MCP server

### Phase 4 — Team (v1.0, post-hire)
16. Rep coaching (talk metrics, methodology compliance, benchmarks)
17. Deal forecasting (weighted pipeline)
18. Separate Deal entity (multi-deal per company)
19. Team dashboards

---

## 13. Why This Beats HubSpot for Pre-PMF

| Dimension | HubSpot | GTMScript |
|-----------|---------|-----------|
| **Cost** | $45-800/mo/user | Free (self-hosted) |
| **PMF tracking** | None | 8-dimension Ellis scoring, Vohra segmentation |
| **Call intelligence** | Basic (requires Gong add-on) | Full DeepScript analysis built-in |
| **Discovery quality** | None | Mom Test scoring, JTBD extraction |
| **Pain aggregation** | Manual notes | Auto-aggregated cross-call pain map |
| **Ghost detection** | Manual | Automatic with re-engagement drafts |
| **Competitive intel** | Manual | Auto-aggregated from all calls |
| **Data ownership** | Their cloud | Your machine, MiNotes pages |
| **AI/LLM** | Limited | 7 providers, local models, benchmarked |
| **Dead post-mortem** | Optional field | Required reason + learnings |
| **Outreach tracking** | Built-in sequences | Channel + template analytics |
| **Setup time** | Days | Minutes |

**When to switch to HubSpot:** When you have 3+ reps, need email sequences at scale, and are past PMF. `gtmscript export hubspot` gets you there.
