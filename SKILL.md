---
name: deepscript
description: "Transcript intelligence — analyze calls for insights, action items, PMF scoring, relationship health, sales methodology. Use when user asks to analyze a transcript, meeting recording, or call."
metadata:
  bflow:
    emoji: "🔬"
    mode: "document"
    requires:
      bins: ["deepscript"]
      env: []
---

# DeepScript — Transcript Intelligence

Analyze call transcripts for structured insights. Works with any transcript source (AudioScript, Zoom, Otter, manual).

## When to Use

- "Analyze this call recording", "What were the action items from that meeting?"
- "Score this sales call", "How did the discovery call go?"
- "Check our PMF signals from customer calls"
- "Score this interview candidate"
- After AudioScript transcription completes

## Quick Reference

| Operation | Command |
|-----------|---------|
| Analyze transcript | `deepscript analyze transcript.json` |
| Force call type | `deepscript analyze transcript.json --type sales-call` |
| Sales with MEDDIC | `deepscript analyze transcript.json -t sales-call` |
| Discovery (Mom Test) | `deepscript analyze transcript.json -t discovery-call` |
| PMF scoring | `deepscript analyze transcript.json -t pmf-call` |
| Interview scoring | `deepscript analyze transcript.json -t interview-behavioral` |
| Support analysis | `deepscript analyze transcript.json -t support-escalation` |
| QBR analysis | `deepscript analyze transcript.json -t qbr` |
| Relationship insights | `deepscript analyze transcript.json -t family --relationship-insights` |
| Classify only | `deepscript classify transcript.json` |
| Batch analyze | `deepscript analyze ./transcripts/ --recursive` |
| Skip processed | `deepscript analyze ./transcripts/ -r --new-only` |
| Write CMS episode | `deepscript analyze transcript.json --cms` |
| Calendar enrichment | `deepscript analyze transcript.json --calendar` |
| Markdown output | `deepscript -o markdown analyze transcript.json` |
| JSON output | `deepscript -o json analyze transcript.json` |

## Supported Call Types

| Type | Analyzer | Key Sections |
|------|----------|-------------|
| `business-meeting` | BusinessAnalyzer | Summary, actions, decisions, questions |
| `sales-call` | SalesAnalyzer | MEDDIC/BANT/SPIN score, buying/risk signals |
| `discovery-call` | DiscoveryAnalyzer | Mom Test, JTBD, pain points, commitments |
| `pmf-call` | PMFAnalyzer | 8-dimension PMF score, Ellis classification |
| `interview-behavioral` | InterviewAnalyzer | STAR completeness, competency mapping |
| `interview-technical` | InterviewAnalyzer | Problem-solving, communication clarity |
| `support-escalation` | SupportAnalyzer | Issue type, emotion arc, resolution, empathy |
| `qbr` | QBRAnalyzer | Health score, expansion signals, churn risk |
| `family`/`partner` | RelationshipAnalyzer | Gottman, NVC, appreciation ratio (opt-in) |

## Execution

```python
execute_skill("deepscript", "analyze transcript.json -o json")
```

## Global Flags

- `-o json|markdown|table|yaml|quiet` — Output format
- `--type <call-type>` — Override auto-classification
- `--no-llm` — Rule-based only (no API calls)
- `--cms` — Write CMS episode after analysis
- `--calendar` — Enrich with calendar context
- `--relationship-insights` — Enable relationship analysis (opt-in)
- `--recursive` — Process directories recursively
- `--new-only` — Skip already-analyzed files
- `--dry-run` — Preview without processing
- `--output-dir <dir>` — Save output files
