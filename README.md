# DeepScript

**Transcript Intelligence Engine** — classification, insights, and analysis for call transcripts.

DeepScript analyzes transcripts from any source (AudioScript, Zoom, Otter, manual) and produces structured intelligence: call classification, communication metrics, action items, sales methodology scoring, PMF analysis, relationship health, and more.

## Quick Start

```bash
pip install -e .
deepscript analyze transcript.json
```

With LLM (recommended):
```bash
export ANTHROPIC_API_KEY=your-key
deepscript analyze transcript.json
```

Without LLM (rule-based fallback):
```bash
deepscript analyze transcript.json --no-llm
```

## What It Does

DeepScript takes a transcript JSON file and produces:

- **Classification** — Auto-detects call type (sales, discovery, interview, meeting, etc.)
- **Communication Metrics** — Speaker talk ratios, question counts, speaking balance, monologue detection
- **Topic Segmentation** — Breaks long calls into named sections with timestamps
- **Action Items** — Extracts tasks with assignees and deadlines
- **Decisions** — Identifies what was decided and by whom
- **Type-Specific Analysis** — Sales methodology scoring (MEDDIC/BANT/SPIN), PMF 8-dimension scoring, interview STAR completeness, support emotion trajectory, and more

## Supported Call Types (50+)

| Tier | Types | Examples |
|------|-------|---------|
| **Tier 1** | Core business | `business-meeting`, `sales-call`, `discovery-call`, `support-escalation`, `qbr`, `interview-behavioral`, `interview-technical` |
| **Tier 2** | Founder-critical | `pmf-call`, `investor-pitch`, `cofounder-alignment`, `advisory-call`, `board-meeting`, `customer-onboarding`, `performance-review`, `sprint-retro`, `postmortem` |
| **Tier 3** | Specialized | `family`, `partner`, `podcast`, `therapy-session`, `coaching-session`, `voice-memo` |

## CLI Commands

```bash
# Core analysis
deepscript analyze transcript.json                    # Full analysis
deepscript analyze transcript.json -t sales-call      # Force call type
deepscript analyze transcript.json --no-llm           # Rule-based only
deepscript -o markdown analyze transcript.json        # Markdown output

# Batch processing
deepscript analyze ./transcripts/ -r                  # Process directory
deepscript analyze ./transcripts/ -r --new-only       # Skip already-analyzed
deepscript analyze ./transcripts/ -r --parallel       # Parallel processing

# Classification only
deepscript classify transcript.json

# CMS integration
deepscript analyze transcript.json --cms              # Write CMS episode
deepscript playbook sales-call                        # Generate playbook from episodes
deepscript dashboard pmf                              # PMF cross-call dashboard
deepscript prep sales-call                            # Call prep from patterns

# LLM usage tracking
deepscript usage                                      # Current month's usage
deepscript usage --all                                # All-time usage
deepscript usage --days 7                             # Last 7 days

# Model benchmarking
deepscript benchmark --list                           # List NIM models
deepscript benchmark -p nim --top 10                  # Benchmark top 10 NIM models
deepscript benchmark --history                        # Past benchmark runs
deepscript benchmark --trend "model-name"             # Quality trend with stddev
```

## LLM Providers

DeepScript supports 7 LLM providers. LLM-first classification is the default — falls back to rule-based keywords when no API key is available.

```yaml
# .deepscript.yaml
llm:
  provider: claude        # Default — best quality
  model: claude-sonnet-4-6

  # Or use local models:
  # provider: ollama
  # model: llama3.1

  # provider: vllm
  # model: meta-llama/Llama-3-8B
  # base_url: http://gpu-server:8000/v1

  # provider: nim
  # model: meta/llama-3.1-405b-instruct

  # provider: openai
  # model: gpt-4
```

| Provider | Auth | Local | Notes |
|----------|------|-------|-------|
| `claude` | `ANTHROPIC_API_KEY` | No | Default, best quality |
| `openai` | `OPENAI_API_KEY` | No | GPT-4 and variants |
| `ollama` | None | Yes | Local, no API key needed |
| `vllm` | None | Yes | Self-hosted GPU inference |
| `sglang` | None | Yes | Self-hosted GPU inference |
| `nim` | `NVIDIA_API_KEY` | No | NVIDIA cloud, 189 models |
| `none` | — | — | Rule-based only, no LLM |

## Input Format

DeepScript accepts any JSON transcript with `text` and/or `segments`:

```json
{
  "text": "Full transcript text...",
  "segments": [
    {
      "start": 0.0,
      "end": 5.0,
      "text": "Hello everyone",
      "speaker": "Alice"
    }
  ]
}
```

Works with AudioScript output (includes `diarization`, `speaker_cluster_id`, `metadata`), Zoom transcripts, Otter exports, or any JSON with text content.

## Configuration

Create `.deepscript.yaml` in your project directory:

```yaml
# Classification
classify: true

# Communication metrics
communication:
  enabled: true

# Topic segmentation
topics:
  enabled: true
  method: hybrid    # rule | llm | hybrid
  min_duration: 60

# LLM
llm:
  provider: claude
  model: claude-sonnet-4-6
  max_retries: 3
  budget_per_month: 50.00

# Output
output:
  format: markdown
  sections: all     # or: [summary, action_items, communication]
```

See [PRD-deepscript.md](PRD-deepscript.md) for the full configuration schema.

## Architecture

```
Transcript JSON → Speaker Enrichment → Classifier → Topic Segmenter
                                           ↓
                                    Type-Specific Analyzer
                                           ↓
                              JSON / Markdown Formatter → Output
                                           ↓
                                    CMS Episode (optional)
```

**Analyzers are auto-discovered** — adding a new call type is one file with a `supported_types` property. No registration code needed.

**LLM calls are consolidated** — combined analysis prompts produce summary + actions + decisions + type-specific sections in 1 API call instead of 3-5.

**Batch processing is parallel** — `asyncio.to_thread()` with configurable concurrency and rate limiting.

## Model Benchmarking

DeepScript includes a full model evaluation system for comparing LLM providers:

```bash
# Benchmark across all test transcripts with ground truth scoring
deepscript benchmark -p nim --top 5

# Results include:
# - Quality score (accuracy, grounding, hallucination detection)
# - Latency per model
# - Cost tracking
# - Per-transcript breakdown
# - Historical trends with standard deviation
```

Benchmark data accumulates in `~/.deepscript/benchmarks/` — run multiple times to compute stddev and track quality trends.

## Integration

### AudioScript
DeepScript is the analysis companion to [AudioScript](https://github.com/Cklaus1/audioscript). AudioScript transcribes audio → text; DeepScript analyzes text → intelligence.

```yaml
# .audioscript.yaml
sync:
  deepscript:
    enabled: true
    config: .deepscript.yaml
```

### BTask CMS
Analysis episodes are stored in BTask's episodic memory for playbook generation:

```bash
deepscript analyze transcript.json --cms    # Write episode
deepscript playbook sales-call              # Auto-generated playbook
deepscript prep discovery-call              # Call prep from patterns
```

### BFlow
DeepScript registers as a BFlow concierge skill. See [SKILL.md](SKILL.md).

### MCP Server
```bash
python -m deepscript.mcp_server              # Start MCP server
```

Tools: `deepscript_analyze`, `deepscript_classify`, `deepscript_list_types`

## Development

```bash
pip install -e ".[dev]"        # Install with dev deps
pip install -e ".[dev,llm]"    # Include Anthropic SDK
pytest                          # Run tests (173 tests)
pytest -k "test_classify"      # Run specific tests
make clean                      # Remove build artifacts
```

## License

MIT
