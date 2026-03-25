"""DeepScript MCP Server — exposes analysis tools for agent integration.

Run with: python -m deepscript.mcp_server
Or register in .mcp.json for BFlow/Claude Code integration.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _analyze_transcript(file_path: str, call_type: str | None = None, no_llm: bool = False) -> dict[str, Any]:
    """Analyze a transcript file and return structured results."""
    from deepscript.analyzers.business import BusinessAnalyzer
    from deepscript.analyzers.discovery import DiscoveryAnalyzer
    from deepscript.analyzers.interview import InterviewAnalyzer
    from deepscript.analyzers.pmf import PMFAnalyzer
    from deepscript.analyzers.qbr import QBRAnalyzer
    from deepscript.analyzers.sales import SalesAnalyzer
    from deepscript.analyzers.support import SupportAnalyzer
    from deepscript.config.settings import get_settings
    from deepscript.core.classifier import classify_transcript
    from deepscript.core.communication import analyze_communication
    from deepscript.formatters.json_formatter import format_json
    from deepscript.llm.provider import LLMProvider

    path = Path(file_path)
    if not path.exists():
        return {"error": f"File not found: {file_path}"}

    with open(path, "r", encoding="utf-8") as f:
        transcript = json.load(f)

    if "text" not in transcript and "segments" in transcript:
        transcript["text"] = " ".join(s.get("text", "") for s in transcript["segments"])

    settings = get_settings()
    llm = None if no_llm else LLMProvider.create(settings.llm)

    # Classify
    if call_type:
        from deepscript.core.classifier import Classification
        classification = Classification(call_type=call_type, confidence=1.0, scores={call_type: 1.0})
    else:
        classification = classify_transcript(transcript, settings.custom_classifications, llm=llm)

    # Communication
    communication = analyze_communication(transcript)

    # Analyzer registry
    analyzers: dict[str, Any] = {
        "business-meeting": BusinessAnalyzer(llm=llm),
        "sales-call": SalesAnalyzer(llm=llm, methodology=settings.sales.methodology),
        "discovery-call": DiscoveryAnalyzer(llm=llm, framework=settings.discovery.framework),
        "pmf-call": PMFAnalyzer(llm=llm),
        "interview-behavioral": InterviewAnalyzer(llm=llm, interview_type="behavioral"),
        "interview-technical": InterviewAnalyzer(llm=llm, interview_type="technical"),
        "support-escalation": SupportAnalyzer(llm=llm),
        "qbr": QBRAnalyzer(llm=llm),
    }

    analyzer = analyzers.get(classification.call_type, analyzers["business-meeting"])
    analysis = analyzer.analyze(transcript)
    analysis.call_type = classification.call_type

    return format_json(classification, communication, analysis, source_file=file_path)


def _classify_transcript(file_path: str) -> dict[str, Any]:
    """Classify a transcript and return the result."""
    from deepscript.config.settings import get_settings
    from deepscript.core.classifier import classify_transcript

    path = Path(file_path)
    if not path.exists():
        return {"error": f"File not found: {file_path}"}

    with open(path, "r", encoding="utf-8") as f:
        transcript = json.load(f)

    if "text" not in transcript and "segments" in transcript:
        transcript["text"] = " ".join(s.get("text", "") for s in transcript["segments"])

    settings = get_settings()
    classification = classify_transcript(transcript, settings.custom_classifications)

    return {
        "file": file_path,
        "call_type": classification.call_type,
        "confidence": classification.confidence,
        "scores": classification.scores,
    }


def _list_types() -> dict[str, Any]:
    """List all supported call types."""
    return {
        "types": [
            {"type": "business-meeting", "analyzer": "BusinessAnalyzer", "tier": 1},
            {"type": "sales-call", "analyzer": "SalesAnalyzer", "tier": 1},
            {"type": "sales-discovery", "analyzer": "SalesAnalyzer", "tier": 1},
            {"type": "discovery-call", "analyzer": "DiscoveryAnalyzer", "tier": 1},
            {"type": "pmf-call", "analyzer": "PMFAnalyzer", "tier": 1},
            {"type": "interview-behavioral", "analyzer": "InterviewAnalyzer", "tier": 1},
            {"type": "interview-technical", "analyzer": "InterviewAnalyzer", "tier": 1},
            {"type": "support-escalation", "analyzer": "SupportAnalyzer", "tier": 1},
            {"type": "qbr", "analyzer": "QBRAnalyzer", "tier": 1},
            {"type": "standup", "analyzer": "BusinessAnalyzer", "tier": 1},
            {"type": "family", "analyzer": "RelationshipAnalyzer", "tier": 3},
            {"type": "partner", "analyzer": "RelationshipAnalyzer", "tier": 3},
            {"type": "voice-memo", "analyzer": "BusinessAnalyzer", "tier": 3},
        ]
    }


# --- MCP Server Entry Point ---

def create_mcp_server() -> Any:
    """Create FastMCP server with DeepScript tools."""
    try:
        from fastmcp import FastMCP
    except ImportError:
        logger.error("fastmcp not installed. Run: pip install fastmcp")
        sys.exit(1)

    mcp = FastMCP("deepscript", description="Transcript Intelligence Engine")

    @mcp.tool()
    def deepscript_analyze(file_path: str, call_type: str | None = None, no_llm: bool = False) -> str:
        """Analyze a transcript file for insights, action items, and scoring.

        Args:
            file_path: Path to transcript JSON file.
            call_type: Override auto-classification (e.g., "sales-call", "pmf-call").
            no_llm: If true, use rule-based analysis only.
        """
        result = _analyze_transcript(file_path, call_type, no_llm)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def deepscript_classify(file_path: str) -> str:
        """Classify a transcript's call type without running full analysis.

        Args:
            file_path: Path to transcript JSON file.
        """
        result = _classify_transcript(file_path)
        return json.dumps(result, indent=2, default=str)

    @mcp.tool()
    def deepscript_list_types() -> str:
        """List all supported call types and their analyzers."""
        result = _list_types()
        return json.dumps(result, indent=2)

    return mcp


if __name__ == "__main__":
    server = create_mcp_server()
    server.run()
