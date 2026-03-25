"""Markdown output formatter for DeepScript analysis results."""

from __future__ import annotations

import re
from typing import Any

from deepscript.analyzers.base import AnalysisResult
from deepscript.core.classifier import Classification
from deepscript.core.communication import CommunicationMetrics
from deepscript.core.topic_segmenter import Topic


def _esc(text: str | None) -> str:
    """Escape markdown-unsafe characters in user-controlled content."""
    if not text:
        return ""
    # Escape characters that could break markdown or inject HTML
    text = text.replace("|", "\\|")  # Table cell breaker
    text = text.replace("<", "&lt;").replace(">", "&gt;")  # HTML injection
    return text


def format_markdown(
    classification: Classification,
    communication: CommunicationMetrics | None,
    analysis: AnalysisResult | None,
    topics: list[Topic] | None = None,
    source_file: str | None = None,
) -> str:
    """Build markdown output from analysis components."""
    lines: list[str] = []

    title = source_file or "Transcript Analysis"
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"**Type:** {classification.call_type} (confidence: {classification.confidence})")
    if classification.reasoning:
        lines.append(f"**Reasoning:** {classification.reasoning}")
    lines.append("")

    # Communication metrics
    if communication and communication.total_speakers > 0:
        lines.append("## Communication")
        lines.append("")
        lines.append(f"- **Speakers:** {communication.total_speakers}")
        lines.append(f"- **Total words:** {communication.total_words}")
        lines.append(f"- **Questions asked:** {communication.total_questions}")
        lines.append(f"- **Speaking balance:** {communication.speaking_balance}")
        lines.append("")
        if communication.speakers:
            lines.append("### Speaker Stats")
            lines.append("")
            lines.append("| Speaker | Words | Talk % | Questions | Longest Monologue |")
            lines.append("|---------|-------|--------|-----------|-------------------|")
            for s in communication.speakers:
                pct = f"{s.talk_ratio * 100:.1f}%"
                lines.append(f"| {_esc(s.speaker)} | {s.word_count} | {pct} | {s.question_count} | {s.longest_monologue_words} words |")
            lines.append("")

    # Topics
    if topics:
        lines.append("## Topics Index")
        lines.append("")
        lines.append("| # | Topic | Time | Speakers |")
        lines.append("|---|-------|------|----------|")
        for i, t in enumerate(topics, 1):
            time_str = _format_timestamp(t.start_seconds)
            speakers = ", ".join(t.speakers) if t.speakers else ""
            lines.append(f"| {i} | {_esc(t.name)} | {time_str} | {_esc(speakers)} |")
        lines.append("")

    # Analysis sections
    if analysis:
        sections = analysis.sections
        _render_summary(lines, sections)
        _render_action_items(lines, sections)
        _render_decisions(lines, sections)
        _render_questions(lines, sections)
        _render_methodology_score(lines, sections)
        _render_signals(lines, sections)
        _render_discovery(lines, sections)
        _render_pmf(lines, sections)
        _render_interview(lines, sections)
        _render_support(lines, sections)
        _render_qbr(lines, sections)
        _render_relationship(lines, sections)
        _render_attendees(lines, sections)

    return "\n".join(lines)


def _render_summary(lines: list[str], sections: dict[str, Any]) -> None:
    summary = sections.get("summary")
    if not summary:
        return
    lines.append("## Summary")
    lines.append("")
    lines.append(summary.get("text", ""))
    duration = summary.get("duration_seconds")
    if duration:
        mins = int(duration // 60)
        secs = int(duration % 60)
        lines.append(f"\n*Duration: {mins}m {secs}s | {summary.get('word_count', 0)} words*")
    lines.append("")


def _render_action_items(lines: list[str], sections: dict[str, Any]) -> None:
    items = sections.get("action_items", [])
    if not items:
        return
    lines.append("## Action Items")
    lines.append("")
    for item in items:
        speaker = item.get("speaker", "")
        text = item.get("text", "")
        assignee = item.get("assignee")
        deadline = item.get("deadline")
        ts = item.get("timestamp")
        parts = [f"- [ ] "]
        if assignee:
            parts.append(f"**{assignee}**")
        elif speaker:
            parts.append(f"**{speaker}**")
        if deadline:
            parts.append(f" (by {deadline})")
        elif ts is not None:
            parts.append(f" ({_format_timestamp(ts)})")
        parts.append(f": {text}")
        lines.append("".join(parts))
    lines.append("")


def _render_decisions(lines: list[str], sections: dict[str, Any]) -> None:
    decisions = sections.get("decisions", [])
    if not decisions:
        return
    lines.append("## Key Decisions")
    lines.append("")
    for d in decisions:
        speaker = d.get("speaker", "")
        text = d.get("text", "")
        context = d.get("context")
        line = f"- **{speaker}:** {text}"
        if context:
            line += f" _{context}_"
        lines.append(line)
    lines.append("")


def _render_questions(lines: list[str], sections: dict[str, Any]) -> None:
    questions = sections.get("questions", [])
    if not questions:
        return
    lines.append("## Questions Raised")
    lines.append("")
    for q in questions:
        speaker = q.get("speaker", "")
        text = q.get("text", "")
        ts = q.get("timestamp")
        ts_str = f" ({_format_timestamp(ts)})" if ts is not None else ""
        lines.append(f"- **{speaker}**{ts_str}: {text}")
    lines.append("")


def _render_methodology_score(lines: list[str], sections: dict[str, Any]) -> None:
    score = sections.get("methodology_score")
    if not score:
        return
    methodology = score.get("methodology", "")
    lines.append(f"## {methodology} Score")
    lines.append("")
    total = score.get("total_score", 0)
    max_possible = score.get("max_possible", 0)
    lines.append(f"**Score: {total}/{max_possible}**")
    lines.append("")
    if score.get("overall_assessment"):
        lines.append(f"*{score['overall_assessment']}*")
        lines.append("")

    scores = score.get("scores", {})
    if scores:
        lines.append("| Dimension | Score | Evidence |")
        lines.append("|-----------|-------|----------|")
        for dim, data in scores.items():
            if isinstance(data, dict):
                s = data.get("score", "?")
                evidence = data.get("evidence", "")[:80]
                lines.append(f"| {_esc(dim)} | {s}/3 | {_esc(evidence)} |")
        lines.append("")

    strengths = score.get("strengths", [])
    if strengths:
        lines.append("**Strengths:** " + ", ".join(strengths))
    gaps = score.get("gaps", [])
    if gaps:
        lines.append("**Gaps:** " + ", ".join(gaps))
    if strengths or gaps:
        lines.append("")


def _render_signals(lines: list[str], sections: dict[str, Any]) -> None:
    buying = sections.get("buying_signals", [])
    risk = sections.get("risk_signals", [])
    next_steps = sections.get("next_steps")
    phases = sections.get("call_phases", [])

    if phases:
        lines.append("## Call Phases")
        lines.append("")
        for p in phases:
            phase = p.get("phase", "")
            summary = p.get("summary", "")
            lines.append(f"- **{phase}**: {summary}")
        lines.append("")

    if buying:
        lines.append("## Buying Signals")
        lines.append("")
        for s in buying:
            signal = s.get("signal", "")
            strength = s.get("strength", "")
            quote = s.get("quote", s.get("keyword", ""))
            lines.append(f"- [{strength}] {signal}: *\"{quote}\"*")
        lines.append("")

    if risk:
        lines.append("## Risk Signals")
        lines.append("")
        for s in risk:
            signal = s.get("signal", "")
            severity = s.get("severity", "")
            quote = s.get("quote", s.get("keyword", ""))
            lines.append(f"- [{severity}] {signal}: *\"{quote}\"*")
        lines.append("")

    if next_steps:
        lines.append("## Next Steps")
        lines.append("")
        quality = next_steps.get("quality", "unknown")
        lines.append(f"**Quality:** {quality}")
        assessment = next_steps.get("assessment")
        if assessment:
            lines.append(f"*{assessment}*")
        items = next_steps.get("items", [])
        for item in items:
            lines.append(f"- {item}")
        lines.append("")


def _render_discovery(lines: list[str], sections: dict[str, Any]) -> None:
    framework_score = sections.get("framework_score")
    pain_points = sections.get("pain_points", [])
    jtbd = sections.get("jtbd", [])
    commitments = sections.get("commitment_signals", [])
    traps = sections.get("compliment_traps", [])
    hidden = sections.get("hidden_opportunities", [])

    if framework_score:
        fw = framework_score.get("framework", "")
        score = framework_score.get("score", 0)
        lines.append(f"## {fw} Score: {score}/10")
        lines.append("")
        notes = framework_score.get("notes", "")
        if notes:
            lines.append(f"*{notes}*")
            lines.append("")

    if pain_points:
        lines.append("## Pain Points")
        lines.append("")
        lines.append("| Pain | Severity | Workaround |")
        lines.append("|------|----------|------------|")
        for p in pain_points:
            pain = p.get("pain", "")
            severity = p.get("severity", "")
            workaround = p.get("workaround") or "-"
            lines.append(f"| {_esc(pain[:60])} | {severity} | {_esc(workaround[:40])} |")
        lines.append("")

    if jtbd:
        lines.append("## Jobs-to-be-Done")
        lines.append("")
        for j in jtbd:
            sit = j.get("situation", "")
            mot = j.get("motivation", "")
            out = j.get("outcome", "")
            lines.append(f"- When {sit}, I want {mot}, so I can {out}")
        lines.append("")

    if commitments:
        lines.append("## Commitment Signals")
        lines.append("")
        for c in commitments:
            ctype = c.get("type", "")
            signal = c.get("signal", "")
            strength = c.get("strength", "")
            lines.append(f"- [{ctype}] {signal} ({strength})")
        lines.append("")

    if traps:
        lines.append("## Compliment Traps")
        lines.append("")
        for t in traps:
            trap = t.get("trap", "")
            quote = t.get("quote", "")
            lines.append(f"- {trap}: *\"{quote}\"*")
        lines.append("")

    if hidden:
        lines.append("## Hidden Opportunities")
        lines.append("")
        for h in hidden:
            lines.append(f"- {h}")
        lines.append("")


def _render_pmf(lines: list[str], sections: dict[str, Any]) -> None:
    """Render PMF analysis sections."""
    pmf_score = sections.get("pmf_score")
    dimensions = sections.get("pmf_dimensions")
    ellis = sections.get("ellis_classification")

    if pmf_score is None and not dimensions:
        return

    lines.append(f"## Product-Market Fit Score: {pmf_score}/10")
    lines.append("")

    if ellis:
        ellis_display = ellis.replace("_", " ").title()
        reasoning = sections.get("ellis_reasoning", "")
        lines.append(f"**Ellis Classification:** {ellis_display}")
        if reasoning:
            lines.append(f"*{reasoning}*")
        lines.append("")

    if dimensions:
        lines.append("### Dimension Scores")
        lines.append("")
        lines.append("| Dimension | Score | Evidence |")
        lines.append("|-----------|-------|----------|")
        for dim, data in dimensions.items():
            if isinstance(data, dict):
                name = dim.replace("_", " ").title()
                score = data.get("score", "?")
                evidence = str(data.get("evidence", ""))[:60]
                lines.append(f"| {_esc(name)} | {score}/10 | {_esc(evidence)} |")
        lines.append("")

    strongest = sections.get("strongest_signals", [])
    if strongest:
        lines.append("### Strongest Signals")
        lines.append("")
        for s in strongest:
            lines.append(f"- {s}")
        lines.append("")

    anti = sections.get("anti_pmf_flags", [])
    if anti:
        lines.append("### Anti-PMF Flags")
        lines.append("")
        for a in anti:
            lines.append(f"- {a}")
        lines.append("")

    quotes = sections.get("key_quotes", [])
    if quotes:
        lines.append("### Key Quotes")
        lines.append("")
        for q in quotes:
            quote = q.get("quote", "")
            signal = q.get("signal_type", "")
            lines.append(f"- *\"{quote}\"* ({signal})")
        lines.append("")


def _render_interview(lines: list[str], sections: dict[str, Any]) -> None:
    """Render interview analysis sections."""
    star = sections.get("star_analysis")
    overall = sections.get("overall_star_score")
    answers = sections.get("interview_answers", [])
    recommendation = sections.get("recommendation")

    if star is None and not answers:
        return

    if overall is not None:
        lines.append(f"## Interview Score: {overall}/10")
        lines.append("")

    if star:
        lines.append("### STAR Analysis")
        lines.append("")
        components = star.get("components", {})
        for comp, count in components.items():
            check = "present" if count > 0 else "missing"
            lines.append(f"- **{comp.title()}**: {check} ({count} indicators)")
        assessment = star.get("assessment", "")
        lines.append(f"\n*Overall: {assessment}*")
        lines.append("")

    if answers:
        lines.append("### Answer Evaluations")
        lines.append("")
        for a in answers:
            q = a.get("question", "")
            strength = a.get("evidence_strength", "")
            star_score = a.get("star_completeness", {}).get("score", "")
            lines.append(f"- **Q:** {q}")
            lines.append(f"  - STAR: {star_score} | Evidence: {strength}")
            competencies = a.get("competencies", [])
            if competencies:
                lines.append(f"  - Competencies: {', '.join(competencies)}")
        lines.append("")

    if recommendation:
        reasoning = sections.get("recommendation_reasoning", "")
        lines.append(f"### Recommendation: {recommendation}")
        if reasoning:
            lines.append(f"*{reasoning}*")
        lines.append("")

    strengths = sections.get("interview_strengths", [])
    concerns = sections.get("interview_concerns", [])
    if strengths:
        lines.append("**Strengths:** " + ", ".join(strengths))
    if concerns:
        lines.append("**Concerns:** " + ", ".join(concerns))
    if strengths or concerns:
        lines.append("")


def _render_support(lines: list[str], sections: dict[str, Any]) -> None:
    """Render support call sections."""
    issue = sections.get("issue")
    emotion = sections.get("emotion_trajectory")
    resolution = sections.get("resolution")
    empathy = sections.get("empathy_score")

    if not issue and not resolution:
        return

    if issue:
        lines.append(f"## Issue: {issue.get('type', 'unknown').replace('_', ' ').title()}")
        lines.append("")
        if issue.get("summary"):
            lines.append(f"*{issue['summary']}*")
        severity = issue.get("severity", "")
        if severity:
            lines.append(f"**Severity:** {severity}")
        lines.append("")

    if emotion:
        start = emotion.get("start", "")
        end = emotion.get("end", "")
        lines.append(f"### Emotion: {start} -> {end}")
        arc = emotion.get("arc")
        if arc:
            lines.append(f"*{arc}*")
        lines.append("")

    if resolution:
        status = resolution.get("status", "unknown")
        lines.append(f"### Resolution: {status}")
        solution = resolution.get("solution")
        if solution:
            lines.append(f"*{solution}*")
        lines.append("")

    if empathy:
        score = empathy.get("score", 0)
        max_s = empathy.get("max", 10)
        lines.append(f"### Empathy Score: {score}/{max_s}")
        lines.append("")


def _render_qbr(lines: list[str], sections: dict[str, Any]) -> None:
    """Render QBR sections."""
    health = sections.get("health_score")
    expansion = sections.get("expansion_signals", [])
    churn = sections.get("churn_risk")
    renewal = sections.get("renewal_outlook")

    if not health and not churn:
        return

    if health:
        score = health.get("score", 0)
        max_s = health.get("max", 10)
        lines.append(f"## Customer Health: {score}/{max_s}")
        lines.append("")
        trend = health.get("trend")
        if trend:
            lines.append(f"**Trend:** {trend}")
        assessment = health.get("assessment")
        if assessment:
            lines.append(f"*{assessment}*")
        lines.append("")

    if expansion:
        lines.append("### Expansion Signals")
        lines.append("")
        for s in expansion:
            signal = s.get("signal", "")
            opp = s.get("opportunity", "")
            line = f"- {signal}"
            if opp:
                line += f" -> {opp}"
            lines.append(line)
        lines.append("")

    if churn:
        level = churn.get("level", "unknown")
        lines.append(f"### Churn Risk: {level}")
        lines.append("")
        indicators = churn.get("indicators", [])
        for ind in indicators:
            lines.append(f"- {ind.get('indicator', '')}")
        lines.append("")

    if renewal:
        likelihood = renewal.get("likelihood", "")
        lines.append(f"### Renewal: {likelihood}")
        blockers = renewal.get("blockers", [])
        for b in blockers:
            lines.append(f"- Blocker: {b}")
        lines.append("")


def _render_relationship(lines: list[str], sections: dict[str, Any]) -> None:
    """Render relationship analysis sections."""
    tone = sections.get("emotional_tone")
    gottman = sections.get("gottman_indicators", [])
    appreciation = sections.get("appreciation_ratio")
    bids = sections.get("bids_for_connection", [])
    growth = sections.get("growth_suggestions", [])
    listening = sections.get("listening_balance")
    we_i = sections.get("we_i_language")
    validations = sections.get("validation_moments", [])

    if not any([tone, gottman, appreciation, bids, listening]):
        return

    lines.append("## Relationship Insights")
    lines.append("")

    if tone:
        overall = tone.get("overall", "")
        arc = tone.get("arc", "")
        lines.append(f"**Emotional Tone:** {overall}")
        if arc:
            lines.append(f"*{arc}*")
        lines.append("")

    if appreciation:
        ratio = appreciation.get("ratio", "")
        assessment = appreciation.get("assessment", "")
        lines.append(f"### Appreciation Ratio: {ratio}")
        lines.append(f"- Positive interactions: {appreciation.get('positive_count', 0)}")
        lines.append(f"- Negative interactions: {appreciation.get('negative_count', 0)}")
        lines.append(f"- Assessment: {assessment}")
        lines.append("")

    if gottman:
        lines.append("### Gottman Four Horsemen Check")
        lines.append("")
        for g in gottman:
            gtype = g.get("type", "")
            quote = g.get("quote", "")
            reframe = g.get("reframe", "")
            lines.append(f"- **{gtype.title()}**: *\"{quote}\"*")
            if reframe:
                lines.append(f"  - Reframe: {reframe}")
        lines.append("")

    if bids:
        lines.append("### Bids for Connection")
        lines.append("")
        lines.append("| Bid | Response | Quality |")
        lines.append("|-----|----------|---------|")
        for b in bids:
            bid = b.get("bid", "")[:50]
            response = b.get("response", "")[:50]
            quality = b.get("quality", "")
            icon = {"toward": "turned toward", "away": "turned away", "against": "turned against"}.get(quality, quality)
            lines.append(f"| {_esc(bid)} | {_esc(response)} | {icon} |")
        lines.append("")

    if listening:
        speakers = listening.get("speakers", {})
        if speakers:
            lines.append("### Listening Balance")
            lines.append("")
            for spk, stats in speakers.items():
                ratio = stats.get("talk_ratio", 0)
                qs = stats.get("questions_asked", 0)
                lines.append(f"- **{spk}**: {ratio*100:.0f}% talk, {qs} questions")
            lines.append("")

    if validations:
        lines.append(f"### Validation Moments ({len(validations)} detected)")
        lines.append("")
        for v in validations[:5]:
            lines.append(f"- **{v.get('speaker', '')}**: {v.get('text', '')}")
        lines.append("")

    if growth:
        lines.append("### Growth Suggestions")
        lines.append("")
        for g in growth:
            lines.append(f"- {g}")
        lines.append("")


def _render_attendees(lines: list[str], sections: dict[str, Any]) -> None:
    attendees = sections.get("attendees", [])
    if not attendees:
        return
    lines.append("## Attendees")
    lines.append("")
    lines.append("| Speaker | Words | Talk % |")
    lines.append("|---------|-------|--------|")
    for a in attendees:
        pct = f"{a.get('talk_ratio', 0) * 100:.1f}%"
        lines.append(f"| {_esc(a.get('speaker', ''))} | {a.get('word_count', 0)} | {pct} |")
    lines.append("")


def _format_timestamp(seconds: float | None) -> str:
    if seconds is None:
        return ""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"
