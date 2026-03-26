"""deepscript speakers — Cross-call speaker identification and profiles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from deepscript.cli.output import CLIContext, OutputFormat, emit


def speakers(
    action: str = typer.Argument(help="Action: identify | profile | list"),
    name_or_id: Optional[str] = typer.Argument(None, help="Speaker name or cluster ID (for profile)."),
    transcripts: Optional[str] = typer.Option(None, "--transcripts", "-t", help="Transcript directory."),
    speaker_db: Optional[str] = typer.Option(None, "--speaker-db", help="AudioScript speaker_identities.json path."),
    calendar: str = typer.Option("none", "--calendar", help="Calendar provider: ms365 | google | none."),
    contacts: str = typer.Option("none", "--contacts", help="Contacts provider: ms365 | google | none."),
    writeback: bool = typer.Option(False, "--writeback", help="Write identified names back to speaker DB."),
    min_confidence: float = typer.Option(0.40, "--min-confidence", help="Minimum confidence for writeback."),
    ctx: typer.Context = typer.Option(None, hidden=True),
) -> None:
    """Cross-call speaker identification and profiles."""
    cli_ctx: CLIContext = ctx.obj if ctx and ctx.obj else CLIContext()

    from deepscript.core.speaker_intelligence import (
        identify_speakers,
        format_speaker_profiles,
        writeback_to_speaker_db,
    )

    if action == "identify":
        if not transcripts:
            cli_ctx.console.print("[red]--transcripts required for identify[/red]")
            raise typer.Exit(1)

        # Auto-find speaker DB if not specified
        db_path = speaker_db
        if not db_path:
            candidate = Path(transcripts) / "speaker_identities.json"
            if candidate.exists():
                db_path = str(candidate)

        profiles = identify_speakers(
            transcript_dir=transcripts,
            speaker_db_path=db_path,
            calendar_provider=calendar,
            contacts_provider=contacts,
        )

        # Writeback to speaker DB if requested
        wb_result = None
        if writeback and db_path:
            wb_result = writeback_to_speaker_db(profiles, db_path, min_confidence=min_confidence)
            cli_ctx.console.print(
                f"[green]Writeback:[/green] {wb_result['updated']} updated, "
                f"{wb_result['skipped_confirmed']} kept (confirmed), "
                f"{wb_result['skipped_low_confidence']} skipped (low confidence)"
            )

        if cli_ctx.format in (OutputFormat.JSON, OutputFormat.QUIET, OutputFormat.YAML):
            result = {
                "total_clusters": len(profiles),
                "identified": sum(1 for p in profiles.values() if p.likely_name),
                "unidentified": sum(1 for p in profiles.values() if not p.likely_name),
                "profiles": {cid: p.to_dict() for cid, p in profiles.items()},
            }
            if wb_result:
                result["writeback"] = wb_result
            emit(result, cli_ctx)
        else:
            print(format_speaker_profiles(profiles))
            if wb_result and wb_result["updated"] > 0:
                print(f"\n## Writeback Summary")
                print(f"Updated {wb_result['updated']} names in speaker DB:")
                for change in wb_result["details"]["updated"]:
                    old = change["old_name"] or "unnamed"
                    print(f"  {change['cluster_id']}: {old} → {change['new_name']} ({change['confidence']:.0%}, {change['calls']} calls)")

    elif action == "profile":
        if not name_or_id:
            cli_ctx.console.print("[red]Specify speaker name or cluster ID[/red]")
            raise typer.Exit(1)
        if not transcripts:
            cli_ctx.console.print("[red]--transcripts required[/red]")
            raise typer.Exit(1)

        db_path = speaker_db
        if not db_path:
            candidate = Path(transcripts) / "speaker_identities.json"
            if candidate.exists():
                db_path = str(candidate)

        profiles = identify_speakers(
            transcript_dir=transcripts,
            speaker_db_path=db_path,
            calendar_provider=calendar,
            contacts_provider=contacts,
        )

        # Find by name or cluster ID
        match = None
        for cid, p in profiles.items():
            if cid == name_or_id or (p.likely_name and p.likely_name.lower() == name_or_id.lower()):
                match = p
                break

        if not match:
            # Partial match
            for p in profiles.values():
                if p.likely_name and name_or_id.lower() in p.likely_name.lower():
                    match = p
                    break

        if match:
            if cli_ctx.format in (OutputFormat.JSON, OutputFormat.QUIET):
                emit(match.to_dict(), cli_ctx)
            else:
                print(format_speaker_profiles({match.cluster_id: match}))
        else:
            cli_ctx.console.print(f"[red]Speaker not found: {name_or_id}[/red]")

    elif action == "list":
        if not transcripts:
            cli_ctx.console.print("[red]--transcripts required[/red]")
            raise typer.Exit(1)

        db_path = speaker_db
        if not db_path:
            candidate = Path(transcripts) / "speaker_identities.json"
            if candidate.exists():
                db_path = str(candidate)

        profiles = identify_speakers(
            transcript_dir=transcripts,
            speaker_db_path=db_path,
        )

        if cli_ctx.format in (OutputFormat.JSON, OutputFormat.QUIET):
            emit({
                "speakers": [
                    {"cluster_id": p.cluster_id, "name": p.likely_name, "confidence": p.name_confidence,
                     "calls": p.total_calls, "role": p.role}
                    for p in sorted(profiles.values(), key=lambda x: -x.name_confidence)
                ],
            }, cli_ctx)
        else:
            named = sorted([p for p in profiles.values() if p.likely_name], key=lambda x: -x.name_confidence)
            unnamed = sorted([p for p in profiles.values() if not p.likely_name], key=lambda x: -x.total_calls)
            print(f"# Speakers — {len(named)} identified, {len(unnamed)} unknown\n")
            for p in named:
                print(f"  ✓ {p.likely_name:<25} {p.cluster_id}  {p.total_calls} calls  {p.name_confidence:.0%}  {p.role or ''}")
            if unnamed:
                print()
                for p in unnamed[:10]:
                    topics = ", ".join(p.topics[:2]) if p.topics else ""
                    print(f"  ? {p.cluster_id:<25} {p.total_calls} calls  {topics}")
                if len(unnamed) > 10:
                    print(f"  ... +{len(unnamed)-10} more")
    else:
        cli_ctx.console.print(f"[red]Unknown action: {action}. Use: identify | profile | list[/red]")
