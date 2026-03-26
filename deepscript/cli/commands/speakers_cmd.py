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
    ctx: typer.Context = typer.Option(None, hidden=True),
) -> None:
    """Cross-call speaker identification and profiles."""
    cli_ctx: CLIContext = ctx.obj if ctx and ctx.obj else CLIContext()

    from deepscript.core.speaker_intelligence import (
        identify_speakers,
        format_speaker_profiles,
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

        if cli_ctx.format in (OutputFormat.JSON, OutputFormat.QUIET, OutputFormat.YAML):
            emit({
                "total_clusters": len(profiles),
                "identified": sum(1 for p in profiles.values() if p.likely_name),
                "unidentified": sum(1 for p in profiles.values() if not p.likely_name),
                "profiles": {cid: p.to_dict() for cid, p in profiles.items()},
            }, cli_ctx)
        else:
            print(format_speaker_profiles(profiles))

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
