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
                f"[green]Writeback:[/green] {wb_result['new_names']} new, "
                f"{wb_result['upgraded']} upgraded, "
                f"{wb_result['conflicts']} conflicts, "
                f"{wb_result['skipped_low_confidence']} skipped (low confidence)"
            )
            if wb_result["conflicts"] > 0:
                cli_ctx.console.print("[yellow]Conflicts need human review — see details[/yellow]")

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
            if wb_result:
                details = wb_result.get("details", {})
                if details.get("updated"):
                    print(f"\n## New Names ({len(details['updated'])})")
                    for c in details["updated"]:
                        aliases = ", ".join(c.get("aliases", []))
                        alias_str = f" (aliases: {aliases})" if aliases else ""
                        print(f'  {c["cluster_id"]}: → {c["new_name"]} ({c["confidence"]:.0%}, {c["calls"]} calls){alias_str}')
                if details.get("upgraded"):
                    print(f"\n## Upgraded ({len(details['upgraded'])})")
                    for c in details["upgraded"]:
                        print(f'  {c["cluster_id"]}: {c["old_name"]} → {c["new_name"]} ({c["confidence"]:.0%})')
                if details.get("conflicts"):
                    print(f"\n## Conflicts — Needs Review ({len(details['conflicts'])})")
                    for c in details["conflicts"]:
                        print(f'  {c["cluster_id"]}: DB has "{c["existing_name"]}" but evidence says "{c["proposed_name"]}" ({c["confidence"]:.0%})')

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

        match = None
        for cid, p in profiles.items():
            if cid == name_or_id or (p.likely_name and p.likely_name.lower() == name_or_id.lower()):
                match = p
                break
        if not match:
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
                    {"cluster_id": p.cluster_id, "name": p.likely_name, "display_name": p.display_name,
                     "confidence": p.name_confidence, "calls": p.total_calls, "role": p.role}
                    for p in sorted(profiles.values(), key=lambda x: -x.name_confidence)
                ],
            }, cli_ctx)
        else:
            named = sorted([p for p in profiles.values() if p.likely_name], key=lambda x: -x.name_confidence)
            unnamed = sorted([p for p in profiles.values() if not p.likely_name], key=lambda x: -x.total_calls)
            print(f"# Speakers — {len(named)} identified, {len(unnamed)} unknown\n")
            for p in named:
                print(f"  ✓ {p.display_name:<30} {p.cluster_id}  {p.total_calls} calls  {p.name_confidence:.0%}  {p.role or ''}")
            if unnamed:
                print()
                for p in unnamed[:10]:
                    topics = ", ".join(p.topics[:2]) if p.topics else ""
                    print(f"  ? {p.cluster_id:<25} {p.total_calls} calls  {topics}")
                if len(unnamed) > 10:
                    print(f"  ... +{len(unnamed)-10} more")
    elif action == "pages":
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

        from deepscript.integrations.minotes import generate_contact_pages, generate_contacts_summary

        output_dir = name_or_id or "CRM/Contacts"
        analysis_dir = None
        for cd in [Path("analysis-output"), Path(transcripts).parent / "analysis-output"]:
            if cd.exists():
                analysis_dir = str(cd)
                break

        pages = generate_contact_pages(
            profiles, transcripts,
            analysis_dir=analysis_dir,
            output_dir=output_dir,
            speaker_db_path=db_path,
            min_calls=2,
        )

        index = generate_contacts_summary(profiles, output_dir)
        index_path = Path(output_dir) / "_Index.md"
        with open(index_path, "w") as f:
            f.write(index)
        pages.append(index_path)

        cli_ctx.console.print(f"[green]Generated {len(pages)} contact pages in {output_dir}/[/green]")
        if cli_ctx.format in (OutputFormat.JSON, OutputFormat.QUIET):
            emit({"pages": [str(p) for p in pages], "count": len(pages), "output_dir": output_dir}, cli_ctx)

    else:
        cli_ctx.console.print(f"[red]Unknown action: {action}. Use: identify | profile | list | pages[/red]")
