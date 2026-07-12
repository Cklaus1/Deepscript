"""deepscript speakers — Cross-call speaker identification and profiles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from deepscript.cli.output import CLIContext, OutputFormat, emit


def speakers(
    action: str = typer.Argument(help="Action: identify | profile | list | pages | dedup | unmerge | not-same"),
    name_or_id: Optional[str] = typer.Argument(None, help="Speaker name or cluster ID (for profile, pages, dedup, unmerge, not-same)."),
    transcripts: Optional[str] = typer.Option(None, "--transcripts", "-t", help="Transcript directory."),
    speaker_db: Optional[str] = typer.Option(None, "--speaker-db", help="AudioScript speaker_identities.json path."),
    calendar: str = typer.Option("none", "--calendar", help="Calendar provider: ms365 | google | none."),
    contacts: str = typer.Option("none", "--contacts", help="Contacts provider: ms365 | google | none."),
    writeback: bool = typer.Option(False, "--writeback", help="Write identified names back to speaker DB."),
    min_confidence: float = typer.Option(0.40, "--min-confidence", help="Minimum confidence for writeback."),
    format: Optional[str] = typer.Option(None, "--format", "-f", help="Output format: json | yaml | table (default: auto)."),
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

        from deepscript.integrations.minotes import generate_crm_pages

        output_dir = name_or_id or "CRM"
        analysis_dir = None
        for cd in [Path("analysis-output"), Path(transcripts).parent / "analysis-output"]:
            if cd.exists():
                analysis_dir = str(cd)
                break

        # Look for CMS store path
        cms_path = None
        for ep in [Path("CMS"), Path(transcripts).parent / "CMS"]:
            if ep.exists():
                cms_path = str(ep)
                break

        result = generate_crm_pages(
            profiles=profiles,
            transcript_dir=transcripts,
            analysis_dir=analysis_dir,
            output_dir=output_dir,
            speaker_db_path=db_path,
            min_calls=2,
            cms_store_path=cms_path,
        )

        total = sum(len(v) for v in result.values())
        cli_ctx.console.print(
            f"[green]Generated {total} CRM pages "
            f"({len(result.get('people', []))} people, "
            f"{len(result.get('companies', []))} companies, "
            f"{len(result.get('interactions', []))} interactions)[/green]"
        )
        if cli_ctx.format in (OutputFormat.JSON, OutputFormat.QUIET):
            emit({"crm": {k: [str(p) for p in v] for k, v in result.items()}, "count": total, "output_dir": output_dir}, cli_ctx)

    elif action == "dedup":
        if not speaker_db and not transcripts:
            cli_ctx.console.print("[red]--speaker-db or --transcripts required for dedup[/red]")
            raise typer.Exit(1)

        db_path = speaker_db
        if not db_path:
            candidate = Path(transcripts) / "speaker_identities.json"
            if candidate.exists():
                db_path = str(candidate)
        if not db_path or not Path(db_path).exists():
            cli_ctx.console.print("[red]Speaker DB not found[/red]")
            raise typer.Exit(1)

        from deepscript.core.speaker_intelligence import (
            find_duplicate_speakers,
            merge_speakers,
            unmerge_speakers,
        )

        candidates = find_duplicate_speakers(db_path)

        if not candidates:
            cli_ctx.console.print("[green]No duplicate speakers found.[/green]")
            return

        if cli_ctx.format == OutputFormat.QUIET:
            emit({"candidates": [c.to_dict() for c in candidates], "count": len(candidates)}, cli_ctx)
        else:
            cli_ctx.console.print(f"\n[bold]Merge Candidates ({len(candidates)}):[/bold]\n")
            cli_ctx.console.print(
                f"{'#':>3}  {'Score':>5}  {'Voice':>5}  {'Name':>5}  {'CoSpk':>5}  "
                f"{'Cluster A':16}  {'Cluster B':16}  {'Name A':25}  {'Name B':25}  Reasons"
            )
            cli_ctx.console.print("─" * 140)
            for i, c in enumerate(candidates, 1):
                cli_ctx.console.print(
                    f"{i:>3}  {c.total_score:>5.2f}  {c.embedding_similarity:>5.2f}  "
                    f"{c.name_similarity:>5.2f}  {c.co_speaker_overlap:>5.2f}  "
                    f"{c.cluster_a:16}  {c.cluster_b:16}  "
                    f"{c.name_a[:25]:25}  {c.name_b[:25]:25}  "
                    f"{'; '.join(c.reasons)}"
                )

            # Auto-merge if --writeback and score >= 0.80
            if writeback:
                high_confidence = [c for c in candidates if c.total_score >= 0.80]
                if high_confidence:
                    cli_ctx.console.print(f"\n[bold yellow]Soft-merging {len(high_confidence)} pairs (score ≥ 0.80, reversible):[/bold yellow]")
                    merged = 0
                    for c in high_confidence:
                        # Keep the cluster with more calls
                        if c.calls_a >= c.calls_b:
                            keep, remove = c.cluster_a, c.cluster_b
                            keep_name, remove_name = c.name_a, c.name_b
                        else:
                            keep, remove = c.cluster_b, c.cluster_a
                            keep_name, remove_name = c.name_b, c.name_a
                        if merge_speakers(db_path, keep, remove, hard=False):
                            cli_ctx.console.print(f"  [green]✓[/green] Linked {remove} ({remove_name}) → {keep} ({keep_name})")
                            merged += 1
                        else:
                            cli_ctx.console.print(f"  [red]✗[/red] Failed: {remove} → {keep}")
                    cli_ctx.console.print(f"\n[green]Soft-merged {merged}/{len(high_confidence)} pairs (use 'unmerge' to undo)[/green]")
                else:
                    cli_ctx.console.print("\n[yellow]No pairs above auto-merge threshold (0.80). Review manually.[/yellow]")

    elif action == "unmerge":
        if not name_or_id:
            cli_ctx.console.print("[red]Specify cluster ID to unmerge[/red]")
            raise typer.Exit(1)

        db_path = speaker_db
        if not db_path:
            if transcripts:
                candidate = Path(transcripts) / "speaker_identities.json"
                if candidate.exists():
                    db_path = str(candidate)
        if not db_path or not Path(db_path).exists():
            cli_ctx.console.print("[red]Speaker DB not found. Use --speaker-db[/red]")
            raise typer.Exit(1)

        from deepscript.core.speaker_intelligence import unmerge_speakers

        if unmerge_speakers(db_path, name_or_id):
            cli_ctx.console.print(f"[green]✓ Unmerged {name_or_id} — cluster is independent again[/green]")
        else:
            cli_ctx.console.print(f"[red]✗ Failed to unmerge {name_or_id}. Is it a soft-merged cluster?[/red]")

    elif action == "not-same":
        if not name_or_id or " " not in name_or_id:
            cli_ctx.console.print("[red]Usage: speakers not-same 'cluster_a cluster_b' --speaker-db <path>[/red]")
            cli_ctx.console.print("[red]Provide two cluster IDs separated by a space[/red]")
            raise typer.Exit(1)

        parts = name_or_id.strip().split()
        if len(parts) != 2:
            cli_ctx.console.print("[red]Provide exactly two cluster IDs[/red]")
            raise typer.Exit(1)

        cluster_a, cluster_b = parts

        db_path = speaker_db
        if not db_path:
            if transcripts:
                candidate = Path(transcripts) / "speaker_identities.json"
                if candidate.exists():
                    db_path = str(candidate)
        if not db_path or not Path(db_path).exists():
            cli_ctx.console.print("[red]Speaker DB not found. Use --speaker-db[/red]")
            raise typer.Exit(1)

        from deepscript.core.speaker_intelligence import mark_not_same

        # Load DB to show names
        with open(db_path) as f:
            _db = json.load(f)
        _ids = _db.get("identities", {})
        name_a = _ids.get(cluster_a, {}).get("canonical_name") or cluster_a
        name_b = _ids.get(cluster_b, {}).get("canonical_name") or cluster_b

        if mark_not_same(db_path, cluster_a, cluster_b):
            cli_ctx.console.print(f"[green]Marked as different people:[/green] {name_a} ({cluster_a}) ≠ {name_b} ({cluster_b})")
            cli_ctx.console.print("[dim]Future dedup scans and conflict detection will skip this pair.[/dim]")
        else:
            cli_ctx.console.print(f"[red]Failed — check cluster IDs exist in the DB[/red]")

    else:
        cli_ctx.console.print(f"[red]Unknown action: {action}. Use: identify | profile | list | pages | dedup | unmerge | not-same[/red]")
