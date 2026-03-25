"""deepscript benchmark — Evaluate, rank, compare, and trend LLM models."""

from __future__ import annotations

import logging
from typing import Optional

import typer

from deepscript.cli.output import CLIContext, OutputFormat, emit

logger = logging.getLogger(__name__)


def benchmark(
    provider: str = typer.Option(
        "nim",
        "--provider",
        "-p",
        help="LLM provider to benchmark (nim, ollama, vllm, sglang, openai, claude).",
    ),
    models: Optional[str] = typer.Option(
        None,
        "--models",
        "-m",
        help="Comma-separated model IDs to test. Default: auto-discover from catalog.",
    ),
    top_n: int = typer.Option(
        10,
        "--top",
        "-n",
        help="Number of top models to benchmark from catalog (when auto-discovering).",
    ),
    tasks: Optional[str] = typer.Option(
        None,
        "--tasks",
        help="Comma-separated tasks to run (classify,summarize,action_items,sales_score,discovery_score). Default: all.",
    ),
    base_url: Optional[str] = typer.Option(
        None,
        "--base-url",
        help="Custom API endpoint.",
    ),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        help="API key (overrides env var).",
    ),
    list_models: bool = typer.Option(
        False,
        "--list",
        help="List available models from catalog without benchmarking.",
    ),
    history: bool = typer.Option(
        False,
        "--history",
        help="Show all past benchmark runs.",
    ),
    compare: Optional[str] = typer.Option(
        None,
        "--compare",
        help="Compare two runs by index (e.g., '1,2' for latest vs second-latest). Default: latest two.",
    ),
    trend: Optional[str] = typer.Option(
        None,
        "--trend",
        help="Show quality trend for a specific model across all runs.",
    ),
    ctx: typer.Context = typer.Option(None, hidden=True),
) -> None:
    """Benchmark LLM models for transcript analysis quality and performance.

    Run benchmarks, view history, compare runs, and track model trends.
    """
    cli_ctx: CLIContext = ctx.obj if ctx and ctx.obj else CLIContext()

    # --- History mode ---
    if history:
        _show_history(cli_ctx)
        return

    # --- Compare mode ---
    if compare is not None:
        _compare_runs(compare, cli_ctx)
        return

    # --- Trend mode ---
    if trend:
        _show_trend(trend, cli_ctx)
        return

    # --- List mode ---
    if list_models:
        _list_catalog(provider, cli_ctx)
        return

    # --- Benchmark mode ---
    model_list: list[str]
    if models:
        model_list = [m.strip() for m in models.split(",")]
    elif provider == "nim":
        model_list = _discover_nim_models(top_n)
    else:
        cli_ctx.console.print("[red]Specify --models for non-NIM providers[/red]")
        raise typer.Exit(1)

    if not model_list:
        cli_ctx.console.print("[red]No models to benchmark[/red]")
        raise typer.Exit(1)

    task_list = [t.strip() for t in tasks.split(",")] if tasks else None

    n_tasks = len(task_list) if task_list else 5
    n_fixtures = 7  # all test transcripts
    total_calls = len(model_list) * n_tasks * n_fixtures
    est_minutes = total_calls / 35
    mode = "parallel" if len(model_list) > 1 else "sequential"
    cli_ctx.console.print(
        f"[bold]Benchmarking {len(model_list)} models × {n_tasks} tasks × {n_fixtures} transcripts = {total_calls} API calls[/bold]"
    )
    cli_ctx.console.print(f"[dim]Rate limited to 35 req/min — ETA ~{est_minutes:.0f} min ({mode})[/dim]")
    for m in model_list:
        cli_ctx.console.print(f"  - {m}")

    from deepscript.benchmark.runner import (
        format_benchmark_markdown,
        run_benchmark,
        save_benchmark_results,
    )

    import time as _time
    bench_start = _time.monotonic()

    results = run_benchmark(
        models=model_list,
        provider=provider,
        tasks=task_list,
        base_url=base_url,
        api_key=api_key,
        parallel=len(model_list) > 1,
        max_parallel=top_n,
    )

    bench_elapsed = round(_time.monotonic() - bench_start, 1)
    cli_ctx.console.print(f"[dim]Completed in {bench_elapsed}s[/dim]")

    output_path = save_benchmark_results(results)

    if cli_ctx.format in (OutputFormat.JSON, OutputFormat.QUIET, OutputFormat.YAML):
        emit(
            {
                "models_tested": len(results),
                "results_file": str(output_path),
                "rankings": [
                    {
                        "model": b.model,
                        "tier": b.tier,
                        "avg_quality": b.avg_quality,
                        "avg_latency_ms": b.avg_latency_ms,
                        "total_cost_usd": b.total_cost_usd,
                        "success_rate": b.success_rate,
                    }
                    for b in results
                ],
            },
            cli_ctx,
        )
    else:
        md = format_benchmark_markdown(results)
        print(md)

    cli_ctx.console.print(f"\n[green]Results saved to {output_path}[/green]")


# --- Subcommand implementations ---


def _show_history(cli_ctx: CLIContext) -> None:
    """Show all past benchmark runs."""
    from deepscript.benchmark.history import format_history_markdown, list_benchmark_runs

    runs = list_benchmark_runs()

    if cli_ctx.format in (OutputFormat.JSON, OutputFormat.QUIET, OutputFormat.YAML):
        emit({"runs": runs, "total": len(runs)}, cli_ctx)
    else:
        print(format_history_markdown(runs))


def _compare_runs(compare_arg: str, cli_ctx: CLIContext) -> None:
    """Compare two benchmark runs."""
    from deepscript.benchmark.history import (
        compare_runs,
        format_comparison_markdown,
        list_benchmark_runs,
        load_benchmark_run,
    )

    runs = list_benchmark_runs()
    if len(runs) < 2:
        cli_ctx.console.print("[yellow]Need at least 2 benchmark runs to compare[/yellow]")
        raise typer.Exit(1)

    # Parse indices (1-based)
    parts = [p.strip() for p in compare_arg.split(",") if p.strip()]
    if len(parts) == 2:
        try:
            idx_a = int(parts[0]) - 1
            idx_b = int(parts[1]) - 1
        except ValueError:
            cli_ctx.console.print("[red]--compare expects two numbers (e.g., '1,2')[/red]")
            raise typer.Exit(1)
    elif len(parts) == 0 or compare_arg == "":
        # Default: compare latest two
        idx_a = 1  # second-latest (baseline)
        idx_b = 0  # latest (current)
    else:
        cli_ctx.console.print("[red]--compare expects two numbers (e.g., '1,2') or empty for latest two[/red]")
        raise typer.Exit(1)

    if idx_a >= len(runs) or idx_b >= len(runs):
        cli_ctx.console.print(f"[red]Only {len(runs)} runs available[/red]")
        raise typer.Exit(1)

    run_a = load_benchmark_run(runs[idx_a]["path"])
    run_b = load_benchmark_run(runs[idx_b]["path"])

    if not run_a or not run_b:
        cli_ctx.console.print("[red]Failed to load benchmark runs[/red]")
        raise typer.Exit(1)

    comparison = compare_runs(run_a, run_b)

    if cli_ctx.format in (OutputFormat.JSON, OutputFormat.QUIET, OutputFormat.YAML):
        emit(comparison, cli_ctx)
    else:
        print(format_comparison_markdown(comparison))


def _show_trend(model_id: str, cli_ctx: CLIContext) -> None:
    """Show quality trend for a model across runs."""
    from deepscript.benchmark.history import (
        format_trend_markdown,
        model_stats,
        model_trend,
    )

    points = model_trend(model_id)
    stats = model_stats(model_id)

    if cli_ctx.format in (OutputFormat.JSON, OutputFormat.QUIET, OutputFormat.YAML):
        emit({
            "model": model_id,
            "data_points": len(points),
            "statistics": stats,
            "trend": points,
        }, cli_ctx)
    else:
        print(format_trend_markdown(model_id, points))


def _list_catalog(provider: str, cli_ctx: CLIContext) -> None:
    """List available models from catalog."""
    if provider != "nim":
        cli_ctx.console.print("[yellow]Catalog listing only available for NIM provider[/yellow]")
        return

    from deepscript.benchmark.nim_catalog import (
        categorize_models,
        fetch_nim_models,
        filter_chat_models,
    )

    cli_ctx.console.print("[bold]Fetching NIM model catalog...[/bold]")
    all_models = fetch_nim_models()
    chat_models = filter_chat_models(all_models)

    if cli_ctx.format in (OutputFormat.JSON, OutputFormat.QUIET):
        emit(
            {
                "total_models": len(all_models),
                "chat_models": len(chat_models),
                "models": [
                    {"id": m.id, "owned_by": m.owned_by, "tier_hint": m.tier_hint}
                    for m in chat_models
                ],
            },
            cli_ctx,
        )
    else:
        by_owner = categorize_models(chat_models)
        print(f"# NIM Model Catalog — {len(chat_models)} chat models (of {len(all_models)} total)\n")
        for owner, models_list in by_owner.items():
            print(f"## {owner} ({len(models_list)} models)")
            for m in models_list:
                tier_str = f" [Tier {m.tier_hint}]" if m.tier_hint else ""
                print(f"  - {m.id}{tier_str}")
            print()


def _discover_nim_models(top_n: int) -> list[str]:
    """Auto-discover top NIM models for benchmarking."""
    from deepscript.benchmark.nim_catalog import fetch_nim_models, filter_chat_models

    all_models = fetch_nim_models()
    chat_models = filter_chat_models(all_models)

    priority_owners = ["meta", "nvidia", "deepseek-ai", "mistralai", "qwen", "google", "microsoft"]

    def sort_key(m):
        tier = m.tier_hint if m.tier_hint else 4
        owner_rank = priority_owners.index(m.owned_by) if m.owned_by in priority_owners else 99
        return (tier, owner_rank, m.id)

    chat_models.sort(key=sort_key)
    return [m.id for m in chat_models[:top_n]]
