"""
main.py
=======
CLI entry point for the Company Data Enrichment System.

Run with:
    python main.py

Presents an interactive Rich menu:
    1. Process CSVs   — load + clean all input files
    2. Enrich Data    — scrape + LLM-extract company information
    3. Validate       — validate enriched data, generate report
    4. Export Results — write final CSV + JSON to output/
    5. Exit

The pipeline is designed to be run in sequence (1 -> 2 -> 3 -> 4), but
individual steps can also be re-run independently (e.g. re-validate
after manual edits to the enriched CSV).

State is held in the PipelineState dataclass and checkpointed to disk
between steps so the program can be restarted without losing progress.
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

# ---------------------------------------------------------------------------
# Bootstrap: logging must be set up BEFORE importing pipeline modules so that
# any module-level code that logs is captured correctly.
# ---------------------------------------------------------------------------
from modules.utils import (
    check_environment,
    console,
    export_dataframe,
    format_elapsed,
    make_progress_bar,
    print_dataframe_summary,
    print_environment_check,
    select_standard_columns,
    setup_logging,
    timed,
)

from config import settings

# Setup logging immediately
setup_logging(log_level="INFO", log_file=settings.log_file)
logger = logging.getLogger(__name__)

# Pipeline modules (imported after logging is ready)
from modules.cleaner import DataCleaner
from modules.data_loader import STANDARD_COLUMNS, DataLoader
from modules.enrichment import EnrichmentEngine
from modules.validator import DataValidator, ValidationReport
from modules.confidence import EnrichmentStatus

# ---------------------------------------------------------------------------
# Version banner
# ---------------------------------------------------------------------------

APP_NAME    = "Company Data Enrichment System"
APP_VERSION = "1.0.0"
APP_AUTHOR  = "Production AI Pipeline"


# ---------------------------------------------------------------------------
# Pipeline state (shared between menu steps in one session)
# ---------------------------------------------------------------------------

@dataclass
class PipelineState:
    """Holds the in-memory state of the pipeline for the current session."""
    raw_df:       Optional[pd.DataFrame] = None   # after load
    cleaned_df:   Optional[pd.DataFrame] = None   # after clean
    enriched_df:  Optional[pd.DataFrame] = None   # after enrich
    validated_df: Optional[pd.DataFrame] = None   # after validate
    report:       Optional[ValidationReport] = None
    load_summary:   dict = field(default_factory=dict)
    enrich_stats:   dict = field(default_factory=dict)
    timings:        dict = field(default_factory=dict)

    # Checkpoint path — so we can resume across sessions
    _checkpoint: Path = settings.enriched_output_file

    def has_cleaned(self) -> bool:
        return self.cleaned_df is not None and not self.cleaned_df.empty

    def has_enriched(self) -> bool:
        return self.enriched_df is not None and not self.enriched_df.empty

    def has_validated(self) -> bool:
        return self.validated_df is not None and self.report is not None

    def try_load_checkpoint(self) -> bool:
        """
        Attempt to resume from a previous run's checkpoint file.
        Returns True if a checkpoint was loaded.
        """
        if self._checkpoint.exists():
            try:
                df = pd.read_csv(self._checkpoint, dtype=str, encoding="utf-8-sig")
                self.enriched_df = df
                console.print(
                    f"[yellow]Checkpoint found:[/yellow] "
                    f"[cyan]{self._checkpoint.name}[/cyan] "
                    f"({len(df)} rows). You can skip to step 3 (Validate) "
                    "or re-run enrichment."
                )
                return True
            except Exception as exc:
                logger.warning("Could not load checkpoint: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _banner() -> None:
    """Print the application banner."""
    banner_text = Text()
    banner_text.append(f"  {APP_NAME}\n", style="bold white")
    banner_text.append(f"  v{APP_VERSION}  |  {APP_AUTHOR}\n", style="dim white")
    banner_text.append(
        "  Scalable · Modular · Production-Ready AI Enrichment Pipeline",
        style="italic cyan",
    )
    console.print(Panel(banner_text, style="bold blue", padding=(1, 4)))


def _menu_table(state: PipelineState) -> Table:
    """Build the interactive menu table with step status indicators."""
    table = Table(
        show_header=False,
        box=None,
        padding=(0, 2),
        style="white",
    )
    table.add_column("Option", style="bold cyan",  width=6)
    table.add_column("Step",   style="white",      width=30)
    table.add_column("Status", style="dim",        width=20)

    def _status(cond: bool, label: str = "Done") -> str:
        return f"[green]* {label}[/green]" if cond else "[dim]—[/dim]"

    input_count = sum(1 for _ in settings.input_dir.glob("**/*.csv"))
    table.add_row(
        "  [1]",
        "Process CSVs (Load + Clean)",
        _status(state.has_cleaned(),
                f"{len(state.cleaned_df)} companies" if state.has_cleaned() else "Done"),
    )
    table.add_row(
        "  [2]",
        "Enrich Data",
        _status(state.has_enriched(),
                f"{len(state.enriched_df)} enriched" if state.has_enriched() else "Done"),
    )
    table.add_row(
        "  [3]",
        "Validate",
        _status(state.has_validated(), "Report ready" if state.has_validated() else "Done"),
    )
    table.add_row(
        "  [4]",
        "Export Results",
        "[dim]— exports CSV + JSON[/dim]",
    )
    table.add_row("  [5]", "Exit", "")

    return table


def _separator(title: str = "") -> None:
    if title:
        console.print(f"\n[bold cyan]{title}[/bold cyan]")
    console.print("-" * 80)


def _success(msg: str) -> None:
    console.print(f"\n[bold green][OK][/bold green]  {msg}")


def _error(msg: str) -> None:
    console.print(f"\n[bold red][FAIL][/bold red]  {msg}")


def _warn(msg: str) -> None:
    console.print(f"\n[bold yellow][!][/bold yellow]  {msg}")


# ---------------------------------------------------------------------------
# Step 1 — Process CSVs (Load + Clean)
# ---------------------------------------------------------------------------

def step_process_csvs(state: PipelineState) -> None:
    """Load all CSVs and clean the resulting DataFrame."""
    _separator("Step 1 · Process CSVs")

    csv_count = sum(1 for _ in settings.input_dir.glob("**/*.csv"))
    if csv_count == 0:
        _error(
            f"No CSV files found in [cyan]{settings.input_dir}[/cyan].\n"
            "  Please add your input CSV files and try again."
        )
        return

    console.print(
        f"Found [bold cyan]{csv_count}[/bold cyan] CSV file(s) in "
        f"[cyan]{settings.input_dir}[/cyan].\n"
    )

    # --- Load ---
    t0 = time.monotonic()
    loader = DataLoader()
    with console.status("[bold blue]Loading CSV files…[/bold blue]", spinner="dots"):
        raw_df = loader.load_all()
    load_elapsed = time.monotonic() - t0

    if raw_df.empty:
        _error("No data could be loaded. Check your CSV files.")
        return

    state.raw_df = raw_df
    state.load_summary = loader.summary()
    state.timings["load"] = load_elapsed

    console.print(
        f"  Loaded [green]{state.load_summary['loaded']}[/green] file(s), "
        f"[red]{state.load_summary['failed']}[/red] failed.  "
        f"Rows: [bold]{len(raw_df)}[/bold]  ({format_elapsed(load_elapsed)})"
    )

    if state.load_summary["failed"]:
        _warn(f"Failed files: {state.load_summary['failed_files']}")

    # --- Clean ---
    t1 = time.monotonic()
    cleaner = DataCleaner()
    with console.status("[bold blue]Cleaning data…[/bold blue]", spinner="dots"):
        cleaned_df = cleaner.clean(raw_df)
    clean_elapsed = time.monotonic() - t1

    state.cleaned_df = cleaned_df
    state.timings["clean"] = clean_elapsed

    removed = len(raw_df) - len(cleaned_df)
    console.print(
        f"  Cleaned: [green]{len(cleaned_df)}[/green] companies remain "
        f"([yellow]{removed}[/yellow] duplicates removed).  "
        f"({format_elapsed(clean_elapsed)})"
    )

    print_dataframe_summary(cleaned_df, title="Cleaned Data — Column Fill Rates")
    _success(
        f"Step 1 complete — [bold]{len(cleaned_df)}[/bold] companies ready for enrichment."
    )
    logger.info(
        "Step 1 complete — loaded: %d, cleaned: %d, removed: %d, "
        "elapsed: %s",
        len(raw_df), len(cleaned_df), removed,
        format_elapsed(load_elapsed + clean_elapsed),
    )


# ---------------------------------------------------------------------------
# Step 2 — Enrich Data
# ---------------------------------------------------------------------------

def step_enrich(state: PipelineState) -> None:
    """Run the scrape + LLM enrichment pipeline."""
    _separator("Step 2 · Enrich Data")

    if not state.has_cleaned():
        _warn("No cleaned data available. Please run Step 1 first.")
        return

    total = len(state.cleaned_df)
    console.print(
        f"  Enriching [bold cyan]{total}[/bold cyan] companies.\n"
        f"  Workers: [cyan]{settings.max_workers}[/cyan]  |  "
        f"LLM enabled: [{'green]Yes' if settings.llm_enabled else 'red]No'}[/{'green' if settings.llm_enabled else 'red'}]\n"
        f"  Batch size: [cyan]{settings.batch_size}[/cyan]  |  "
        f"Skip enriched: [cyan]{settings.skip_already_enriched}[/cyan]\n"
    )

    if not settings.llm_enabled:
        _warn(
            "OPENAI_API_KEY is not set. "
            "Enrichment will run scraping only (no LLM extraction). "
            "Set the key in your .env file to enable AI enrichment."
        )

    if not Confirm.ask("  Start enrichment now?", default=True):
        console.print("  [dim]Enrichment skipped.[/dim]")
        return

    engine = EnrichmentEngine()
    t0 = time.monotonic()

    try:
        with make_progress_bar() as progress:
            task = progress.add_task(
                f"Enriching {total} companies…", total=total
            )

            # We run enrich() synchronously — the engine uses internal threads.
            # Wrap in a simple callback-based approach using a monitor thread.
            import threading

            result_holder: dict = {}
            error_holder:  dict = {}

            def _run() -> None:
                try:
                    result_holder["df"] = engine.enrich(state.cleaned_df)
                except Exception as exc:
                    error_holder["exc"] = exc

            thread = threading.Thread(target=_run, daemon=True)
            thread.start()

            last_processed = 0
            while thread.is_alive():
                time.sleep(0.5)
                stats = engine.stats
                delta = stats["processed"] - last_processed
                if delta > 0:
                    progress.advance(task, delta)
                    last_processed = stats["processed"]

            thread.join()

            # Final advance for any remainder
            stats = engine.stats
            remaining = total - last_processed
            if remaining > 0:
                progress.advance(task, remaining)

    except KeyboardInterrupt:
        _warn("Enrichment interrupted by user. Partial results may be saved.")
        engine.close()
        return

    if "exc" in error_holder:
        _error(f"Enrichment failed: {error_holder['exc']}")
        logger.error("Enrichment error: %s", error_holder["exc"], exc_info=True)
        engine.close()
        return

    enriched_df = result_holder.get("df")
    if enriched_df is None or enriched_df.empty:
        _error("Enrichment returned no results.")
        engine.close()
        return

    elapsed = time.monotonic() - t0
    state.enriched_df = enriched_df
    state.enrich_stats = engine.stats
    state.timings["enrich"] = elapsed
    engine.close()

    # Quick status summary
    if "status" in enriched_df.columns:
        sc = enriched_df["status"].value_counts().to_dict()
        table = Table(title="Enrichment Results", show_header=True, header_style="bold magenta")
        table.add_column("Status",    style="cyan")
        table.add_column("Count",     style="white", justify="right")
        table.add_column("% of total", style="yellow", justify="right")
        for status, count in sc.items():
            pct = f"{100 * count / len(enriched_df):.1f}%"
            table.add_row(status, str(count), pct)
        console.print(table)

    _success(
        f"Step 2 complete — enriched [bold]{len(enriched_df)}[/bold] companies "
        f"in [bold]{format_elapsed(elapsed)}[/bold]."
    )
    logger.info(
        "Step 2 complete — enriched: %d | elapsed: %s | stats: %s",
        len(enriched_df), format_elapsed(elapsed), engine.stats,
    )


# ---------------------------------------------------------------------------
# Step 3 — Validate
# ---------------------------------------------------------------------------

def step_validate(state: PipelineState) -> None:
    """Validate the enriched DataFrame and produce the validation report."""
    _separator("Step 3 · Validate")

    # Allow validating from checkpoint even if step 2 wasn't run this session
    target_df = state.enriched_df
    if target_df is None or target_df.empty:
        if state._checkpoint.exists():
            _warn(
                "No in-memory enriched data. Loading from checkpoint: "
                f"[cyan]{state._checkpoint.name}[/cyan]"
            )
            try:
                target_df = pd.read_csv(
                    state._checkpoint, dtype=str, encoding="utf-8-sig"
                )
                state.enriched_df = target_df
            except Exception as exc:
                _error(f"Could not load checkpoint: {exc}")
                return
        else:
            _warn("No enriched data available. Please run Step 2 first.")
            return

    console.print(
        f"  Validating [bold cyan]{len(target_df)}[/bold cyan] records…\n"
    )

    t0 = time.monotonic()
    validator = DataValidator()

    with console.status("[bold blue]Running validation checks…[/bold blue]", spinner="dots"):
        validated_df, report = validator.validate(target_df)

    elapsed = time.monotonic() - t0
    state.validated_df = validated_df
    state.report = report
    state.timings["validate"] = elapsed

    # Print summary table
    summary_table = Table(
        title="Validation Summary", show_header=True, header_style="bold magenta"
    )
    summary_table.add_column("Metric",  style="cyan")
    summary_table.add_column("Value",   style="white", justify="right")

    summary_table.add_row("Total Companies",       str(report.total_companies))
    summary_table.add_row("Valid Records",         f"[green]{report.total_valid}[/green]")
    summary_table.add_row("Invalid Records",       f"[red]{report.total_invalid}[/red]")
    summary_table.add_row("Total Warnings",        f"[yellow]{report.total_warnings}[/yellow]")
    summary_table.add_row("Duplicate Domains",     str(report.duplicate_domains_found))
    summary_table.add_row("Duplicate Names (fuzzy)", str(report.duplicate_names_found))
    summary_table.add_row("Status: FOUND",         f"[green]{report.status_found}[/green]")
    summary_table.add_row("Status: PARTIALLY",     f"[yellow]{report.status_partially_found}[/yellow]")
    summary_table.add_row("Status: NOT_FOUND",     f"[red]{report.status_not_found}[/red]")
    summary_table.add_row("Status: NEEDS_REVIEW",  f"[magenta]{report.status_needs_review}[/magenta]")
    summary_table.add_row("Avg Confidence Score",  f"{report.average_confidence_score:.1f}")
    summary_table.add_row("Min / Max Score",       f"{report.min_confidence_score} / {report.max_confidence_score}")
    summary_table.add_row("Records with CEO",      str(report.fields_with_ceo))
    summary_table.add_row("Records with Founder",  str(report.fields_with_founder))
    console.print(summary_table)

    # Top issues
    if report.issue_counts:
        issue_table = Table(
            title="Top Issues", show_header=True, header_style="bold red"
        )
        issue_table.add_column("Issue Code", style="red")
        issue_table.add_column("Count", style="white", justify="right")
        for code, count in sorted(
            report.issue_counts.items(), key=lambda x: -x[1]
        )[:10]:
            issue_table.add_row(code, str(count))
        console.print(issue_table)

    # Save report
    validator.save_report(report)
    DataValidator.log_summary(report)

    _success(
        f"Step 3 complete — validation report saved to "
        f"[cyan]{settings.validation_report_file.name}[/cyan]  "
        f"({format_elapsed(elapsed)})"
    )
    logger.info(
        "Step 3 complete — valid: %d | invalid: %d | elapsed: %s",
        report.total_valid, report.total_invalid, format_elapsed(elapsed),
    )


# ---------------------------------------------------------------------------
# Step 4 — Export Results
# ---------------------------------------------------------------------------

def step_export(state: PipelineState) -> None:
    """Export the final enriched CSV and validation JSON."""
    _separator("Step 4 · Export Results")

    export_df = state.validated_df
    if export_df is None:
        export_df = state.enriched_df
    if export_df is None:
        export_df = state.cleaned_df

    if export_df is None or export_df.empty:
        _warn("No data available to export. Please run at least Step 1 first.")
        return

    t0 = time.monotonic()

    # Build final output with standard columns ordered correctly
    output_df = select_standard_columns(export_df)

    # Write enriched CSV
    csv_path = export_dataframe(
        df=output_df,
        output_path=settings.enriched_output_file,
        drop_internal=True,
    )

    # Write validation report if available
    json_path: Optional[Path] = None
    if state.report:
        validator = DataValidator()
        json_path = validator.save_report(state.report)

    elapsed = time.monotonic() - t0
    state.timings["export"] = elapsed

    # Summary
    console.print(f"\n  [bold green]Files written:[/bold green]")
    console.print(f"    📄  [cyan]{csv_path}[/cyan]  ({len(output_df)} rows)")
    if json_path:
        console.print(f"    📋  [cyan]{json_path}[/cyan]")
    console.print(f"    📝  [cyan]{settings.log_file}[/cyan]")

    _success(f"Step 4 complete — all files exported ({format_elapsed(elapsed)}).")
    logger.info(
        "Step 4 complete — exported %d rows → %s | elapsed: %s",
        len(output_df), csv_path, format_elapsed(elapsed),
    )


# ---------------------------------------------------------------------------
# Session timing summary
# ---------------------------------------------------------------------------

def _print_session_summary(state: PipelineState) -> None:
    """Print a summary of elapsed time for each step."""
    if not state.timings:
        return
    _separator("Session Summary")
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Step",    style="cyan")
    table.add_column("Elapsed", style="green", justify="right")
    step_labels = {
        "load":     "1a. Load CSVs",
        "clean":    "1b. Clean Data",
        "enrich":   "2.  Enrich Data",
        "validate": "3.  Validate",
        "export":   "4.  Export",
    }
    total_elapsed = 0.0
    for key, elapsed in state.timings.items():
        table.add_row(step_labels.get(key, key), format_elapsed(elapsed))
        total_elapsed += elapsed
    table.add_row("[bold]Total[/bold]", f"[bold]{format_elapsed(total_elapsed)}[/bold]")
    console.print(table)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    """Interactive CLI menu loop."""
    _banner()

    state = PipelineState()
    state.try_load_checkpoint()

    # Pre-flight environment check
    console.print()
    print_environment_check()
    console.print()

    while True:
        _separator()
        console.print(
            Panel(
                _menu_table(state),
                title="[bold blue]Main Menu[/bold blue]",
                subtitle=f"[dim]Input dir: {settings.input_dir}  |  "
                         f"Output dir: {settings.output_dir}[/dim]",
                padding=(1, 2),
                style="blue",
            )
        )

        choice = Prompt.ask(
            "\n  [bold cyan]Select an option[/bold cyan]",
            choices=["1", "2", "3", "4", "5"],
            default="1",
        ).strip()

        console.print()

        if choice == "1":
            step_process_csvs(state)

        elif choice == "2":
            step_enrich(state)

        elif choice == "3":
            step_validate(state)

        elif choice == "4":
            step_export(state)

        elif choice == "5":
            _print_session_summary(state)
            console.print(
                "\n[bold blue]Thank you for using the Company Data Enrichment System.[/bold blue]\n"
            )
            logger.info("Application exited by user.")
            sys.exit(0)

        console.print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n\n[bold yellow]Interrupted by user.[/bold yellow]")
        logger.info("Application interrupted (KeyboardInterrupt).")
        sys.exit(0)
    except Exception as exc:
        logger.critical("Unhandled exception in main: %s", exc, exc_info=True)
        console.print(f"\n[bold red]Fatal error:[/bold red] {exc}")
        sys.exit(1)
