"""
modules/utils.py
================
Shared utilities for the Company Data Enrichment System.

Contents:
    - Logging setup (Rich-formatted console + rotating file handler)
    - CSV export with safe encoding
    - Timer / elapsed-time decorator
    - Progress display (Rich live table)
    - Directory / path helpers
    - DataFrame inspection helpers
    - Singleton guard (prevent duplicate initialisation)

Import from here in all other modules — never duplicate utility code.
"""

from __future__ import annotations

import csv
import functools
import logging
import logging.handlers
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

import pandas as pd
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from config import settings

# ---------------------------------------------------------------------------
# Module-level console (shared Rich console for all output)
# ---------------------------------------------------------------------------

console = Console(highlight=True, markup=True, emoji=False)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

_LOGGING_INITIALISED: bool = False


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    max_bytes: int = 10 * 1024 * 1024,   # 10 MB per file
    backup_count: int = 5,
) -> logging.Logger:
    """
    Configure the root logger with:
        - Rich-formatted, colour-coded console handler
        - Rotating file handler (logs/app.log)

    Call once at application startup (main.py). Safe to call multiple
    times — duplicate handlers are not added.

    Parameters
    ----------
    log_level : str
        Logging level name ('DEBUG', 'INFO', 'WARNING', 'ERROR').
    log_file : Path, optional
        Path to the log file. Defaults to settings.log_file.
    max_bytes : int
        Maximum size of each log file before rotation.
    backup_count : int
        Number of rotated log files to keep.

    Returns
    -------
    logging.Logger
        The configured root logger.
    """
    global _LOGGING_INITIALISED

    log_file = log_file or settings.log_file
    log_file.parent.mkdir(parents=True, exist_ok=True)

    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    root_logger = logging.getLogger()

    # Guard against duplicate handlers on repeated calls
    if _LOGGING_INITIALISED:
        return root_logger

    root_logger.setLevel(numeric_level)

    # --- Rich console handler ---
    rich_handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        tracebacks_show_locals=False,
        show_time=True,
        show_level=True,
        show_path=False,
        markup=True,
    )
    rich_handler.setLevel(numeric_level)
    rich_format = logging.Formatter("%(message)s", datefmt="[%X]")
    rich_handler.setFormatter(rich_format)

    # --- Rotating file handler ---
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(numeric_level)
    file_format = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_format)

    root_logger.addHandler(rich_handler)
    root_logger.addHandler(file_handler)

    # Quiet down noisy third-party loggers
    for noisy in ("urllib3", "httpx", "openai", "httpcore", "charset_normalizer"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _LOGGING_INITIALISED = True
    root_logger.info(
        "Logging initialised — level: %s | file: %s", log_level, log_file
    )
    return root_logger


# ---------------------------------------------------------------------------
# Timer decorator
# ---------------------------------------------------------------------------

F = TypeVar("F", bound=Callable[..., Any])


def timed(label: Optional[str] = None) -> Callable[[F], F]:
    """
    Decorator that logs how long a function took to execute.

    Usage:
        @timed("Data loading")
        def load():
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            name = label or func.__qualname__
            logger_ = logging.getLogger(func.__module__)
            start = time.monotonic()
            try:
                result = func(*args, **kwargs)
                elapsed = time.monotonic() - start
                logger_.info("[timer] %s completed in %.2fs", name, elapsed)
                return result
            except Exception:
                elapsed = time.monotonic() - start
                logger_.error("[timer] %s FAILED after %.2fs", name, elapsed)
                raise
        return wrapper  # type: ignore[return-value]
    return decorator


# ---------------------------------------------------------------------------
# Progress bar (Rich)
# ---------------------------------------------------------------------------

def make_progress_bar() -> Progress:
    """
    Create a Rich Progress bar pre-configured for the enrichment pipeline.

    Usage:
        with make_progress_bar() as progress:
            task = progress.add_task("Enriching…", total=1000)
            for company in companies:
                process(company)
                progress.advance(task)
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        refresh_per_second=10,
    )


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_dataframe(
    df: pd.DataFrame,
    output_path: Path,
    columns: Optional[list[str]] = None,
    drop_internal: bool = True,
) -> Path:
    """
    Export a DataFrame to CSV with safe encoding (UTF-8 BOM for Excel compat).

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to export.
    output_path : Path
        Destination file path.
    columns : list[str], optional
        Ordered list of columns to include. If None, uses all standard
        output columns present in the DataFrame.
    drop_internal : bool
        If True (default), drop columns whose names start with '_'.
        These are pipeline-internal columns not needed in the final export.

    Returns
    -------
    Path
        The path to the written file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = df.copy()

    # Drop internal columns unless explicitly keeping them
    if drop_internal:
        internal_cols = [c for c in df.columns if c.startswith("_")]
        df = df.drop(columns=internal_cols, errors="ignore")

    # Select and order columns
    if columns:
        present = [c for c in columns if c in df.columns]
        extra   = [c for c in df.columns if c not in columns]
        df = df[present + extra]

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",       # BOM makes Excel open it correctly
        quoting=csv.QUOTE_MINIMAL,
        na_rep="",
    )

    logger = logging.getLogger(__name__)
    logger.info("Exported %d rows -> %s", len(df), output_path)
    return output_path


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def ensure_dirs(*paths: Path) -> None:
    """Create all given directories (and parents) if they don't exist."""
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def count_csv_files(directory: Path) -> int:
    """Return the number of *.csv files recursively under ``directory``."""
    if not directory.exists():
        return 0
    return sum(1 for _ in directory.glob("**/*.csv"))


def get_output_path(filename: str) -> Path:
    """Return a full path inside the configured output directory."""
    return settings.output_dir / filename


# ---------------------------------------------------------------------------
# DataFrame inspection helpers
# ---------------------------------------------------------------------------

def dataframe_summary(df: pd.DataFrame) -> dict:
    """
    Return a concise summary dict describing a DataFrame's shape,
    column nullness, and data types.
    """
    total = len(df)
    summary = {
        "rows": total,
        "columns": len(df.columns),
        "column_info": {},
    }
    for col in df.columns:
        non_null = int(df[col].apply(
            lambda v: v is not None
            and not (isinstance(v, float) and pd.isna(v))
            and str(v).strip() not in {"", "nan", "None"}
        ).sum())
        summary["column_info"][col] = {
            "non_null": non_null,
            "null": total - non_null,
            "fill_pct": round(100 * non_null / total, 1) if total else 0,
            "dtype": str(df[col].dtype),
        }
    return summary


def print_dataframe_summary(df: pd.DataFrame, title: str = "DataFrame Summary") -> None:
    """Print a Rich table summarising the DataFrame to the console."""
    summary = dataframe_summary(df)
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Column",   style="cyan",  no_wrap=True)
    table.add_column("Non-Null", style="green", justify="right")
    table.add_column("Null",     style="red",   justify="right")
    table.add_column("Fill %",   style="yellow", justify="right")
    table.add_column("Dtype",    style="white")

    for col, info in summary["column_info"].items():
        table.add_row(
            col,
            str(info["non_null"]),
            str(info["null"]),
            f"{info['fill_pct']}%",
            info["dtype"],
        )

    console.print(f"\n[bold]Rows:[/bold] {summary['rows']}  "
                  f"[bold]Columns:[/bold] {summary['columns']}")
    console.print(table)


# ---------------------------------------------------------------------------
# Column selection helpers
# ---------------------------------------------------------------------------

from modules.data_loader import STANDARD_COLUMNS   # noqa: E402 (circular-safe here)


def select_standard_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of the DataFrame with only the standard output schema
    columns, in the correct order. Missing columns are added as empty strings.
    """
    for col in STANDARD_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[STANDARD_COLUMNS].copy()


# ---------------------------------------------------------------------------
# Safe string coercion (re-exported here so callers only need one import)
# ---------------------------------------------------------------------------

def safe_str(value: Any, default: str = "") -> str:
    """
    Convert any value to a clean string.
    Returns ``default`` for None / NaN / blank / NA-like values.
    """
    if value is None:
        return default
    if isinstance(value, float) and pd.isna(value):
        return default
    s = str(value).strip()
    return default if s.lower() in {"nan", "none", "null", "n/a", "na", ""} else s


# ---------------------------------------------------------------------------
# Environment / startup checks
# ---------------------------------------------------------------------------

def check_environment() -> dict[str, bool]:
    """
    Perform pre-flight checks on the environment.

    Returns a dict of {check_name: passed} for display in the CLI menu.
    """
    checks: dict[str, bool] = {}

    # OpenAI key present
    checks["openai_api_key_set"] = bool(settings.openai_api_key)

    # Input directory has CSV files
    checks["input_csvs_present"] = count_csv_files(settings.input_dir) > 0

    # Output directory writable
    try:
        test_file = settings.output_dir / ".write_test"
        test_file.touch()
        test_file.unlink()
        checks["output_dir_writable"] = True
    except OSError:
        checks["output_dir_writable"] = False

    # Log directory writable
    try:
        test_file = settings.log_dir / ".write_test"
        test_file.touch()
        test_file.unlink()
        checks["log_dir_writable"] = True
    except OSError:
        checks["log_dir_writable"] = False

    # Python version >= 3.10
    checks["python_version_ok"] = sys.version_info >= (3, 10)

    return checks


def print_environment_check() -> None:
    """Print a Rich table of environment pre-flight results."""
    checks = check_environment()
    table = Table(title="Environment Check", show_header=True, header_style="bold blue")
    table.add_column("Check",   style="cyan",  no_wrap=True)
    table.add_column("Status",  justify="center")

    labels = {
        "openai_api_key_set":   "OpenAI API Key set",
        "input_csvs_present":   "Input CSVs present",
        "output_dir_writable":  "Output directory writable",
        "log_dir_writable":     "Log directory writable",
        "python_version_ok":    "Python >= 3.10",
    }

    all_ok = True
    for key, passed in checks.items():
        label  = labels.get(key, key)
        status = "[bold green]OK[/bold green]" if passed else "[bold red]FAIL[/bold red]"
        table.add_row(label, status)
        if not passed:
            all_ok = False

    console.print(table)
    if not all_ok:
        console.print(
            "\n[bold yellow][!] Some checks failed. "
            "Review the issues above before running the pipeline.[/bold yellow]"
        )
    else:
        console.print("\n[bold green][OK] All environment checks passed.[/bold green]")


# ---------------------------------------------------------------------------
# Elapsed time formatter
# ---------------------------------------------------------------------------

def format_elapsed(seconds: float) -> str:
    """Convert a float seconds value to a human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours, mins = divmod(minutes, 60)
    return f"{hours}h {mins}m {secs}s"
