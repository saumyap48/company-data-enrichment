"""
modules/enrichment.py
=====================
Orchestration engine for the Company Data Enrichment System.

This module ties together every other module:
    DataLoader  → DataCleaner → WebScraper → LLMExtractor
    → ConfidenceScorer → output DataFrame

Processing pipeline (per company):
    1.  Determine the best identifier (domain › LinkedIn › name)
    2.  Scrape the company website (cached, retried)
    3.  Feed scraped text to the LLM extractor
    4.  Merge LLM results back into the record
    5.  Collect social signals found during scraping
    6.  Apply confidence scoring and status
    7.  Update last_updated timestamp

Performance features:
    - ThreadPoolExecutor for parallel company processing
    - One shared WebScraper (disk-cached, pooled sessions per thread)
    - One shared LLMExtractor (OpenAI client is thread-safe)
    - Domain-level deduplication cache (never enrich the same domain twice)
    - Batch checkpointing (saves progress to disk every N records)
    - Rate-limiting between scrape requests

Public API:
    engine = EnrichmentEngine()
    enriched_df = engine.enrich(df)
"""

from __future__ import annotations

import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import diskcache
import pandas as pd

from config import settings
from modules.cleaner import _to_str
from modules.confidence import ConfidenceScorer, DataSource, EnrichmentStatus
from modules.llm_extractor import CompanyExtraction, LLMExtractor
from modules.scraper import ScrapeResult, WebScraper

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# How often (in records processed) to checkpoint progress to disk
_CHECKPOINT_EVERY: int = 50

# Minimum delay between requests to the same domain (seconds)
_INTER_REQUEST_DELAY: float = 1.0 / max(settings.requests_per_second, 0.1)

# Columns written back from enrichment (never overwrite non-empty existing values)
_ENRICHMENT_WRITE_COLS: list[str] = [
    "description", "industry", "size", "ceo", "founder",
    "email", "phone", "facebook", "twitter", "instagram", "linkedin",
    "source_used", "confidence_score", "status", "last_updated",
]


# ---------------------------------------------------------------------------
# Domain-level enrichment cache
# ---------------------------------------------------------------------------

class _DomainCache:
    """
    Thread-safe persistent cache of already-enriched domain results using diskcache.
    Prevents enriching the same domain twice across runs.
    """

    def __init__(self) -> None:
        self._cache = diskcache.Cache(
            directory=str(settings.cache_dir / "enrichment_domain_cache"),
            timeout=86_400 * 7,  # 7 days
        )

    def get(self, domain: str) -> Optional[dict]:
        return self._cache.get(domain.lower().strip())

    def set(self, domain: str, data: dict) -> None:
        self._cache.set(domain.lower().strip(), data)

    def __len__(self) -> int:
        return len(self._cache)
    
    def close(self) -> None:
        self._cache.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _best_url(record: dict) -> tuple[str, str]:
    """
    Choose the best URL/identifier to enrich a company.

    Priority: website domain > LinkedIn URL > company name (web search fallback)

    Returns (url_to_fetch, source_label)
    """
    website  = _to_str(record.get("website", ""))
    linkedin = _to_str(record.get("linkedin", ""))

    if website:
        return website, DataSource.OFFICIAL_WEBSITE.value
    if linkedin:
        return linkedin, DataSource.LINKEDIN.value
    return "", DataSource.NOT_FOUND.value


def _merge_field(existing, new_value) -> object:
    """
    Return the best value between existing and new.
    Prefers non-empty existing values (idempotent enrichment).
    """
    if _to_str(existing):
        return existing           # keep existing data
    if _to_str(str(new_value) if new_value is not None else ""):
        return new_value
    return existing


def _now_iso() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalise_domain_for_cache(url: str) -> str:
    """Strip scheme and www for use as a cache key."""
    url = url.lower().strip()
    url = re.sub(r"^https?://", "", url)
    url = re.sub(r"^www\.", "", url)
    return url.split("/")[0]


# ---------------------------------------------------------------------------
# Single-record enrichment
# ---------------------------------------------------------------------------

class _RecordEnricher:
    """
    Enriches one company record.
    Designed to be called inside a thread-pool worker.
    """

    def __init__(
        self,
        scraper: WebScraper,
        llm: LLMExtractor,
        scorer: ConfidenceScorer,
        domain_cache: _DomainCache,
    ) -> None:
        self._scraper = scraper
        self._llm     = llm
        self._scorer  = scorer
        self._cache   = domain_cache

    def enrich(self, record: dict) -> dict:
        """
        Run the full enrichment pipeline on one record dict.
        Returns an updated copy of the record.
        """
        record = dict(record)   # shallow copy — never mutate caller's dict
        company_name = _to_str(record.get("name", ""))

        # --- Determine URL to fetch ---
        url, source_label = _best_url(record)
        cache_key = _normalise_domain_for_cache(url) if url else ""

        # --- Check domain cache (avoid re-enriching same domain) ---
        if cache_key:
            cached = self._cache.get(cache_key)
            if cached is not None:
                logger.debug(
                    "Domain cache hit for '%s' (%s).", company_name, cache_key
                )
                record = self._apply_enrichment(record, cached, source_label)
                record["last_updated"] = _now_iso()
                return record

        # --- Scrape ---
        scrape_result: Optional[ScrapeResult] = None
        if url:
            try:
                scrape_result = self._scraper.scrape(url)
                time.sleep(_INTER_REQUEST_DELAY)   # polite rate-limit
            except Exception as exc:
                logger.warning(
                    "Scrape failed for '%s' (%s): %s", company_name, url, exc
                )

        # --- LLM extraction ---
        llm_result: CompanyExtraction = CompanyExtraction()
        if scrape_result and scrape_result.success:
            # Handle redirect: update website domain if it changed (preserves original name)
            if scrape_result.is_redirect and scrape_result.final_url:
                record["website"] = scrape_result.final_url
            
            try:
                llm_result = self._llm.extract(
                    website_text=scrape_result.visible_text,
                    company_name=company_name,
                    about_text=scrape_result.about_text,
                    contact_text=scrape_result.contact_text,
                    team_text=scrape_result.team_text,
                )
            except Exception as exc:
                logger.warning(
                    "LLM extraction failed for '%s': %s", company_name, exc
                )

        # --- Build enrichment payload ---
        enrichment = self._build_enrichment_payload(
            record, scrape_result, llm_result, source_label
        )

        # --- Store in domain cache ---
        if cache_key:
            self._cache.set(cache_key, enrichment)

        # --- Apply enrichment to record ---
        record = self._apply_enrichment(record, enrichment, source_label)

        # --- Score ---
        score, status, flags = self._scorer.score(record)
        record["confidence_score"] = score
        record["status"]           = status.value
        record["_review_flags"]    = ", ".join(f.value for f in flags)
        record["last_updated"]     = _now_iso()

        logger.info(
            "Enriched '%s' — status: %s, score: %d",
            company_name, status.value, score,
        )
        return record

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_enrichment_payload(
        self,
        record: dict,
        scrape: Optional[ScrapeResult],
        llm: CompanyExtraction,
        source_label: str,
    ) -> dict:
        """
        Combine scrape signals + LLM output into a single enrichment dict.
        This dict is what gets stored in the domain cache and merged into
        the record.
        """
        payload: dict = {
            "source_used": source_label,
        }

        # --- From LLM ---
        if not llm.is_empty():
            payload.update({
                "description": llm.description,
                "industry":    llm.industry,
                "size":        llm.company_size,
                "ceo":         llm.ceo,
                "founder":     llm.founder,
            })
            # Blend LLM confidence into source label
            payload["source_used"] = (
                f"{source_label},{DataSource.LLM_INFERRED.value}"
            )

        # --- From scraper (social signals, email, phone) ---
        if scrape and scrape.success:
            # Social links — only fill if scraper found them
            if scrape.linkedin_found:
                payload["linkedin"]  = scrape.linkedin_found
            if scrape.facebook_found:
                payload["facebook"]  = scrape.facebook_found
            if scrape.twitter_found:
                payload["twitter"]   = scrape.twitter_found
            if scrape.instagram_found:
                payload["instagram"] = scrape.instagram_found

            # Contacts — take first email/phone found on page
            if scrape.emails_found:
                payload["email"] = scrape.emails_found[0]
            if scrape.phones_found:
                payload["phone"] = scrape.phones_found[0]

            # If we don't have a description yet, use meta/OG description
            if not payload.get("description") and scrape.best_description():
                payload["description"] = scrape.best_description()
                payload["source_used"] = (
                    f"{source_label},{DataSource.WEB_SCRAPE.value}"
                )

        return payload

    def _apply_enrichment(
        self, record: dict, enrichment: dict, source_label: str
    ) -> dict:
        """
        Write enrichment values into the record, respecting the rule:
        never overwrite an already-populated field.
        """
        for col, new_val in enrichment.items():
            if col.startswith("_"):
                continue   # skip internal keys
            if col in record:
                record[col] = _merge_field(record[col], new_val)
            else:
                record[col] = new_val

        # Always update source_used to reflect what we used this run
        existing_source = _to_str(record.get("source_used", ""))
        new_source      = _to_str(enrichment.get("source_used", source_label))
        if new_source and new_source not in existing_source:
            record["source_used"] = (
                f"{existing_source},{new_source}".strip(",")
                if existing_source else new_source
            )
        return record


# ---------------------------------------------------------------------------
# Checkpoint manager
# ---------------------------------------------------------------------------

class _Checkpointer:
    """Saves partial results to disk so long runs can be resumed."""

    def __init__(self, output_path: Path) -> None:
        self._path = output_path
        self._lock = threading.Lock()

    def save(self, df: pd.DataFrame) -> None:
        with self._lock:
            try:
                df.to_csv(self._path, index=False, encoding="utf-8-sig")
                logger.debug("Checkpoint saved → %s (%d rows)", self._path, len(df))
            except Exception as exc:
                logger.warning("Checkpoint save failed: %s", exc)


# ---------------------------------------------------------------------------
# EnrichmentEngine
# ---------------------------------------------------------------------------

class EnrichmentEngine:
    """
    Orchestrates the full enrichment pipeline for all company records.

    Parameters
    ----------
    max_workers : int, optional
        ThreadPoolExecutor threads. Defaults to settings.max_workers.
    batch_size : int, optional
        Records per checkpoint flush. Defaults to settings.batch_size.
    skip_enriched : bool, optional
        Skip rows already marked FOUND. Defaults to settings.skip_already_enriched.
    """

    def __init__(
        self,
        max_workers: Optional[int] = None,
        batch_size: Optional[int] = None,
        skip_enriched: Optional[bool] = None,
    ) -> None:
        self._max_workers   = max_workers   or settings.max_workers
        self._batch_size    = batch_size    or settings.batch_size
        self._skip_enriched = skip_enriched if skip_enriched is not None \
                              else settings.skip_already_enriched

        # Shared resources (thread-safe)
        self._scraper      = WebScraper()
        self._llm          = LLMExtractor()
        self._scorer       = ConfidenceScorer()
        self._domain_cache = _DomainCache()
        self._checkpointer = _Checkpointer(settings.enriched_output_file)

        # Stats counters (protected by lock)
        self._lock           = threading.Lock()
        self._processed      = 0
        self._succeeded      = 0
        self._failed         = 0
        self._skipped        = 0
        self._start_time: Optional[float] = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich every company in the DataFrame.

        Workflow:
            1. Partition into "to enrich" vs "already enriched" (skip_enriched)
            2. Submit each record to the thread pool
            3. Collect results, checkpoint every _batch_size records
            4. Reassemble and return the final DataFrame

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned company DataFrame (output of DataCleaner.clean).

        Returns
        -------
        pd.DataFrame
            Same structure with enrichment columns populated.
        """
        self._start_time = time.monotonic()
        total = len(df)
        logger.info(
            "EnrichmentEngine starting — %d records, %d workers, batch=%d.",
            total, self._max_workers, self._batch_size,
        )

        df = df.copy()

        # --- Partition ---
        if self._skip_enriched:
            skip_mask = df["status"].astype(str).str.strip() == EnrichmentStatus.FOUND.value
            skip_df   = df[skip_mask].copy()
            work_df   = df[~skip_mask].copy()
            self._skipped = len(skip_df)
            logger.info(
                "Skipping %d already-FOUND records. Enriching %d.",
                self._skipped, len(work_df),
            )
        else:
            skip_df = pd.DataFrame(columns=df.columns)
            work_df = df

        if work_df.empty:
            logger.info("Nothing to enrich.")
            return df

        # --- Convert to list of dicts for thread-pool ---
        records: list[dict] = work_df.to_dict(orient="records")
        enriched_records: list[dict] = [None] * len(records)   # type: ignore

        # --- Thread pool ---
        record_enricher = _RecordEnricher(
            scraper=self._scraper,
            llm=self._llm,
            scorer=self._scorer,
            domain_cache=self._domain_cache,
        )

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            # Submit all futures, preserving original index
            future_to_idx = {
                executor.submit(self._safe_enrich, record_enricher, rec): i
                for i, rec in enumerate(records)
            }

            completed_since_last_checkpoint = 0

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    enriched_records[idx] = future.result()
                    with self._lock:
                        self._succeeded += 1
                except Exception as exc:
                    logger.error(
                        "Enrichment failed for record %d: %s", idx, exc, exc_info=True
                    )
                    enriched_records[idx] = records[idx]   # keep original on failure
                    with self._lock:
                        self._failed += 1

                with self._lock:
                    self._processed += 1
                    completed_since_last_checkpoint += 1

                # --- Progress log ---
                with self._lock:
                    proc = self._processed
                if proc % 10 == 0 or proc == len(records):
                    elapsed = time.monotonic() - (self._start_time or time.monotonic())
                    rate = proc / elapsed if elapsed > 0 else 0
                    logger.info(
                        "Progress: %d/%d (%.1f%%) | %.1f rec/s | "
                        "ok=%d fail=%d skip=%d",
                        proc, len(records),
                        100 * proc / len(records),
                        rate, self._succeeded, self._failed, self._skipped,
                    )

                # --- Checkpoint ---
                if completed_since_last_checkpoint >= _CHECKPOINT_EVERY:
                    completed_since_last_checkpoint = 0
                    self._checkpoint(enriched_records, work_df, skip_df)

        # --- Assemble final DataFrame ---
        enriched_df = pd.DataFrame(
            [r for r in enriched_records if r is not None],
            columns=work_df.columns.tolist(),
        )

        # Re-attach any extra columns that appeared during enrichment
        extra_cols = [c for c in enriched_df.columns if c not in work_df.columns]
        for col in extra_cols:
            if col not in skip_df.columns:
                skip_df[col] = pd.NA

        final_df = pd.concat([skip_df, enriched_df], ignore_index=True, sort=False)

        elapsed = time.monotonic() - (self._start_time or 0)
        logger.info(
            "EnrichmentEngine complete — total: %d | enriched: %d | "
            "failed: %d | skipped: %d | elapsed: %.1fs",
            total, self._succeeded, self._failed, self._skipped, elapsed,
        )

        # Final checkpoint
        self._checkpoint(enriched_records, work_df, skip_df)

        return final_df

    # ------------------------------------------------------------------
    # Thread-safe wrapper around _RecordEnricher.enrich
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_enrich(enricher: _RecordEnricher, record: dict) -> dict:
        """
        Wraps _RecordEnricher.enrich in a try/except so a single bad
        record never kills the thread-pool worker.
        """
        try:
            return enricher.enrich(record)
        except Exception as exc:
            company = record.get("name", record.get("company_id", "?"))
            logger.error("Unhandled error enriching '%s': %s", company, exc, exc_info=True)
            record["status"] = EnrichmentStatus.NOT_FOUND.value
            record["confidence_score"] = 0
            record["last_updated"] = _now_iso()
            return record

    # ------------------------------------------------------------------
    # Checkpoint helper
    # ------------------------------------------------------------------

    def _checkpoint(
        self,
        enriched_records: list,
        work_df: pd.DataFrame,
        skip_df: pd.DataFrame,
    ) -> None:
        """Flush current progress to disk."""
        valid = [r for r in enriched_records if r is not None]
        if not valid:
            return
        partial = pd.DataFrame(valid, columns=work_df.columns.tolist())
        combined = pd.concat([skip_df, partial], ignore_index=True, sort=False)
        self._checkpointer.save(combined)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def stats(self) -> dict:
        """Return a summary of processing statistics."""
        with self._lock:
            elapsed = (
                time.monotonic() - self._start_time
                if self._start_time else 0.0
            )
            return {
                "processed": self._processed,
                "succeeded": self._succeeded,
                "failed":    self._failed,
                "skipped":   self._skipped,
                "elapsed_seconds": round(elapsed, 2),
                "domain_cache_size": len(self._domain_cache),
            }

    def close(self) -> None:
        """Release resources."""
        self._scraper.close()
        self._domain_cache.close()
