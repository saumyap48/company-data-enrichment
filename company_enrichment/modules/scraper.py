"""
modules/scraper.py
==================
HTTP scraping layer for the Company Data Enrichment System.

Responsibilities:
    - Fetch raw HTML from any URL with timeouts and retries
    - Respect robots.txt (configurable via settings)
    - Rotate User-Agent headers to reduce bot-detection
    - Follow and detect HTTP redirects (captures final URL)
    - Extract structured page sections:
        • Page title
        • Meta description
        • Open Graph description
        • Visible body text (cleaned, no scripts/styles)
        • About-page content
        • Contact-page content
    - Connection pooling via requests.Session (reuse TCP connections)
    - Persistent domain-level cache (diskcache) — avoids re-fetching

Public API:
    scraper = WebScraper()
    result  = scraper.scrape("example.com")
    print(result.title, result.meta_description, result.visible_text)
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
import urllib.parse
import urllib.robotparser
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urljoin, urlparse

import diskcache
import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Common paths checked when hunting for an "About" page
_ABOUT_PATHS: list[str] = [
    "/about", "/about-us", "/about_us", "/company", "/who-we-are",
    "/our-story", "/overview", "/mission", "/vision",
]

# Common paths checked for a "Contact" page
_CONTACT_PATHS: list[str] = [
    "/contact", "/contact-us", "/contact_us", "/get-in-touch",
    "/reach-us", "/support", "/help",
]

# Common paths checked for a "Leadership/Team" page
_LEADERSHIP_PATHS: list[str] = [
    "/leadership", "/team", "/our-team", "/management", "/executives",
    "/board-of-directors", "/leadership-team"
]

# Tags whose content we always discard when extracting visible text
_SKIP_TAGS: set[str] = {
    "script", "style", "noscript", "header", "footer",
    "nav", "aside", "meta", "head", "link", "form",
}

# Maximum characters of visible text kept (to bound LLM token cost)
_MAX_TEXT_CHARS = settings.max_text_chars_for_llm

# Cache TTL: 24 hours (avoid hitting the same domain repeatedly in one run)
_CACHE_TTL_SECONDS: int = 86_400


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ScrapeResult:
    """Holds all data extracted from a single URL fetch."""

    url: str                          # Original requested URL
    final_url: str = ""               # URL after following redirects
    status_code: int = 0
    success: bool = False
    error: str = ""

    title: str = ""
    meta_description: str = ""
    og_description: str = ""
    visible_text: str = ""            # Cleaned body text (truncated)

    about_text: str = ""              # Text from /about or similar
    contact_text: str = ""            # Text from /contact or similar
    team_text: str = ""               # Text from /leadership or similar

    # Extracted social / contact signals from the page itself
    emails_found: list[str] = field(default_factory=list)
    phones_found: list[str] = field(default_factory=list)
    linkedin_found: str = ""
    facebook_found: str = ""
    twitter_found: str = ""
    instagram_found: str = ""

    # Redirect chain info
    redirect_count: int = 0
    is_redirect: bool = False

    def best_description(self) -> str:
        """Return the richest available description text."""
        for candidate in (self.og_description, self.meta_description, self.title):
            if candidate.strip():
                return candidate.strip()
        return ""

    def full_text(self) -> str:
        """Concatenate all text sections for LLM input."""
        parts = filter(None, [
            self.visible_text,
            self.about_text,
            self.contact_text,
            self.team_text,
        ])
        combined = "\n\n".join(parts)
        return combined[:_MAX_TEXT_CHARS]


# ---------------------------------------------------------------------------
# robots.txt cache (per domain, in-memory for the process lifetime)
# ---------------------------------------------------------------------------

class _RobotsCache:
    """Thread-safe in-memory cache for robots.txt parsers."""

    def __init__(self) -> None:
        self._cache: dict[str, urllib.robotparser.RobotFileParser] = {}

    def is_allowed(self, url: str, user_agent: str = "*") -> bool:
        """Return True if ``user_agent`` is allowed to fetch ``url``."""
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        if base not in self._cache:
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(f"{base}/robots.txt")
            try:
                rp.read()
            except Exception:
                # If robots.txt is unreachable, assume allowed
                self._cache[base] = None   # type: ignore[assignment]
                return True
            self._cache[base] = rp
        rp = self._cache[base]
        if rp is None:
            return True
        return rp.can_fetch(user_agent, url)


_robots_cache = _RobotsCache()


# ---------------------------------------------------------------------------
# Retry decorator (shared across fetch calls)
# ---------------------------------------------------------------------------

def _make_retry(max_attempts: int, backoff_factor: float):
    """Build a tenacity retry decorator from config values."""
    return retry(
        reraise=True,
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=backoff_factor, min=1, max=30),
        retry=retry_if_exception_type((
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.ChunkedEncodingError,
        )),
        before_sleep=lambda rs: logger.debug(
            "Retry %d/%d for %s after error.",
            rs.attempt_number, max_attempts, rs.args[1] if len(rs.args) > 1 else "?",
        ),
    )


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------

def _extract_visible_text(soup: BeautifulSoup) -> str:
    """
    Extract human-readable visible text from a BeautifulSoup document.

    Strips script/style tags, collapses whitespace, and truncates.
    """
    for tag in soup(list(_SKIP_TAGS)):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)
    # Collapse multiple spaces/newlines
    text = re.sub(r"\s{2,}", " ", text)
    return text[:_MAX_TEXT_CHARS]


def _extract_emails(text: str) -> list[str]:
    """Find all email addresses in a block of text."""
    pattern = re.compile(
        r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
    )
    return list(set(pattern.findall(text)))


def _extract_phones(text: str) -> list[str]:
    """Find common phone number patterns in a block of text."""
    pattern = re.compile(
        r"(?:\+?\d[\d\s\-().]{7,}\d)"
    )
    raw = pattern.findall(text)
    cleaned = []
    for r in raw:
        digits = re.sub(r"\D", "", r)
        if 7 <= len(digits) <= 15:
            cleaned.append(r.strip())
    return list(set(cleaned))[:5]   # cap at 5 to avoid junk


def _extract_linkedin(html: str) -> str:
    """Find a LinkedIn company URL in raw HTML."""
    m = re.search(
        r'https?://(?:www\.)?linkedin\.com/company/([^/"\'<>\s]+)',
        html, re.IGNORECASE,
    )
    return m.group(0) if m else ""


def _extract_social(html: str, platform: str) -> str:
    """Generic social link extractor (facebook, twitter/x, instagram)."""
    patterns = {
        "facebook":  r'https?://(?:www\.)?facebook\.com/(?!sharer)[^"\'<>\s]+',
        "twitter":   r'https?://(?:www\.)?(?:twitter|x)\.com/(?!share|intent)[^"\'<>\s]+',
        "instagram": r'https?://(?:www\.)?instagram\.com/[^"\'<>\s]+',
    }
    pat = patterns.get(platform, "")
    if not pat:
        return ""
    m = re.search(pat, html, re.IGNORECASE)
    return m.group(0) if m else ""


def _parse_page(html: str, base_url: str) -> dict:
    """
    Parse raw HTML and return a dict of extracted fields.

    Returns keys: title, meta_description, og_description, visible_text,
                  emails_found, phones_found, linkedin_found, facebook_found,
                  twitter_found, instagram_found
    """
    soup = BeautifulSoup(html, "html.parser")

    # Title
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""

    # Meta description
    meta_desc = ""
    meta_tag = soup.find("meta", attrs={"name": re.compile(r"^description$", re.I)})
    if meta_tag:
        meta_desc = meta_tag.get("content", "").strip()

    # Open Graph description
    og_tag = soup.find("meta", property="og:description")
    og_desc = og_tag.get("content", "").strip() if og_tag else ""

    # Visible text
    visible = _extract_visible_text(soup)

    # Contacts from text
    emails = _extract_emails(visible)
    phones = _extract_phones(visible)

    # Social links from raw HTML
    li_url  = _extract_linkedin(html)
    fb_url  = _extract_social(html, "facebook")
    tw_url  = _extract_social(html, "twitter")
    ig_url  = _extract_social(html, "instagram")

    return {
        "title": title,
        "meta_description": meta_desc,
        "og_description": og_desc,
        "visible_text": visible,
        "emails_found": emails,
        "phones_found": phones,
        "linkedin_found": li_url,
        "facebook_found": fb_url,
        "twitter_found": tw_url,
        "instagram_found": ig_url,
    }


# ---------------------------------------------------------------------------
# WebScraper
# ---------------------------------------------------------------------------

class WebScraper:
    """
    HTTP scraper with:
        - Persistent disk cache (diskcache)
        - robots.txt compliance
        - User-Agent rotation (fake_useragent)
        - Connection pooling (requests.Session)
        - Retry with exponential back-off (tenacity)
        - About + Contact sub-page discovery

    Thread-safe: a single instance can be shared across threads in
    ThreadPoolExecutor because requests.Session is not thread-safe —
    each thread gets its own session via a thread-local pattern.
    To avoid that complexity we create the session lazily per-thread
    by using a threading.local() store.
    """

    def __init__(self) -> None:
        self._cache = diskcache.Cache(
            directory=str(settings.cache_dir / "scraper"),
            timeout=_CACHE_TTL_SECONDS,
        )
        self._ua = _SafeUserAgent()
        self._timeout = settings.request_timeout
        self._respect_robots = settings.respect_robots_txt
        self._retry = _make_retry(
            settings.max_retries, settings.retry_backoff_factor
        )
        # Thread-local sessions
        import threading
        self._local = threading.local()

    # ------------------------------------------------------------------
    # Session (one per thread)
    # ------------------------------------------------------------------

    def _session(self) -> requests.Session:
        """Return the thread-local requests.Session, creating if needed."""
        if not hasattr(self._local, "session"):
            session = requests.Session()
            # Mount a retry-aware HTTP adapter
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=10,
                pool_maxsize=20,
                max_retries=0,          # tenacity handles retries
            )
            session.mount("https://", adapter)
            session.mount("http://", adapter)
            self._local.session = session
        return self._local.session

    # ------------------------------------------------------------------
    # robots.txt check
    # ------------------------------------------------------------------

    def _is_allowed(self, url: str) -> bool:
        if not self._respect_robots:
            return True
        allowed = _robots_cache.is_allowed(url, self._ua.current)
        if not allowed:
            logger.info("robots.txt disallows: %s", url)
        return allowed

    # ------------------------------------------------------------------
    # Low-level fetch
    # ------------------------------------------------------------------

    def _fetch(self, url: str) -> requests.Response:
        """
        Perform a single GET request. Raises on HTTP errors.
        Tenacity wraps this method for retries — do NOT call directly
        from outside the class.
        """
        headers = {
            "User-Agent": self._ua.random,
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;q=0.9,"
                "image/webp,*/*;q=0.8"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
        }
        response = self._session().get(
            url,
            headers=headers,
            timeout=self._timeout,
            allow_redirects=True,
            stream=False,
        )
        response.raise_for_status()
        return response

    def _fetch_with_retry(self, url: str) -> Optional[requests.Response]:
        """Fetch with tenacity retry wrapped around _fetch."""
        try:
            decorated = self._retry(self._fetch)
            return decorated(url)
        except requests.exceptions.HTTPError as exc:
            logger.warning("HTTP error fetching '%s': %s", url, exc)
        except requests.exceptions.Timeout:
            logger.warning("Timeout fetching '%s'.", url)
        except requests.exceptions.ConnectionError as exc:
            logger.warning("Connection error fetching '%s': %s", url, exc)
        except Exception as exc:
            logger.warning("Unexpected error fetching '%s': %s", url, exc)
        return None

    # ------------------------------------------------------------------
    # URL normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_scheme(url: str) -> str:
        """Prepend https:// if no scheme is present."""
        url = url.strip()
        if not url:
            return ""
        if not re.match(r"^https?://", url, re.IGNORECASE):
            return "https://" + url
        return url

    @staticmethod
    def _cache_key(url: str) -> str:
        """Deterministic cache key from URL."""
        return hashlib.md5(url.lower().encode()).hexdigest()   # noqa: S324

    # ------------------------------------------------------------------
    # Sub-page discovery
    # ------------------------------------------------------------------

    def _fetch_subpage(self, base_url: str, paths: list[str]) -> str:
        """
        Try a list of relative paths on the same domain.
        Return the visible text of the first successfully fetched page.
        """
        parsed = urlparse(base_url)
        origin = f"{parsed.scheme}://{parsed.netloc}"

        for path in paths:
            candidate = urljoin(origin, path)
            if not self._is_allowed(candidate):
                continue
            # Quick cache check for sub-pages too
            cache_key = self._cache_key(candidate)
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached.get("visible_text", "")

            response = self._fetch_with_retry(candidate)
            if response is None:
                logger.debug("Failed to fetch sub-page: %s", candidate)
                continue
            if response.status_code == 200:
                parsed_page = _parse_page(response.text, candidate)
                # Cache the sub-page result
                self._cache.set(cache_key, parsed_page, expire=_CACHE_TTL_SECONDS)
                text = parsed_page.get("visible_text", "")
                if text:
                    logger.debug("Successfully fetched sub-page: %s (%d chars)", candidate, len(text))
                    return text
                else:
                    logger.debug("Sub-page %s returned 200 but had no visible text.", candidate)
            else:
                logger.debug("Sub-page %s returned status %d", candidate, response.status_code)
            time.sleep(0.3)  # polite delay between sub-page requests

        return ""

    # ------------------------------------------------------------------
    # Public scrape method
    # ------------------------------------------------------------------

    def scrape(self, domain_or_url: str) -> ScrapeResult:
        """
        Scrape a company's website and return a structured ScrapeResult.

        Accepts a bare domain ("example.com") or a full URL.

        Workflow:
            1. Normalise URL (add scheme if missing)
            2. Check disk cache — return cached result if fresh
            3. Check robots.txt
            4. Fetch homepage (with retry)
            5. Parse HTML — extract title, descriptions, text, socials
            6. Fetch About page
            7. Fetch Contact page
            8. Cache result and return
        """
        url = self._ensure_scheme(domain_or_url)
        if not url:
            return ScrapeResult(url=domain_or_url, error="Empty URL", success=False)

        cache_key = self._cache_key(url)
        cached_result = self._cache.get(cache_key)
        if cached_result is not None:
            logger.debug("Cache hit for: %s", url)
            return ScrapeResult(**cached_result)

        result = ScrapeResult(url=url)

        # robots.txt check
        if not self._is_allowed(url):
            result.error = "Blocked by robots.txt"
            logger.info("Skipping %s — blocked by robots.txt", url)
            return result

        # Fetch homepage
        logger.info("Scraping: %s", url)
        response = self._fetch_with_retry(url)
        if response is None:
            result.error = "Fetch failed after retries"
            return result

        result.status_code = response.status_code
        result.final_url = str(response.url)
        result.is_redirect = (result.final_url.rstrip("/") != url.rstrip("/"))
        result.redirect_count = len(response.history)

        if result.is_redirect:
            logger.info("Redirect: %s → %s", url, result.final_url)

        # Parse homepage
        try:
            parsed = _parse_page(response.text, result.final_url)
        except Exception as exc:
            result.error = f"Parse error: {exc}"
            logger.warning("Parse error for '%s': %s", url, exc)
            return result

        result.title          = parsed["title"]
        result.meta_description = parsed["meta_description"]
        result.og_description   = parsed["og_description"]
        result.visible_text     = parsed["visible_text"]
        result.emails_found     = parsed["emails_found"]
        result.phones_found     = parsed["phones_found"]
        result.linkedin_found   = parsed["linkedin_found"]
        result.facebook_found   = parsed["facebook_found"]
        result.twitter_found    = parsed["twitter_found"]
        result.instagram_found  = parsed["instagram_found"]

        # Fetch About page
        result.about_text = self._fetch_subpage(result.final_url, _ABOUT_PATHS)

        # Fetch Contact page
        result.contact_text = self._fetch_subpage(result.final_url, _CONTACT_PATHS)

        # Fetch Leadership/Team page
        result.team_text = self._fetch_subpage(result.final_url, _LEADERSHIP_PATHS)

        result.success = True

        # Cache the result (serialise dataclass → dict)
        self._cache.set(
            cache_key,
            result.__dict__,
            expire=_CACHE_TTL_SECONDS,
        )

        logger.info(
            "Scraped '%s' — title: '%s', text: %d chars, about: %d chars, team: %d chars.",
            url,
            result.title[:60],
            len(result.visible_text),
            len(result.about_text),
            len(result.team_text),
        )
        return result

    # ------------------------------------------------------------------
    # Bulk scraping (convenience)
    # ------------------------------------------------------------------

    def scrape_many(self, urls: list[str]) -> list[ScrapeResult]:
        """
        Scrape a list of URLs sequentially (rate-limited).
        For parallel scraping use enrichment.py's ThreadPoolExecutor.
        """
        results = []
        for url in urls:
            results.append(self.scrape(url))
            time.sleep(1.0 / settings.requests_per_second)
        return results

    def close(self) -> None:
        """Close the disk cache. Call when the scraper is no longer needed."""
        self._cache.close()


# ---------------------------------------------------------------------------
# Safe UserAgent — graceful fallback if fake_useragent DB unavailable
# ---------------------------------------------------------------------------

class _SafeUserAgent:
    """
    Wraps fake_useragent.UserAgent with a hard-coded fallback so the
    scraper works even in offline / restricted environments.
    """

    _FALLBACK = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    )

    def __init__(self) -> None:
        try:
            self._ua = UserAgent(fallback=self._FALLBACK)
            self.current = self._ua.random
        except Exception:
            self._ua = None
            self.current = self._FALLBACK

    @property
    def random(self) -> str:
        if self._ua:
            try:
                return self._ua.random
            except Exception:
                pass
        return self._FALLBACK
