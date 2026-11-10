"""OpenAlex access: HTTP client (auth, retry, redaction) and the two search operations.

Authentication follows the current OpenAlex docs: API key in the `Authorization: Bearer` header
(`mailto` polite pool is deprecated). The key is never placed in URLs, logs or error messages.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence

import httpx

from backend.agent.schemas import PaperMetadata
from backend.tools.processing import (
    FilterStats,
    SeenIndex,
    filter_papers,
    normalize_openalex_work,
    short_openalex_id,
)

logger = logging.getLogger(__name__)

BASE_URL = "https://api.openalex.org"
WORK_FIELDS = (
    "id,doi,title,display_name,publication_year,authorships,primary_location,"
    "best_oa_location,abstract_inverted_index"
)
MAX_RETRY_WAIT_SECONDS = 10.0
MAX_BATCH_IDS = 100  # OpenAlex allows up to 100 values in one OR filter


class OpenAlexError(Exception):
    """OpenAlex failure with a stable machine code (one of the OPENALEX_* codes)."""

    def __init__(self, code: str, message: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable


class SearchBackend(Protocol):
    """What the agent needs from a paper source. Tests replace it with FakeOpenAlex."""

    def search_works(
        self, query: str, *, year_from: Optional[int], year_to: Optional[int], limit: int
    ) -> List[Dict[str, Any]]:
        """Keyword search; returns raw OpenAlex work objects."""

    def related_works(self, seed_id: str, *, limit: int) -> List[Dict[str, Any]]:
        """Works related to `seed_id` (short OpenAlex ID); returns raw work objects."""


class OpenAlexClient:
    """Synchronous OpenAlex client built on httpx."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = BASE_URL,
        timeout: float = 20.0,
        max_retries: int = 3,
        transport: Optional[httpx.BaseTransport] = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._key = api_key
        self._max_retries = max_retries
        self._sleep = sleep
        self._http = httpx.Client(
            base_url=base_url,
            timeout=timeout,
            transport=transport,
            headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
        )

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._http.close()

    def _redact(self, text: str) -> str:
        return text.replace(self._key, "***") if self._key else text

    @staticmethod
    def _retry_after(resp: httpx.Response) -> Optional[float]:
        try:
            return float(resp.headers.get("Retry-After", ""))
        except ValueError:
            return None

    def _body_message(self, resp: httpx.Response) -> str:
        try:
            body = resp.json()
            text = str(body.get("message") or body.get("error") or "") if isinstance(body, dict) else ""
        except ValueError:
            text = resp.text
        return self._redact(text.strip())[:200]

    def _get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        last_error: Optional[OpenAlexError] = None
        for attempt in range(self._max_retries + 1):
            wait: Optional[float] = None
            try:
                resp = self._http.get(path, params=params)
            except httpx.TimeoutException:
                last_error = OpenAlexError("OPENALEX_TIMEOUT", "OpenAlex request timed out.", retryable=True)
            except httpx.HTTPError as exc:
                last_error = OpenAlexError(
                    "OPENALEX_HTTP", self._redact(f"Network error: {type(exc).__name__}"), retryable=True
                )
            else:
                status = resp.status_code
                if status == 200:
                    try:
                        data = resp.json()
                    except ValueError:
                        raise OpenAlexError("OPENALEX_HTTP", "OpenAlex returned invalid JSON.") from None
                    if not isinstance(data, dict):
                        raise OpenAlexError("OPENALEX_HTTP", "OpenAlex returned an unexpected payload.")
                    return data
                detail = self._body_message(resp)
                if status in (401, 403):
                    raise OpenAlexError(
                        "OPENALEX_AUTH", f"OpenAlex rejected the API key (HTTP {status}). {detail}".strip()
                    )
                if status == 429:
                    wait = self._retry_after(resp)
                    last_error = OpenAlexError(
                        "OPENALEX_RATE_LIMIT", f"OpenAlex rate limit or daily budget exceeded. {detail}".strip(),
                        retryable=True,
                    )
                    if wait is not None and wait > MAX_RETRY_WAIT_SECONDS:
                        raise last_error  # e.g. daily budget: waiting is pointless
                elif status >= 500:
                    last_error = OpenAlexError(
                        "OPENALEX_HTTP", f"OpenAlex server error (HTTP {status}). {detail}".strip(), retryable=True
                    )
                else:
                    raise OpenAlexError("OPENALEX_HTTP", f"OpenAlex rejected the request (HTTP {status}). {detail}".strip())
            if attempt < self._max_retries:
                delay = min(wait if wait is not None else 0.5 * (2 ** attempt), MAX_RETRY_WAIT_SECONDS)
                logger.warning("OpenAlex %s, retry %d/%d in %.1fs", last_error.code, attempt + 1, self._max_retries, delay)
                self._sleep(delay)
        assert last_error is not None
        raise last_error

    def search_works(
        self, query: str, *, year_from: Optional[int], year_to: Optional[int], limit: int
    ) -> List[Dict[str, Any]]:
        """Keyword search on works (title/abstract/fulltext), with a server-side year filter."""
        params: Dict[str, Any] = {"search": query, "per_page": max(1, min(limit, 100)), "select": WORK_FIELDS}
        filters = []
        if year_from is not None:
            filters.append(f"from_publication_date:{year_from}-01-01")
        if year_to is not None:
            filters.append(f"to_publication_date:{year_to}-12-31")
        if filters:
            params["filter"] = ",".join(filters)
        results = self._get("/works", params).get("results")
        return [w for w in results if isinstance(w, dict)] if isinstance(results, list) else []

    def related_works(self, seed_id: str, *, limit: int) -> List[Dict[str, Any]]:
        """Fetch the `related_works` of a seed (1 free lookup) and then those works (1 list request)."""
        seed = self._get(f"/works/{seed_id}", {"select": "id,related_works"})
        ids = [i for i in (short_openalex_id(r) for r in seed.get("related_works") or []) if i]
        ids = ids[: max(1, min(limit, MAX_BATCH_IDS))]
        if not ids:
            return []
        params = {"filter": "ids.openalex:" + "|".join(ids), "per_page": len(ids), "select": WORK_FIELDS}
        results = self._get("/works", params).get("results")
        return [w for w in results if isinstance(w, dict)] if isinstance(results, list) else []


@dataclass
class SearchOutcome:
    """Result of one search operation after normalisation and filtering."""

    new_papers: List[PaperMetadata]
    stats: FilterStats


def _process(
    raw_works: Sequence[Dict[str, Any]],
    year_from: Optional[int],
    year_to: Optional[int],
    seen: SeenIndex,
) -> SearchOutcome:
    papers: List[PaperMetadata] = []
    invalid = 0
    for raw in raw_works:
        paper = normalize_openalex_work(raw)
        if paper is None:
            invalid += 1
        else:
            papers.append(paper)
    kept, stats = filter_papers(papers, year_from, year_to, seen)
    stats.returned = len(raw_works)
    stats.dropped_invalid += invalid
    return SearchOutcome(new_papers=kept, stats=stats)


def search_openalex(
    backend: SearchBackend,
    query: str,
    *,
    year_from: Optional[int],
    year_to: Optional[int],
    limit: int,
    seen: SeenIndex,
) -> SearchOutcome:
    """Keyword search -> normalise -> year filter -> de-duplicate against `seen`. Raises OpenAlexError."""
    raw = backend.search_works(query, year_from=year_from, year_to=year_to, limit=limit)
    return _process(raw, year_from, year_to, seen)


def find_related(
    backend: SearchBackend,
    seed_id: str,
    *,
    year_from: Optional[int],
    year_to: Optional[int],
    limit: int,
    seen: SeenIndex,
) -> SearchOutcome:
    """Related works of a seed -> same pipeline as search_openalex. Raises OpenAlexError."""
    raw = backend.related_works(seed_id, limit=limit)
    return _process(raw, year_from, year_to, seen)
