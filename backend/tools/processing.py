"""Normalise raw OpenAlex works and filter them (year range, duplicates). Pure functions, no I/O."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from backend.agent.schemas import PaperMetadata

_DOI_PREFIX = re.compile(r"^(?:https?://)?(?:dx\.)?doi\.org/|^doi:", re.IGNORECASE)
_WORK_ID = re.compile(r"(W\d+)\s*$", re.IGNORECASE)
_MAX_ABSTRACT_POSITIONS = 200_000


def normalize_doi(raw: Any) -> Optional[str]:
    """Return a DOI as `https://doi.org/10.xxx/...`, or None if `raw` is not a DOI."""
    if not isinstance(raw, str):
        return None
    value = _DOI_PREFIX.sub("", raw.strip()).strip()
    if not value.startswith("10."):
        return None
    return "https://doi.org/" + value


def doi_key(doi: Optional[str]) -> Optional[str]:
    """Case-insensitive comparison key for a DOI."""
    norm = normalize_doi(doi)
    return norm.lower() if norm else None


def short_openalex_id(raw: Any) -> Optional[str]:
    """`https://openalex.org/W123` -> `W123` (None if no work id found)."""
    if not isinstance(raw, str):
        return None
    match = _WORK_ID.search(raw.strip())
    return match.group(1).upper() if match else None


def normalize_title(title: str) -> str:
    """Comparison key for titles: NFKC, case-folded, punctuation removed, whitespace collapsed."""
    text = unicodedata.normalize("NFKC", title).casefold()
    return re.sub(r"[\W_]+", " ", text).strip()


def reconstruct_abstract(inverted: Any) -> Optional[str]:
    """Rebuild plain text from OpenAlex `abstract_inverted_index`. Returns None if absent/empty."""
    if not isinstance(inverted, dict) or not inverted:
        return None
    slots: Dict[int, str] = {}
    for word, positions in inverted.items():
        if not isinstance(positions, list):
            continue
        for pos in positions:
            if isinstance(pos, int) and 0 <= pos < _MAX_ABSTRACT_POSITIONS:
                slots[pos] = word
    if not slots:
        return None
    text = " ".join(slots[i] for i in sorted(slots)).strip()
    return text or None


def _landing_page(location: Any) -> Optional[str]:
    if isinstance(location, dict):
        url = location.get("landing_page_url")
        if isinstance(url, str) and url.strip():
            return url.strip()
    return None


def normalize_openalex_work(raw: Any) -> Optional[PaperMetadata]:
    """Convert one OpenAlex work object into PaperMetadata, or None if it lacks an id or a title."""
    if not isinstance(raw, dict):
        return None
    paper_id = short_openalex_id(raw.get("id"))
    title_raw = raw.get("title") or raw.get("display_name")
    if not paper_id or not isinstance(title_raw, str):
        return None
    title = " ".join(title_raw.split())
    if not title:
        return None

    authors: List[str] = []
    for authorship in raw.get("authorships") or []:
        author = authorship.get("author") if isinstance(authorship, dict) else None
        name = author.get("display_name") if isinstance(author, dict) else None
        if isinstance(name, str) and name.strip():
            authors.append(name.strip())

    year = raw.get("publication_year")
    year = year if isinstance(year, int) and not isinstance(year, bool) and 1000 <= year <= 3000 else None

    doi = normalize_doi(raw.get("doi"))
    if doi is None:
        doi = normalize_doi((raw.get("ids") or {}).get("doi") if isinstance(raw.get("ids"), dict) else None)

    source_url = (
        doi
        or _landing_page(raw.get("primary_location"))
        or _landing_page(raw.get("best_oa_location"))
        or f"https://openalex.org/{paper_id}"
    )
    return PaperMetadata(
        paper_id=paper_id,
        title=title,
        authors=authors,
        year=year,
        doi=doi,
        source_url=source_url,
        abstract=reconstruct_abstract(raw.get("abstract_inverted_index")),
    )


@dataclass
class SeenIndex:
    """Papers already seen in this run, for duplicate detection (ID, DOI, normalised title + year)."""

    ids: Set[str] = field(default_factory=set)
    dois: Set[str] = field(default_factory=set)
    title_years: Set[Tuple[str, Optional[int]]] = field(default_factory=set)

    def is_duplicate(self, paper: PaperMetadata) -> bool:
        key = doi_key(paper.doi)
        return (
            paper.paper_id in self.ids
            or (key is not None and key in self.dois)
            or (normalize_title(paper.title), paper.year) in self.title_years
        )

    def add(self, paper: PaperMetadata) -> None:
        self.ids.add(paper.paper_id)
        key = doi_key(paper.doi)
        if key:
            self.dois.add(key)
        self.title_years.add((normalize_title(paper.title), paper.year))


@dataclass
class FilterStats:
    """Counts for the tool result / log summary."""

    returned: int = 0
    dropped_year: int = 0
    dropped_duplicate: int = 0
    dropped_invalid: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {
            "returned": self.returned,
            "dropped_year": self.dropped_year,
            "dropped_duplicate": self.dropped_duplicate,
            "dropped_invalid": self.dropped_invalid,
        }


def filter_papers(
    papers: Iterable[PaperMetadata],
    year_from: Optional[int],
    year_to: Optional[int],
    seen: SeenIndex,
) -> Tuple[List[PaperMetadata], FilterStats]:
    """Keep new, valid, in-range papers; add them to `seen`. Metadata is never modified.

    Drops: missing id/title (invalid); year outside [year_from, year_to] or unknown year when a
    bound is set; duplicates (same OpenAlex id, same normalised DOI, or same normalised title + year,
    either from earlier rounds via `seen` or earlier in this batch).
    """
    stats = FilterStats()
    kept: List[PaperMetadata] = []
    bounded = year_from is not None or year_to is not None
    for paper in papers:
        stats.returned += 1
        if not paper.paper_id or not paper.title.strip():
            stats.dropped_invalid += 1
            continue
        if bounded:
            y = paper.year
            if y is None or (year_from is not None and y < year_from) or (year_to is not None and y > year_to):
                stats.dropped_year += 1
                continue
        if seen.is_duplicate(paper):
            stats.dropped_duplicate += 1
            continue
        seen.add(paper)
        kept.append(paper)
    return kept, stats
