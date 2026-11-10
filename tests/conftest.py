"""Shared fixtures. Tests run offline: no network, no real keys."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from backend.agent.schemas import (
    PaperMetadata,
    RelevanceLabel,
    ResearchRequest,
    ResearchResult,
    SelectedPaper,
)

FIXTURES = Path(__file__).parent / "fixtures"


def load_works() -> List[Dict[str, Any]]:
    """Raw synthetic OpenAlex work objects."""
    return json.loads((FIXTURES / "works.json").read_text(encoding="utf-8"))


def load_queries() -> Dict[str, List[int]]:
    """Map of query string -> list of fixture work numbers (W90000000NN)."""
    return json.loads((FIXTURES / "queries.json").read_text(encoding="utf-8"))


@pytest.fixture
def raw_works() -> List[Dict[str, Any]]:
    return load_works()


@pytest.fixture
def raw_by_id(raw_works) -> Dict[str, Dict[str, Any]]:
    return {w["id"].rsplit("/", 1)[-1]: w for w in raw_works}


def make_result(selected: List[SelectedPaper], **overrides: Any) -> ResearchResult:
    """Build a minimal ResearchResult for export/API tests."""
    now = dt.datetime.now(dt.timezone.utc)
    base = dict(
        run_id="abcdef012345",
        status="completed",
        message="ok",
        request=ResearchRequest(topic="test topic"),
        selected_ids=[s.metadata.paper_id for s in selected],
        selected=selected,
        label_counts={},
        started_at=now,
        finished_at=now,
    )
    base.update(overrides)
    return ResearchResult(**base)


def make_selected(paper_id: str = "W1", title: str = "Title", **kw: Any) -> SelectedPaper:
    meta = PaperMetadata(
        paper_id=paper_id,
        title=title,
        authors=kw.pop("authors", ["A B"]),
        year=kw.pop("year", 2021),
        doi=kw.pop("doi", None),
        source_url=kw.pop("source_url", "https://openalex.org/" + paper_id),
        abstract=kw.pop("abstract", None),
    )
    return SelectedPaper(
        metadata=meta,
        label=kw.pop("label", RelevanceLabel.RELEVANT),
        reason=kw.pop("reason", "Matches the topic."),
    )


FAKE_GEMINI_KEY = "GKEY-SECRET-123456"
FAKE_OPENALEX_KEY = "OKEY-SECRET-654321"


@pytest.fixture
def settings(tmp_path):
    """Settings with fake keys and temp dirs; criteria file missing on purpose (fallback path)."""
    from pydantic import SecretStr

    from backend.config import Settings

    return Settings(
        gemini_api_key=SecretStr(FAKE_GEMINI_KEY),
        gemini_model="fake-model",
        gemini_min_interval_seconds=0,
        openalex_api_key=SecretStr(FAKE_OPENALEX_KEY),
        log_dir=tmp_path / "logs",
        runs_dir=tmp_path / "runs",
        criteria_path=tmp_path / "no_such_criteria.md",
    )
