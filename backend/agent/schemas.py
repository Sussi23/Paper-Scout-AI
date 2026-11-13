"""Pydantic v2 schemas: request, paper metadata, assessments, logs and final result."""
from __future__ import annotations

import datetime as dt
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MIN_YEAR = 1900


def max_year() -> int:
    """Latest accepted publication year (current year + 1, for early-access papers)."""
    return dt.date.today().year + 1


class RelevanceLabel(str, Enum):
    """The four allowed screening labels."""

    RELEVANT = "Relevant"
    POSSIBLY_RELEVANT = "Possibly Relevant"
    NOT_RELEVANT = "Not Relevant"
    INSUFFICIENT_INFORMATION = "Insufficient Information"


class ResearchRequest(BaseModel):
    """Input from the website."""

    topic: str
    year_from: Optional[int] = None
    year_to: Optional[int] = None
    num_papers: int = Field(default=10, ge=1, le=30)
    criteria: Optional[str] = None
    language: Literal["vi", "en"] = "vi"

    @field_validator("topic", mode="before")
    @classmethod
    def _strip_topic(cls, v: Any) -> Any:
        return v.strip() if isinstance(v, str) else v

    @field_validator("topic")
    @classmethod
    def _topic_length(cls, v: str) -> str:
        if not 3 <= len(v) <= 300:
            raise ValueError("topic must be 3-300 characters after trimming")
        return v

    @field_validator("year_from", "year_to")
    @classmethod
    def _year_range(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and not MIN_YEAR <= v <= max_year():
            raise ValueError(f"year must be between {MIN_YEAR} and {max_year()}")
        return v

    @field_validator("criteria", mode="before")
    @classmethod
    def _clean_criteria(cls, v: Any) -> Any:
        if isinstance(v, str):
            v = v.strip()
            return v or None
        return v

    @field_validator("criteria")
    @classmethod
    def _criteria_length(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and len(v) > 1000:
            raise ValueError("criteria must be at most 1000 characters")
        return v

    @model_validator(mode="after")
    def _years_ordered(self) -> "ResearchRequest":
        if self.year_from is not None and self.year_to is not None and self.year_from > self.year_to:
            raise ValueError("year_from must be <= year_to")
        return self


class PaperMetadata(BaseModel):
    """Bibliographic data. Comes ONLY from OpenAlex (via code), never from the LLM."""

    model_config = ConfigDict(frozen=True)

    paper_id: str
    title: str
    authors: List[str] = Field(default_factory=list)
    year: Optional[int] = None
    doi: Optional[str] = None
    source_url: str
    abstract: Optional[str] = None
    source: str = "openalex"


class PaperAssessment(BaseModel):
    """What the LLM may say about a paper: id + label + reason. Extra fields are ignored."""

    model_config = ConfigDict(extra="ignore")

    paper_id: str
    label: RelevanceLabel
    reason: str = Field(min_length=5, max_length=600)


class SelectedPaper(BaseModel):
    """A paper in the final answer: registry metadata plus the model's label/reason."""

    metadata: PaperMetadata
    label: RelevanceLabel
    reason: str


class StepMetrics(BaseModel):
    """Numbers for one tool call, for the UI. Computed by code; never sent to the model.

    Search fields are set only for successful search_papers / find_related_papers calls, screening fields
    only for record_screening; the state snapshot (taken AFTER the step) is present for every call.
    """

    # search_papers / find_related_papers only
    returned: Optional[int] = None
    new_papers: Optional[int] = None
    dropped_year: Optional[int] = None
    dropped_duplicate: Optional[int] = None
    dropped_invalid: Optional[int] = None
    # record_screening only
    accepted: Optional[int] = None
    rejected: Optional[int] = None
    relabeled: Optional[int] = None
    labels_this_step: Optional[Dict[str, int]] = None
    # snapshot of the run state after this step (every step)
    papers_found: int = 0
    screened: int = 0
    unscreened: int = 0
    relevant: int = 0
    possibly_relevant: int = 0
    searches_used: int = 0


class ToolCallLog(BaseModel):
    """One tool call as shown on the Agent Activity screen."""

    step: int
    tool: str
    args: Dict[str, Any] = Field(default_factory=dict)
    rationale: Optional[str] = None
    status: Literal["ok", "error", "rejected"]
    error_code: Optional[str] = None
    summary: str = ""
    latency_ms: int = 0
    timestamp: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.timezone.utc))
    metrics: Optional[StepMetrics] = None  # None in results saved before this field existed


class ScreenedPaper(BaseModel):
    """Every paper found in the run (bibliographic data from the registry; label/reason from the model)."""

    paper_id: str
    title: str
    authors: List[str] = Field(default_factory=list)
    year: Optional[int] = None
    doi: Optional[str] = None
    source_url: str
    has_abstract: bool = False
    label: Optional[RelevanceLabel] = None  # None = not screened
    reason: Optional[str] = None
    selected: bool = False
    found_at_step: int = 0
    found_by_tool: str = ""
    found_via: str = ""  # the search query, or "related:<seed_id>"


class Funnel(BaseModel):
    """Totals over successful searches. Invariant:
    returned == dropped_invalid + dropped_year + dropped_duplicate + unique_found."""

    returned: int = 0
    dropped_invalid: int = 0
    dropped_year: int = 0
    dropped_duplicate: int = 0
    unique_found: int = 0
    screened: int = 0
    selected: int = 0


class ResearchResult(BaseModel):
    """Final output. All metadata here was read from the registry, not from the model."""

    run_id: str
    status: Literal["completed", "partial", "error"]
    message: str
    request: ResearchRequest
    selected_ids: List[str] = Field(default_factory=list)
    selected: List[SelectedPaper] = Field(default_factory=list)
    label_counts: Dict[str, int] = Field(default_factory=dict)
    searches_used: int = 0
    queries: List[str] = Field(default_factory=list)
    tool_history: List[ToolCallLog] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    terminated_by: str = ""
    model: str = ""
    started_at: dt.datetime
    finished_at: dt.datetime
    # added later; results saved before these existed still validate with the defaults
    screened_papers: List[ScreenedPaper] = Field(default_factory=list)
    funnel: Optional[Funnel] = None
