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
