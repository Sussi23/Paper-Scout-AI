"""Mutable state of one run. The registry here is the only source of paper metadata."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.agent.schemas import (
    PaperAssessment,
    PaperMetadata,
    RelevanceLabel,
    ResearchRequest,
    ToolCallLog,
)
from backend.config import Settings
from backend.tools.processing import SeenIndex


@dataclass
class RunState:
    """Everything the code (not the model) knows and decides during a run."""

    request: ResearchRequest
    settings: Settings
    run_id: str
    registry: Dict[str, PaperMetadata] = field(default_factory=dict)  # insertion order = discovery order
    seen: SeenIndex = field(default_factory=SeenIndex)
    assessments: Dict[str, PaperAssessment] = field(default_factory=dict)
    searches_used: int = 0
    successful_searches: int = 0
    queries: List[str] = field(default_factory=list)
    tool_history: List[ToolCallLog] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    finished: bool = False
    finish_reason: Optional[str] = None
    finish_confirm_asked: bool = False
    text_nudged: bool = False
    terminated_by: str = ""
    started_at: dt.datetime = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc))

    # -- budget ---------------------------------------------------------------------------
    @property
    def searches_remaining(self) -> int:
        return max(0, self.settings.max_searches - self.searches_used)

    # -- registry / screening -------------------------------------------------------------
    def register(self, papers: List[PaperMetadata]) -> None:
        for paper in papers:
            self.registry.setdefault(paper.paper_id, paper)

    def ids_with_label(self, label: RelevanceLabel) -> List[str]:
        """Paper ids carrying `label`, in discovery order."""
        return [pid for pid in self.registry if pid in self.assessments and self.assessments[pid].label == label]

    def unscreened_ids(self) -> List[str]:
        return [pid for pid in self.registry if pid not in self.assessments]

    def progress(self) -> Dict[str, Any]:
        return {
            "target": self.request.num_papers,
            "relevant_so_far": len(self.ids_with_label(RelevanceLabel.RELEVANT)),
            "possibly_so_far": len(self.ids_with_label(RelevanceLabel.POSSIBLY_RELEVANT)),
            "unscreened_ids": self.unscreened_ids(),
        }

    def budget(self) -> Dict[str, int]:
        return {"searches_used": self.searches_used, "searches_remaining": self.searches_remaining}

    def label_counts(self) -> Dict[str, int]:
        counts = {label.value: 0 for label in RelevanceLabel}
        for assessment in self.assessments.values():
            counts[assessment.label.value] += 1
        return counts

    def selected_ids(self) -> List[str]:
        """All Relevant, then Possibly Relevant (discovery order), cut at num_papers."""
        ids = self.ids_with_label(RelevanceLabel.RELEVANT) + self.ids_with_label(RelevanceLabel.POSSIBLY_RELEVANT)
        return ids[: self.request.num_papers]
