"""CSV export of the selected papers (Excel-friendly: UTF-8 with BOM)."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Union

from backend.agent.schemas import ResearchResult

CSV_HEADER = ["Title", "Authors", "Year", "DOI", "Source URL", "Abstract", "Relevance", "Relevance Reason"]


def _cell(value: object) -> str:
    return "" if value is None else str(value)


def export_csv(result: ResearchResult, path: Union[str, Path]) -> Path:
    """Write `result.selected` to `path` as CSV (utf-8-sig). None values become empty cells."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_HEADER)
        for item in result.selected:
            meta = item.metadata
            writer.writerow(
                [
                    _cell(meta.title),
                    "; ".join(meta.authors),
                    _cell(meta.year),
                    _cell(meta.doi),
                    _cell(meta.source_url),
                    _cell(meta.abstract),
                    item.label.value,
                    _cell(item.reason),
                ]
            )
    return out
