"""Run 3 REAL requests through the agent (uses Gemini + OpenAlex quota) and save logs to demo/.

Usage (from repo root):
    python scripts/live_demo.py            # all three
    python scripts/live_demo.py 2          # only request #2 (comma list allowed: 1,3)

Outputs: demo/logs/<run_id>.jsonl, demo/results/<name>.json and <name>.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.agent.research_agent import run_research  # noqa: E402
from backend.agent.schemas import ResearchRequest  # noqa: E402
from backend.config import ConfigError, load_settings  # noqa: E402
from backend.tools.export import export_csv  # noqa: E402
from scripts.show_log import show  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "demo"

REQUESTS = {
    "1": ("popular_topic", ResearchRequest(
        topic="machine learning for medical image diagnosis", num_papers=8, language="en")),
    "2": ("narrow_topic", ResearchRequest(
        topic="3D laser scanning for conservation of Cham Hindu temples in Vietnam", num_papers=8, language="vi")),
    "3": ("tight_years", ResearchRequest(
        topic="large language models for education", year_from=2025, year_to=2026, num_papers=6, language="vi")),
}


def main() -> int:
    try:
        settings = load_settings(strict=True)
    except ConfigError as exc:
        print(exc)
        return 2
    settings = settings.model_copy(update={"log_dir": DEMO / "logs"})
    (DEMO / "results").mkdir(parents=True, exist_ok=True)
    chosen = sys.argv[1].split(",") if len(sys.argv) > 1 else list(REQUESTS)

    for key in chosen:
        if key not in REQUESTS:
            print(f"Unknown request '{key}' (choose from {', '.join(REQUESTS)})")
            return 2
        name, request = REQUESTS[key]
        print("=" * 100)
        print(f"[{key}] {name}: {request.model_dump_json(exclude_none=True)}")
        print("=" * 100)
        result = run_research(request, settings=settings)
        show(DEMO / "logs" / f"{result.run_id}.jsonl")
        (DEMO / "results" / f"{name}.json").write_text(
            json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        export_csv(result, DEMO / "results" / f"{name}.csv")
        print(f"\n>>> {result.status.upper()} ({result.terminated_by}): {result.message}")
        for item in result.selected:
            print(f"    [{item.label.value}] {item.metadata.year} {item.metadata.title[:90]}")
        print(f"    queries: {result.queries}")
        print(f"    run_id={result.run_id}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
