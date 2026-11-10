"""Live check of the OpenAlex connection: one small search + one related-works lookup.

Usage (from repo root):  python scripts/check_openalex.py
Cost: ~1 search ($0.001) + 1 free lookup + 1 list request. Never prints the API key.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import ConfigError, load_settings  # noqa: E402
from backend.tools.processing import SeenIndex  # noqa: E402
from backend.tools.search import OpenAlexClient, OpenAlexError, find_related, search_openalex  # noqa: E402


def main() -> int:
    try:
        settings = load_settings(strict=False)
    except ConfigError as exc:
        print(exc)
        return 2
    key = settings.openalex_key()
    if not key:
        print("OPENALEX_API_KEY is missing in .env")
        return 2
    client = OpenAlexClient(key)
    try:
        seen = SeenIndex()
        out = search_openalex(
            client, "adaptive learning systems", year_from=2020, year_to=2024, limit=5, seen=seen
        )
        print(f"[1/2] search: {out.stats.returned} results returned, {len(out.new_papers)} kept after filtering")
        if not out.new_papers:
            print("No papers kept; cannot test related works.")
            return 1
        first = out.new_papers[0]
        print(f"      first: {first.paper_id} | {first.year} | {first.title}")
        print(f"      abstract present: {first.abstract is not None} | url: {first.source_url}")
        rel = find_related(client, first.paper_id, year_from=None, year_to=None, limit=5, seen=SeenIndex())
        print(f"[2/2] related works of {first.paper_id}: {rel.stats.returned} returned, {len(rel.new_papers)} kept")
        if rel.new_papers:
            print(f"      first related: {rel.new_papers[0].title}")
    except OpenAlexError as exc:
        print(f"FAILED [{exc.code}]: {exc.message}")
        return 1
    finally:
        client.close()
    print("OpenAlex connection OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
