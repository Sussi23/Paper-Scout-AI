"""Test doubles. FakeOpenAlex serves synthetic works from tests/fixtures (no network)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.tools.search import OpenAlexError
from tests.conftest import load_queries, load_works


class FakeOpenAlex:
    """Implements SearchBackend from fixtures.

    Args:
        fail_with: if set, every call raises this error (still recorded in `calls`).
        empty: if True, every search returns no results.
        queries: override the query -> work-numbers map.
    """

    def __init__(
        self,
        fail_with: Optional[OpenAlexError] = None,
        empty: bool = False,
        queries: Optional[Dict[str, List[int]]] = None,
    ) -> None:
        self.by_id = {w["id"].rsplit("/", 1)[-1]: w for w in load_works()}
        self.queries = {k.lower(): v for k, v in (queries or load_queries()).items()}
        self.fail_with = fail_with
        self.empty = empty
        self.calls: List[Dict[str, Any]] = []

    def search_works(self, query, *, year_from, year_to, limit):
        self.calls.append({"op": "search", "query": query, "year_from": year_from, "year_to": year_to, "limit": limit})
        if self.fail_with:
            raise self.fail_with
        if self.empty:
            return []
        numbers = self.queries.get(query.strip().lower(), [])
        return [self.by_id[f"W90000000{n:02d}"] for n in numbers][:limit]

    def related_works(self, seed_id, *, limit):
        self.calls.append({"op": "related", "seed_id": seed_id, "limit": limit})
        if self.fail_with:
            raise self.fail_with
        seed = self.by_id.get(seed_id)
        if not seed or self.empty:
            return []
        ids = [r.rsplit("/", 1)[-1] for r in seed.get("related_works", [])][:limit]
        return [self.by_id[i] for i in ids if i in self.by_id]


# --------------------------------------------------------------------------------------------
# FakeLLM: scripted model turns
# --------------------------------------------------------------------------------------------
from typing import Callable, Sequence, Union  # noqa: E402

from google.genai import types  # noqa: E402

from backend.agent.llm import LLMError, LLMResponse  # noqa: E402


def fc(name: str, **args: Any) -> types.Part:
    """A function_call part."""
    return types.Part.from_function_call(name=name, args=args)


def turn(*parts: types.Part, text: Optional[str] = None) -> types.Content:
    """A model Content made of the given parts (optionally preceded by a text part)."""
    all_parts = ([types.Part(text=text)] if text else []) + list(parts)
    return types.Content(role="model", parts=all_parts)


def tool_responses(contents: Sequence[types.Content]) -> List[Dict[str, Any]]:
    """function_response dicts of the last user Content (what the model just received)."""
    last = contents[-1]
    return [p.function_response.response for p in (last.parts or []) if p.function_response]


def last_progress(contents: Sequence[types.Content]) -> Dict[str, Any]:
    """Most recent `progress` object found in any tool response so far."""
    for content in reversed(list(contents)):
        for part in reversed(content.parts or []):
            resp = part.function_response.response if part.function_response else None
            if isinstance(resp, dict) and "progress" in resp:
                return resp["progress"]
    return {}


def screen_all(label: str = "Not Relevant", reason: str = "Judged from the abstract.", rationale: str = "Screening new papers."):
    """Script item: record_screening for every unscreened id in the latest progress."""

    def build(contents):
        ids = last_progress(contents).get("unscreened_ids", [])
        return turn(
            fc("record_screening", assessments=[{"paper_id": i, "label": label, "reason": reason} for i in ids],
               rationale=rationale)
        )

    return build


ScriptItem = Union[types.Content, Exception, Callable[[Sequence[types.Content]], types.Content]]


class FakeLLM:
    """Plays a script of model turns. Records a snapshot of `contents` for every request."""

    model = "fake-model"

    def __init__(self, script: Sequence[ScriptItem]) -> None:
        self.script = list(script)
        self.requests: List[List[types.Content]] = []
        self.system_instructions: List[str] = []
        self.tools_seen: List[Any] = []

    def generate(self, contents, tools, system_instruction):
        self.requests.append(list(contents))
        self.system_instructions.append(system_instruction)
        self.tools_seen.append(tools)
        if not self.script:
            raise AssertionError("FakeLLM script exhausted: the agent asked for more turns than scripted")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        content = item(contents) if callable(item) else item
        return LLMResponse.from_content(content, usage={"total_token_count": 1}, finish_reason="STOP")
