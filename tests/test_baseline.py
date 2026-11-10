import json

import pytest

from backend.tools.search import OpenAlexError
from scripts.baseline_script import compare, run_baseline
from tests.fakes import FakeLLM, FakeOpenAlex, turn
from tests.test_agent import req, W

QUERIES = ["adaptive learning", "learning analytics", "virtual reality education"]


def reply(*ids, label="Relevant"):
    items = [{"paper_id": i, "label": label, "reason": "Judged from the abstract."} for i in ids]
    return turn(text="Here you go:\n" + json.dumps(items))


def test_always_three_searches_even_when_target_already_met(settings):
    # target 1 is met after the first query, but the baseline never stops early
    backend = FakeOpenAlex()
    llm = FakeLLM([reply(*[W(n) for n in (1, 2, 3, 4, 5, 10, 11, 12, 13, 15)])])
    result = run_baseline(req(num_papers=1), QUERIES, llm=llm, search_backend=backend, settings=settings)
    assert [c["op"] for c in backend.calls] == ["search"] * 3
    assert result.searches_used == 3 and result.queries == QUERIES
    assert result.terminated_by == "fixed_pipeline" and result.status == "completed"
    assert len(result.selected_ids) == 1  # cut at num_papers, same rule as the agent


def test_requires_exactly_three_queries(settings):
    with pytest.raises(ValueError):
        run_baseline(req(), QUERIES[:2], llm=FakeLLM([]), search_backend=FakeOpenAlex(), settings=settings)


def test_screening_ignores_unknown_ids_and_bad_json(settings):
    ids = [W(1), W(2)]
    bad = [turn(text="not json at all")]
    llm = FakeLLM(bad)
    result = run_baseline(req(), QUERIES, llm=llm, search_backend=FakeOpenAlex(), settings=settings)
    assert result.label_counts["Relevant"] == 0 and result.selected == []
    llm2 = FakeLLM([turn(text=json.dumps([
        {"paper_id": "W9999999999", "label": "Relevant", "reason": "Invented paper id."},
        {"paper_id": ids[0], "label": "Highly Relevant", "reason": "Invalid label value."},
        {"paper_id": ids[1], "label": "Relevant", "reason": "Valid entry here."},
    ]))])
    result2 = run_baseline(req(), QUERIES, llm=llm2, search_backend=FakeOpenAlex(), settings=settings)
    assert result2.selected_ids == [ids[1]]


def test_search_errors_count_against_the_three_but_do_not_crash(settings):
    backend = FakeOpenAlex(fail_with=OpenAlexError("OPENALEX_HTTP", "down"))
    result = run_baseline(req(), QUERIES, llm=FakeLLM([]), search_backend=backend, settings=settings)
    assert result.status == "error" and result.searches_used == 3 and len(result.errors) == 3


def test_compare_lines(settings):
    result = run_baseline(req(), QUERIES, llm=FakeLLM([reply(W(1))]), search_backend=FakeOpenAlex(), settings=settings)
    agent = {"label_counts": result.label_counts, "selected_ids": [W(1)], "status": "partial",
             "searches_used": 1, "queries": ["adaptive learning"]}
    text = "\n".join(compare(agent, result))
    assert "AGENT" in text and "BASELINE" in text and "selected in common" in text
