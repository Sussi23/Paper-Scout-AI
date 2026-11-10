"""Agent loop, state, tool registry, run logger and Gemini client tests (offline: FakeLLM / FakeOpenAlex)."""
from __future__ import annotations
import json
from typing import Any, Dict, List
from google.genai import types
from backend.agent.llm import LLMError
from backend.agent.research_agent import run_research
from backend.agent.schemas import ResearchRequest
from backend.tools.processing import normalize_openalex_work, normalize_title
from backend.tools.search import OpenAlexError
from tests.conftest import FAKE_GEMINI_KEY, FAKE_OPENALEX_KEY
from tests.fakes import FakeLLM, FakeOpenAlex, fc, screen_all, tool_responses, turn
import httpx
import pytest
from google.genai import errors, types
from backend.agent.llm import GeminiClient, LLMError


# ----- from test_agent.py ----------------------------------------
def W(n: int) -> str:
    return f"W90000000{n:02d}"


def req(**kw: Any) -> ResearchRequest:
    base: Dict[str, Any] = dict(topic="adaptive learning", year_from=2019, year_to=2024, num_papers=3)
    base.update(kw)
    return ResearchRequest(**base)


def run(script, settings, request=None, backend=None):
    llm = FakeLLM(script)
    backend = backend or FakeOpenAlex()
    events: List[dict] = []
    result = run_research(request or req(), llm=llm, search_backend=backend, settings=settings, on_event=events.append)
    return result, llm, backend, events


def read_log(settings, run_id) -> List[dict]:
    path = settings.log_dir / f"{run_id}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def assess(pid, label, reason="Judged from title and abstract."):
    return {"paper_id": pid, "label": label, "reason": reason}


SEARCH1 = turn(fc("search_papers", query="adaptive learning", rationale="Initial broad query on the topic."))


def all_ids_returned(llm: FakeLLM) -> set:
    ids = set()
    for request in llm.requests:
        for content in request:
            for part in content.parts or []:
                resp = part.function_response.response if part.function_response else None
                if resp and "new_papers" in resp:
                    ids.update(p["paper_id"] for p in resp["new_papers"])
    return ids


# --- core scenarios --------------------------------------------------------------------------
def test_case1_enough_papers(settings):
    script = [
        SEARCH1,
        turn(fc("record_screening", rationale="Screening the five new papers.", assessments=[
            assess(W(1), "Relevant"), assess(W(2), "Relevant"), assess(W(3), "Relevant"),
            assess(W(4), "Insufficient Information", "No abstract available."), assess(W(5), "Not Relevant"),
        ])),
        turn(fc("finish_research", reason="Target reached.")),
    ]
    result, llm, _, _ = run(script, settings)
    assert result.status == "completed" and result.terminated_by == "model_finish"
    assert result.selected_ids == [W(1), W(2), W(3)]
    assert set(result.selected_ids) <= all_ids_returned(llm)
    assert result.label_counts == {
        "Relevant": 3, "Possibly Relevant": 0, "Not Relevant": 1, "Insufficient Information": 1,
    }
    assert result.searches_used == 1 and result.queries == ["adaptive learning"]
    assert "3/3" in result.message


def test_case2_changes_keywords(settings):
    script = [
        SEARCH1,
        turn(fc("record_screening", rationale="Screen first batch.", assessments=[
            assess(W(1), "Relevant"), assess(W(2), "Relevant"), assess(W(3), "Relevant"),
            assess(W(4), "Insufficient Information", "No abstract."), assess(W(5), "Not Relevant"),
        ])),
        turn(fc("search_papers", query="learning analytics",
                rationale="Only 3 relevant so far and the first query lacked analytics-focused work.")),
        turn(fc("record_screening", rationale="Screen second batch.", assessments=[
            assess(W(10), "Relevant"), assess(W(11), "Relevant"), assess(W(12), "Not Relevant"),
        ])),
        turn(fc("finish_research", reason="Five relevant papers found.")),
    ]
    result, _, _, _ = run(script, settings, req(num_papers=5))
    assert result.status == "completed"
    assert len(result.queries) == 2
    assert normalize_title(result.queries[0]) != normalize_title(result.queries[1])
    searches = [t for t in result.tool_history if t.tool == "search_papers"]
    assert len(searches) == 2 and all(t.rationale for t in searches)


def test_case3_tool_error_every_search(settings):
    backend = FakeOpenAlex(fail_with=OpenAlexError("OPENALEX_AUTH", "key rejected"))
    script = [
        turn(fc("search_papers", query="q one", rationale="First attempt.")),
        turn(fc("search_papers", query="q two", rationale="Retry with other words.")),
        turn(fc("search_papers", query="q three", rationale="Last attempt.")),
    ]
    result, llm, _, _ = run(script, settings, backend=backend)
    assert result.status == "error" and result.terminated_by == "budget"
    assert len(llm.requests) == 3 and len(result.errors) == 3
    first_reply = tool_responses(llm.requests[1])[0]
    assert first_reply["status"] == "error" and first_reply["code"] == "OPENALEX_AUTH"
    assert all(t.status == "error" for t in result.tool_history)


def test_empty_results_is_partial(settings):
    script = [
        SEARCH1,
        turn(fc("finish_research", reason="Nothing found.")),
        turn(fc("finish_research", reason="Confirmed: topic has no hits.")),
    ]
    result, _, _, _ = run(script, settings, backend=FakeOpenAlex(empty=True))
    assert result.status == "partial" and "0/3" in result.message
    assert [t.error_code for t in result.tool_history] == [None, "PREMATURE_FINISH_CONFIRM", None]


def test_budget_enforced(settings):
    queries = ["adaptive learning", "learning analytics", "virtual reality education", "fourth query", "fifth query"]
    script = [
        turn(*[fc("search_papers", query=q, rationale=f"Try {q}.") for q in queries]),
        screen_all(),
    ]
    result, _, backend, _ = run(script, settings)
    assert result.searches_used == 3 and len(backend.calls) == 3
    assert [t.status for t in result.tool_history[:5]] == ["ok", "ok", "ok", "rejected", "rejected"]
    assert [t.error_code for t in result.tool_history[3:5]] == ["SEARCH_BUDGET_EXHAUSTED"] * 2
    guards = [e for e in read_log(settings, result.run_id) if e["event"] == "guard"]
    assert sum(g["data"]["kind"] == "SEARCH_BUDGET_EXHAUSTED" for g in guards) == 2
    assert result.terminated_by == "budget"


def test_hallucinated_paper_id_rejected(settings):
    script = [
        SEARCH1,
        turn(fc("record_screening", rationale="Screen.", assessments=[
            assess("W9999999999", "Relevant", "Invented paper that does not exist."), assess(W(1), "Relevant"),
        ])),
        screen_all(),
        turn(fc("finish_research", reason="Done.")),
    ]
    result, llm, _, _ = run(script, settings, req(num_papers=1))
    reply = tool_responses(llm.requests[2])[0]
    assert reply["status"] == "ok" and reply["accepted"] == [W(1)]
    assert reply["rejected"][0]["code"] == "UNKNOWN_PAPER_ID"
    assert "W9999999999" not in result.selected_ids and result.selected_ids == [W(1)]
    kinds = [e["data"]["kind"] for e in read_log(settings, result.run_id) if e["event"] == "guard"]
    assert "UNKNOWN_PAPER_ID" in kinds


def test_invalid_label_rejected(settings):
    script = [
        SEARCH1,
        turn(fc("record_screening", rationale="Screen.", assessments=[
            assess(W(1), "Highly Relevant"), assess(W(2), "Relevant"),
        ])),
        screen_all(),
        turn(fc("finish_research", reason="Done.")),
    ]
    result, llm, _, _ = run(script, settings, req(num_papers=1))
    reply = tool_responses(llm.requests[2])[0]
    assert reply["rejected"][0]["code"] == "INVALID_LABEL"
    assert W(1) in reply["progress"]["unscreened_ids"]  # still unscreened after the bad label
    assert result.selected_ids == [W(2)]


def test_metadata_integrity(settings, raw_by_id):
    fake_extras = {"doi": "10.1/FAKE", "title": "FAKE TITLE", "year": 1900, "authors": ["Mallory"], "abstract": "fake"}
    script = [
        SEARCH1,
        turn(fc("record_screening", rationale="Screen.", assessments=[
            {**assess(W(1), "Relevant"), **fake_extras}, assess(W(2), "Relevant"), assess(W(3), "Relevant"),
            assess(W(4), "Not Relevant"), assess(W(5), "Not Relevant"),
        ])),
        turn(fc("finish_research", reason="Done.")),
    ]
    result, _, _, _ = run(script, settings)
    assert len(result.selected) == 3
    for item in result.selected:
        expected = normalize_openalex_work(raw_by_id[item.metadata.paper_id])
        raw = raw_by_id[item.metadata.paper_id]
        assert item.metadata == expected
        assert item.metadata.title == raw["title"] and item.metadata.year == raw["publication_year"]
        assert item.metadata.authors == [a["author"]["display_name"] for a in raw["authorships"]]
        assert item.metadata.doi == raw["doi"]
    assert "FAKE" not in json.dumps([i.metadata.model_dump() for i in result.selected])


def test_premature_finish_flow(settings):
    script = [
        SEARCH1,
        turn(fc("finish_research", reason="Looks good.")),
        turn(fc("record_screening", rationale="Screen all.", assessments=[
            assess(W(1), "Relevant"), assess(W(2), "Not Relevant"), assess(W(3), "Not Relevant"),
            assess(W(4), "Insufficient Information", "No abstract."), assess(W(5), "Not Relevant"),
        ])),
        turn(fc("finish_research", reason="Stopping.")),
        turn(fc("finish_research", reason="Confirmed: more searching unlikely to help.")),
    ]
    result, llm, _, _ = run(script, settings)
    assert [t.error_code for t in result.tool_history] == [
        None, "UNSCREENED_PAPERS_REMAIN", None, "PREMATURE_FINISH_CONFIRM", None,
    ]
    assert tool_responses(llm.requests[2])[0]["code"] == "UNSCREENED_PAPERS_REMAIN"
    assert result.status == "partial" and result.terminated_by == "model_finish"
    assert "1/3" in result.message


def test_step_cap_terminates(settings):
    bad = turn(fc("record_screening", rationale="Try.", assessments=[assess("W1", "Relevant")]))
    s = settings.model_copy(update={"max_model_steps": 3})
    result, llm, _, _ = run([SEARCH1, bad, bad], s)
    assert result.terminated_by == "step_cap" and len(llm.requests) == 3
    assert result.status == "partial"


def test_parallel_calls_handled(settings):
    script = [
        turn(
            fc("search_papers", query="adaptive learning", rationale="Broad query."),
            fc("search_papers", query="learning analytics", rationale="Complementary angle in parallel."),
        ),
        screen_all("Relevant"),
        turn(fc("finish_research", reason="Done.")),
    ]
    result, llm, _, _ = run(script, settings)
    last = llm.requests[1][-1]
    assert last.role == "user" and len(last.parts) == 2
    assert [p.function_response.name for p in last.parts] == ["search_papers", "search_papers"]
    first, second = tool_responses(llm.requests[1])
    assert first["stats"]["returned"] == 8 and second["stats"]["returned"] == 6
    assert second["stats"]["dropped_duplicate"] == 3  # overlap with the first parallel call
    assert result.status == "completed" and result.searches_used == 2


def test_thought_signature_preserved(settings):
    step1 = types.Content(role="model", parts=[
        types.Part(
            function_call=types.FunctionCall(name="search_papers", args={
                "query": "adaptive learning", "rationale": "Initial broad query on the topic."}),
            thought_signature=b"sig-1",
        )
    ])
    script = [step1, screen_all("Relevant"), turn(fc("finish_research", reason="Done."))]
    result, llm, _, _ = run(script, settings)
    assert llm.requests[1][1] is step1  # same object, not rebuilt
    assert llm.requests[1][1].parts[0].thought_signature == b"sig-1"
    assert llm.requests[2][1] is step1
    assert result.status == "completed"


def test_text_without_tool_call_nudged_once_then_stops(settings):
    result, llm, _, _ = run([turn(text="I am done."), turn(text="Really done.")], settings)
    assert result.terminated_by == "model_text_stop" and len(llm.requests) == 2
    assert llm.requests[1][-1].role == "user" and "finish_research" in llm.requests[1][-1].parts[0].text
    assert result.status == "error"


def test_llm_error_ends_run_gracefully(settings):
    result, _, _, _ = run([LLMError("LLM_QUOTA", "429 quota exceeded")], settings)
    assert result.status == "error" and result.terminated_by == "llm_error"
    assert "Gemini" in result.message and result.errors


# --- guards and validation ---------------------------------------------------------------------
def test_year_clamped_and_logged(settings):
    script = [
        turn(fc("search_papers", query="adaptive learning", year_from=2000.0, year_to=2026,
                rationale="Try a wide range.")),
        screen_all(),
    ]
    result, _, backend, _ = run(script, settings.model_copy(update={"max_model_steps": 2}))
    assert backend.calls[0]["year_from"] == 2019 and backend.calls[0]["year_to"] == 2024
    guards = [e["data"]["kind"] for e in read_log(settings, result.run_id) if e["event"] == "guard"]
    assert "YEAR_CLAMPED" in guards


def test_year_range_outside_request_rejected_without_cost(settings):
    script = [turn(fc("search_papers", query="adaptive learning", year_from=2000, year_to=2010, rationale="Old."))]
    s = settings.model_copy(update={"max_model_steps": 1})
    result, _, backend, _ = run(script, s)
    assert result.tool_history[0].error_code == "INVALID_ARGS"
    assert result.searches_used == 0 and backend.calls == []


def test_argument_validation_codes(settings):
    script = [
        turn(
            fc("search_papers", query="   ", rationale="Empty query."),
            fc("do_magic", rationale="Unknown tool."),
            fc("search_papers", query="x", limit="abc", rationale="Bad limit."),
            fc("search_papers", query="adaptive learning"),  # missing rationale
            fc("search_papers", query="adaptive learning", year_from=2020.0, rationale="Float year is fine."),
        ),
    ]
    s = settings.model_copy(update={"max_model_steps": 1})
    result, _, backend, _ = run(script, s)
    assert [t.error_code for t in result.tool_history] == [
        "EMPTY_QUERY", "UNKNOWN_TOOL", "INVALID_ARGS", "INVALID_ARGS", None,
    ]
    assert result.searches_used == 1 and backend.calls[0]["year_from"] == 2020


def test_find_related_requires_eligible_seed(settings):
    script = [
        SEARCH1,
        turn(fc("find_related_papers", seed_paper_id=W(1), rationale="Expand from first paper.")),
        turn(fc("record_screening", rationale="Screen.", assessments=[
            assess(W(1), "Relevant"), assess(W(2), "Not Relevant"), assess(W(3), "Not Relevant"),
            assess(W(4), "Not Relevant", "No abstract."), assess(W(5), "Not Relevant"),
        ])),
        turn(fc("find_related_papers", seed_paper_id="W123", rationale="Invented seed.")),
        turn(fc("find_related_papers", seed_paper_id=W(1), rationale="Seed is relevant and keywords saturate.")),
        screen_all("Relevant"),
        turn(fc("finish_research", reason="Done.")),
    ]
    result, _, backend, _ = run(script, settings)
    assert [t.error_code for t in result.tool_history] == [
        None, "SEED_NOT_ELIGIBLE", None, "UNKNOWN_PAPER_ID", None, None, None,
    ]
    assert backend.calls[-1]["op"] == "related" and result.searches_used == 2
    assert result.status == "completed"


def test_registry_declares_four_tools_without_metadata_fields(settings):
    result, llm, _, _ = run([turn(text="x"), turn(text="y")], settings)
    declarations = llm.tools_seen[0][0].function_declarations
    assert [d.name for d in declarations] == [
        "search_papers", "find_related_papers", "record_screening", "finish_research",
    ]
    schema_text = json.dumps([d.parameters_json_schema for d in declarations]).lower()
    for forbidden in ("doi", "author", "\"year\"", "abstract\""):
        assert forbidden not in schema_text
    for d in declarations:
        assert "rationale" in d.parameters_json_schema["properties"] or d.name == "finish_research"


# --- instructions / criteria -------------------------------------------------------------------
def test_system_instruction_content_and_criteria_loading(settings, tmp_path):
    criteria = tmp_path / "criteria.md"
    criteria.write_text("UNIQUE-CRITERIA-MARKER: custom rules", encoding="utf-8")
    s = settings.model_copy(update={"criteria_path": criteria})
    _, llm, _, _ = run([turn(text="x"), turn(text="y")], s)
    system = llm.system_instructions[0]
    assert "UNIQUE-CRITERIA-MARKER" in system
    assert "untrusted" in system.lower() and "ignore previous instructions" in system.lower()
    first_user = llm.requests[0][0].parts[0].text
    assert "adaptive learning" in first_user and "2019" in first_user and "Vietnamese" in first_user


def test_missing_criteria_falls_back_with_warning(settings):
    result, llm, _, _ = run([turn(text="x"), turn(text="y")], settings)  # settings point to a missing file
    assert "Relevant" in llm.system_instructions[0]
    kinds = [e["data"].get("kind") for e in read_log(settings, result.run_id) if e["event"] == "guard"]
    assert "CRITERIA_FALLBACK" in kinds


# --- logging ------------------------------------------------------------------------------------
def test_every_tool_call_has_call_and_result_events(settings):
    script = [SEARCH1, screen_all("Relevant"), turn(fc("finish_research", reason="Done."))]
    result, _, _, events = run(script, settings)
    log = read_log(settings, result.run_id)
    calls = [e for e in log if e["event"] == "tool_call"]
    results = [e for e in log if e["event"] == "tool_result"]
    assert len(calls) == len(results) == len(result.tool_history) == 3
    assert all(e["data"]["rationale"] for e in calls[:2])
    assert all("latency_ms" in e["data"] and "summary" in e["data"] for e in results)
    names = [e["event"] for e in log]
    assert names[0] == "run_started" and names[-1] == "run_finished"
    assert "model_request" in names and "model_response" in names
    assert [e["event"] for e in events] == names  # on_event mirrors the file
    assert results[0]["data"]["summary"].startswith("search_papers('adaptive learning') -> 8 trả về")


def test_secrets_never_reach_logs_or_results(settings):
    backend = FakeOpenAlex(fail_with=OpenAlexError("OPENALEX_HTTP", f"upstream echoed {FAKE_OPENALEX_KEY} back"))
    script = [
        turn(fc("search_papers", query="q one", rationale=f"use {FAKE_GEMINI_KEY} as query?")),
        turn(fc("search_papers", query="q two", rationale="Retry.")),
        LLMError("LLM_ERROR", f"bad request key={FAKE_GEMINI_KEY}"),
    ]
    result, _, _, _ = run(script, settings, backend=backend)
    blob = (settings.log_dir / f"{result.run_id}.jsonl").read_text(encoding="utf-8") + result.model_dump_json()
    assert FAKE_GEMINI_KEY not in blob and FAKE_OPENALEX_KEY not in blob
    assert "***" in blob


# ----- from test_llm.py ----------------------------------------
KEY = "gem-secret-key-777"


def ok_response(text="hello"):
    content = types.Content(role="model", parts=[types.Part(text=text)])
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=content, finish_reason=types.FinishReason.STOP)],
        usage_metadata=types.GenerateContentResponseUsageMetadata(prompt_token_count=5, candidates_token_count=2),
    )


class StubClient:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []
        self.models = self

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def make(outcomes, **kw):
    sleeps = []
    stub = StubClient(outcomes)
    client = GeminiClient(KEY, "m", client=stub, sleep=sleeps.append, min_interval_seconds=0, **kw)
    return client, stub, sleeps


def api_error(cls, code, message="boom"):
    return cls(code, {"error": {"message": message, "status": "X"}})


def test_success_returns_original_content_and_config():
    resp = ok_response("hi")
    client, stub, _ = make([resp])
    tool = types.Tool(function_declarations=[types.FunctionDeclaration(name="t", description="d")])
    out = client.generate([types.Content(role="user", parts=[types.Part(text="q")])], [tool], "sys")
    assert out.content is resp.candidates[0].content  # untouched
    assert out.text == "hi" and out.finish_reason == "STOP"
    assert out.usage == {"prompt_token_count": 5, "candidates_token_count": 2}
    cfg = stub.calls[0]["config"]
    assert cfg.automatic_function_calling.disable is True
    assert cfg.tool_config.function_calling_config.mode == types.FunctionCallingConfigMode.AUTO
    assert cfg.temperature is None and cfg.system_instruction == "sys"


def test_temperature_passed_only_when_set():
    client, stub, _ = make([ok_response()], temperature=0.3)
    client.generate([], [], "s")
    assert stub.calls[0]["config"].temperature == 0.3


def test_retries_429_with_backoff_then_succeeds():
    client, stub, sleeps = make([api_error(errors.ClientError, 429), api_error(errors.ClientError, 429), ok_response()])
    assert client.generate([], [], "s").text == "hello"
    assert sleeps == [4.0, 8.0] and len(stub.calls) == 3


def test_gives_up_after_three_retries_quota():
    client, stub, sleeps = make([api_error(errors.ClientError, 429)] * 4)
    with pytest.raises(LLMError) as exc:
        client.generate([], [], "s")
    assert exc.value.code == "LLM_QUOTA" and len(stub.calls) == 4 and len(sleeps) == 3


def test_503_maps_to_unavailable():
    client, _, _ = make([api_error(errors.ServerError, 503)] * 4)
    with pytest.raises(LLMError) as exc:
        client.generate([], [], "s")
    assert exc.value.code == "LLM_UNAVAILABLE"


def test_400_not_retried_and_key_redacted():
    client, stub, sleeps = make([api_error(errors.ClientError, 400, f"bad key {KEY}")])
    with pytest.raises(LLMError) as exc:
        client.generate([], [], "s")
    assert exc.value.code == "LLM_ERROR" and len(stub.calls) == 1 and sleeps == []
    assert KEY not in exc.value.message


def test_network_error_retried():
    client, _, sleeps = make([httpx.ConnectError("down"), ok_response()])
    assert client.generate([], [], "s").text == "hello" and sleeps == [4.0]


def test_empty_candidates_retried_then_fails():
    empty = types.GenerateContentResponse(candidates=[])
    client, stub, _ = make([empty] * 4)
    with pytest.raises(LLMError):
        client.generate([], [], "s")
    assert len(stub.calls) == 4


def test_min_interval_throttle():
    now = {"t": 100.0}
    sleeps = []
    stub = StubClient([ok_response(), ok_response()])
    client = GeminiClient(
        KEY, "m", client=stub, min_interval_seconds=2.0, sleep=sleeps.append, clock=lambda: now["t"]
    )
    client.generate([], [], "s")
    now["t"] += 0.5
    client.generate([], [], "s")
    assert sleeps == [pytest.approx(1.5)]


def test_no_tools_means_no_tool_config():
    client, stub, _ = make([ok_response()])
    client.generate([], [], "sys")
    cfg = stub.calls[0]["config"]
    assert cfg.tools is None and cfg.tool_config is None and cfg.system_instruction == "sys"
