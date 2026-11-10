"""The agent loop: Gemini decides what to do, this code executes, validates and decides the outcome."""
from __future__ import annotations

import datetime as dt
import logging
import uuid
from typing import Any, Dict, List, Optional

from google.genai import types

from backend.agent.instructions import build_system_instruction, load_criteria, task_message
from backend.agent.llm import GeminiClient, LLMClient, LLMError
from backend.agent.run_logger import EventCallback, RunLogger
from backend.agent.schemas import RelevanceLabel, ResearchRequest, ResearchResult, SelectedPaper
from backend.agent.state import RunState
from backend.agent.tool_registry import ToolRegistry
from backend.config import Settings, load_settings
from backend.tools.search import OpenAlexClient, SearchBackend

logger = logging.getLogger(__name__)

NUDGE_TEXT = (
    "You answered without calling a tool. Either continue (search_papers / find_related_papers / "
    "record_screening) or call finish_research if you are done."
)

_MESSAGES = {
    "vi": {
        "completed": "Đã tìm đủ {relevant}/{target} bài Relevant sau {searches} lượt tìm.",
        "completed_over": "Đã tìm đủ: {relevant} bài Relevant (mục tiêu {target}) sau {searches} lượt tìm; hiển thị {target} bài.",
        "partial": "Chỉ tìm được {relevant}/{target} bài Relevant sau {searches} lượt tìm (có {possibly} bài Possibly Relevant).",
        "no_search": "Không có lượt tìm nào thành công, chưa có kết quả nào.",
        "llm": "Gemini gặp sự cố ({detail}); đã dừng sớm. Hiện có {relevant}/{target} bài Relevant.",
    },
    "en": {
        "completed": "Found {relevant}/{target} Relevant papers after {searches} searches.",
        "completed_over": "Target met: {relevant} Relevant papers found (target {target}) after {searches} searches; showing {target}.",
        "partial": "Only {relevant}/{target} Relevant papers found after {searches} searches ({possibly} Possibly Relevant).",
        "no_search": "No search succeeded, so there are no results.",
        "llm": "Gemini failed ({detail}); stopped early. Currently {relevant}/{target} Relevant papers.",
    },
}


def build_result(state: RunState, model: str, finished_at: Optional[dt.datetime] = None) -> ResearchResult:
    """Compute the final result. Status and metadata come from code and the registry, not the model."""
    request = state.request
    counts = state.label_counts()
    relevant = counts[RelevanceLabel.RELEVANT.value]
    possibly = counts[RelevanceLabel.POSSIBLY_RELEVANT.value]
    ids = state.selected_ids()
    selected = [
        SelectedPaper(
            metadata=state.registry[pid],
            label=state.assessments[pid].label,
            reason=state.assessments[pid].reason,
        )
        for pid in ids
    ]
    msgs = _MESSAGES[request.language]
    fmt = dict(relevant=relevant, target=request.num_papers, searches=state.searches_used, possibly=possibly)
    llm_failed = state.terminated_by == "llm_error"
    if llm_failed:
        status = "error"
        message = msgs["llm"].format(detail=state.errors[-1] if state.errors else "unknown", **fmt)
    elif state.successful_searches == 0:
        status, message = "error", msgs["no_search"]
    elif relevant >= request.num_papers:
        status = "completed"
        message = msgs["completed" if relevant == request.num_papers else "completed_over"].format(**fmt)
    else:
        status, message = "partial", msgs["partial"].format(**fmt)
    return ResearchResult(
        run_id=state.run_id,
        status=status,
        message=message,
        request=request,
        selected_ids=ids,
        selected=selected,
        label_counts=counts,
        searches_used=state.searches_used,
        queries=list(state.queries),
        tool_history=list(state.tool_history),
        errors=list(state.errors),
        terminated_by=state.terminated_by,
        model=model,
        started_at=state.started_at,
        finished_at=finished_at or dt.datetime.now(dt.timezone.utc),
    )


def _build_defaults(settings: Settings, llm: Optional[LLMClient], backend: Optional[SearchBackend]):
    if llm is None or backend is None:
        settings.require_ready()
    if llm is None:
        llm = GeminiClient(
            settings.gemini_key() or "",
            settings.gemini_model or "",
            temperature=settings.gemini_temperature,
            min_interval_seconds=settings.gemini_min_interval_seconds,
            secrets=settings.secret_values(),
        )
    if backend is None:
        backend = OpenAlexClient(settings.openalex_key() or "")
    return llm, backend


def run_research(
    request: ResearchRequest,
    *,
    llm: Optional[LLMClient] = None,
    search_backend: Optional[SearchBackend] = None,
    settings: Optional[Settings] = None,
    on_event: Optional[EventCallback] = None,
) -> ResearchResult:
    """Run one research request end to end and return the result (never raises for tool/LLM failures)."""
    settings = settings or load_settings(strict=False)
    owns_backend = search_backend is None
    llm, backend = _build_defaults(settings, llm, search_backend)
    run_id = uuid.uuid4().hex[:12]
    log = RunLogger(run_id, settings.log_dir, settings.secret_values(), on_event)
    state = RunState(request=request, settings=settings, run_id=run_id)
    registry = ToolRegistry(settings, backend)
    model_name = getattr(llm, "model", None) or settings.gemini_model or ""

    log.log("run_started", None, request=request.model_dump(mode="json"), model=model_name,
            max_searches=settings.max_searches, max_model_steps=settings.max_model_steps)
    criteria_text, warning = load_criteria(settings.criteria_path)
    if warning:
        log.guard("CRITERIA_FALLBACK", None, message=warning)
    system_instruction = build_system_instruction(criteria_text)

    contents: List[types.Content] = [
        types.Content(role="user", parts=[types.Part.from_text(text=task_message(request, settings))])
    ]
    tools = registry.tools()

    for step in range(1, settings.max_model_steps + 1):
        log.log("model_request", step, messages=len(contents))
        try:
            resp = llm.generate(contents, tools, system_instruction)
        except LLMError as exc:
            message = log.redact(f"{exc.code}: {exc.message}")
            state.errors.append(message)
            state.terminated_by = "llm_error"
            log.log("model_error", step, code=exc.code, message=message)
            break
        log.log(
            "model_response", step, text=resp.text,
            function_calls=[{"name": c.name, "args": dict(c.args or {})} for c in resp.function_calls],
            usage=resp.usage, finish_reason=resp.finish_reason,
        )
        # Keep the model's Content object exactly as received (preserves thought signatures).
        contents.append(resp.content)

        calls = resp.function_calls
        if not calls:
            if not state.text_nudged:
                state.text_nudged = True
                log.guard("TEXT_WITHOUT_TOOL_CALL", step)
                contents.append(types.Content(role="user", parts=[types.Part.from_text(text=NUDGE_TEXT)]))
                continue
            state.terminated_by = "model_text_stop"
            break

        response_parts: List[types.Part] = []
        for call in calls:  # parallel calls: handled in order, one response per call
            result: Dict[str, Any] = registry.dispatch(call, state, step, log)
            response_parts.append(types.Part.from_function_response(name=call.name or "", response=result))
        contents.append(types.Content(role="user", parts=response_parts))

        if state.finished:
            state.terminated_by = "model_finish"
            break
        if state.searches_remaining == 0 and not state.unscreened_ids():
            state.terminated_by = "budget"  # nothing useful left to do; skip a pointless extra model call
            break
    else:
        state.terminated_by = "step_cap"

    if owns_backend and hasattr(backend, "close"):
        backend.close()
    result = build_result(state, model_name)
    log.log("run_finished", None, status=result.status, terminated_by=result.terminated_by,
            relevant=result.label_counts.get(RelevanceLabel.RELEVANT.value, 0), searches_used=result.searches_used)
    return result
