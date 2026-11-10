"""Tool declarations for Gemini, argument validation, execution and error mapping.

The model proposes; this module decides. Every exception inside a tool becomes a coded error
result so the agent loop never crashes. Metadata never flows back from the model.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from google.genai import types

from backend.agent.run_logger import RunLogger, redact_secrets
from backend.agent.schemas import MIN_YEAR, PaperMetadata, RelevanceLabel, ToolCallLog, max_year
from backend.agent.state import RunState
from backend.config import Settings
from backend.tools.processing import normalize_title
from backend.tools.search import OpenAlexError, SearchBackend, SearchOutcome, find_related, search_openalex

logger = logging.getLogger(__name__)

# --- error codes -----------------------------------------------------------------------------
SEARCH_BUDGET_EXHAUSTED = "SEARCH_BUDGET_EXHAUSTED"
UNKNOWN_TOOL = "UNKNOWN_TOOL"
INVALID_ARGS = "INVALID_ARGS"
EMPTY_QUERY = "EMPTY_QUERY"
UNKNOWN_PAPER_ID = "UNKNOWN_PAPER_ID"
INVALID_LABEL = "INVALID_LABEL"
SEED_NOT_ELIGIBLE = "SEED_NOT_ELIGIBLE"
UNSCREENED_PAPERS_REMAIN = "UNSCREENED_PAPERS_REMAIN"
PREMATURE_FINISH_CONFIRM = "PREMATURE_FINISH_CONFIRM"
INTERNAL_ERROR = "INTERNAL_ERROR"  # extra code: unexpected exception inside a tool

MAX_REASON_CHARS = 600
MAX_RATIONALE_LOG_CHARS = 500
MAX_IDS_IN_MESSAGE = 30


class ToolError(Exception):
    """Raised by tool handlers to produce a coded error result."""

    def __init__(self, code: str, message: str, retryable: bool = False, **extra: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.extra = extra


@dataclass
class ToolOutcome:
    """What a handler returns: the model-facing response, a one-line summary, and log status."""

    response: Dict[str, Any]
    summary: str
    status: str = "ok"
    error_code: Optional[str] = None


# --- declarations ------------------------------------------------------------------------------
_RATIONALE = {
    "type": "string",
    "description": "1-2 sentences: why you chose this step, based on what you observed in earlier results.",
}


def _declarations() -> List[types.FunctionDeclaration]:
    return [
        types.FunctionDeclaration(
            name="search_papers",
            description=(
                "Keyword search on OpenAlex. Costs 1 search from a limited budget. Returns only papers "
                "not seen before in this run (title, year, abstract) plus stats, budget and progress."
            ),
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search keywords, preferably English academic terms."},
                    "rationale": _RATIONALE,
                    "year_from": {"type": "integer", "description": "Optional. Clamped to the user's year range."},
                    "year_to": {"type": "integer", "description": "Optional. Clamped to the user's year range."},
                    "limit": {"type": "integer", "description": "Optional. Max results (default and cap: page size)."},
                },
                "required": ["query", "rationale"],
            },
        ),
        types.FunctionDeclaration(
            name="find_related_papers",
            description=(
                "Find papers related to a seed paper (OpenAlex related works). Costs 1 search from the same "
                "budget. The seed must already be labelled Relevant or Possibly Relevant."
            ),
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "seed_paper_id": {"type": "string", "description": "A paper_id returned by an earlier tool call."},
                    "rationale": _RATIONALE,
                    "limit": {"type": "integer", "description": "Optional. Max results."},
                },
                "required": ["seed_paper_id", "rationale"],
            },
        ),
        types.FunctionDeclaration(
            name="record_screening",
            description=(
                "Record your relevance label and short reason for papers you have read. Free (no budget). "
                "You may relabel a paper by recording it again. Only paper_id, label and reason are accepted."
            ),
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "assessments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "paper_id": {"type": "string"},
                                "label": {"type": "string", "enum": [label.value for label in RelevanceLabel]},
                                "reason": {"type": "string", "description": "1-3 sentences, based only on the title/abstract."},
                            },
                            "required": ["paper_id", "label", "reason"],
                        },
                    },
                    "rationale": _RATIONALE,
                },
                "required": ["assessments", "rationale"],
            },
        ),
        types.FunctionDeclaration(
            name="finish_research",
            description=(
                "Declare the research finished. Rejected while unscreened papers remain. If fewer Relevant "
                "papers than the target were found and searches remain, the first call asks for confirmation."
            ),
            parameters_json_schema={
                "type": "object",
                "properties": {"reason": {"type": "string", "description": "Why you are stopping now."}},
                "required": ["reason"],
            },
        ),
    ]


# --- argument helpers --------------------------------------------------------------------------
def _opt_int(name: str, value: Any) -> Optional[int]:
    """Accept ints, integral floats (protobuf numbers) and digit strings; None stays None."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ToolError(INVALID_ARGS, f"'{name}' must be an integer.")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    raise ToolError(INVALID_ARGS, f"'{name}' must be an integer.")


def _required_str(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ToolError(INVALID_ARGS, f"'{name}' is required and must be a non-empty string.")
    return value.strip()


def _year(name: str, value: Any) -> Optional[int]:
    year = _opt_int(name, value)
    if year is not None and not MIN_YEAR <= year <= max_year():
        raise ToolError(INVALID_ARGS, f"'{name}' must be between {MIN_YEAR} and {max_year()}.")
    return year


def _paper_view(paper: PaperMetadata, max_chars: int) -> Dict[str, Any]:
    """Compact paper record for the model: no DOI/authors, abstract truncated."""
    abstract = paper.abstract
    truncated = abstract is not None and len(abstract) > max_chars
    return {
        "paper_id": paper.paper_id,
        "title": paper.title,
        "year": paper.year,
        "has_abstract": abstract is not None,
        "abstract": abstract[:max_chars] if abstract is not None else None,
        "abstract_truncated": truncated,
    }


def error_response(code: str, message: str, retryable: bool = False, **extra: Any) -> Dict[str, Any]:
    """The standard error payload returned to the model."""
    return {"status": "error", "code": code, "message": message, "retryable": retryable, **extra}


class ToolRegistry:
    """Owns tool declarations and executes calls against a RunState."""

    def __init__(self, settings: Settings, backend: SearchBackend) -> None:
        self._settings = settings
        self._backend = backend
        self._handlers = {
            "search_papers": self._search_papers,
            "find_related_papers": self._find_related_papers,
            "record_screening": self._record_screening,
            "finish_research": self._finish_research,
        }

    def tools(self) -> List[types.Tool]:
        """Tool objects to pass to Gemini."""
        return [types.Tool(function_declarations=_declarations())]

    # -- dispatch ---------------------------------------------------------------------------
    def dispatch(self, call: types.FunctionCall, state: RunState, step: int, log: RunLogger) -> Dict[str, Any]:
        """Validate -> execute -> catch errors -> log. Always returns a JSON-serialisable dict."""
        name = call.name or ""
        raw_args = call.args if isinstance(call.args, dict) else dict(call.args or {})
        rationale = raw_args.get("rationale") if isinstance(raw_args.get("rationale"), str) else None
        if rationale is None and isinstance(raw_args.get("reason"), str):
            rationale = raw_args["reason"]
        if rationale:
            rationale = rationale[:MAX_RATIONALE_LOG_CHARS]
        log.log("tool_call", step, name=name, args=raw_args, rationale=rationale)

        started = time.perf_counter()
        secrets = state.settings.secret_values()
        try:
            handler = self._handlers.get(name)
            if handler is None:
                raise ToolError(UNKNOWN_TOOL, f"Unknown tool '{name}'. Available: {', '.join(self._handlers)}.")
            if state.finished:
                raise ToolError(INVALID_ARGS, "Research is already finished; no further tool calls are accepted.")
            outcome = handler(raw_args, state, step, log)
        except ToolError as exc:
            message = redact_secrets(exc.message, secrets)
            outcome = ToolOutcome(
                response=error_response(exc.code, message, exc.retryable, **exc.extra),
                summary=f"{name} -> {exc.code}: {message}"[:300],
                status="rejected",
                error_code=exc.code,
            )
            log.guard(exc.code, step, tool=name, message=message)
        except OpenAlexError as exc:
            message = redact_secrets(exc.message, secrets)
            state.errors.append(f"{name}: {exc.code}: {message}")
            outcome = ToolOutcome(
                response=error_response(exc.code, message, exc.retryable),
                summary=f"{name} -> {exc.code}: {message}"[:300],
                status="error",
                error_code=exc.code,
            )
        except Exception as exc:  # last line of defence: the loop must not crash
            logger.exception("Unexpected error in tool %s", name)
            message = redact_secrets(f"Unexpected internal error: {type(exc).__name__}", secrets)
            state.errors.append(f"{name}: {INTERNAL_ERROR}: {message}")
            outcome = ToolOutcome(
                response=error_response(INTERNAL_ERROR, message, True),
                summary=f"{name} -> {INTERNAL_ERROR}",
                status="error",
                error_code=INTERNAL_ERROR,
            )
        latency_ms = int((time.perf_counter() - started) * 1000)
        state.tool_history.append(
            ToolCallLog(
                step=step,
                tool=name,
                args=json.loads(redact_secrets(json.dumps(raw_args, default=str), secrets)),
                rationale=redact_secrets(rationale, secrets) if rationale else rationale,
                status=outcome.status,  # type: ignore[arg-type]
                error_code=outcome.error_code,
                summary=outcome.summary,
                latency_ms=latency_ms,
            )
        )
        log.log(
            "tool_result", step, name=name, status=outcome.status, error_code=outcome.error_code,
            summary=outcome.summary, latency_ms=latency_ms,
        )
        return outcome.response

    # -- shared helpers ---------------------------------------------------------------------
    def _effective_years(self, state: RunState, asked_from: Optional[int], asked_to: Optional[int], step: int, log: RunLogger):
        """Clamp model-requested years into the request's range; log YEAR_CLAMPED if changed."""
        req_from, req_to = state.request.year_from, state.request.year_to
        eff_from = asked_from if asked_from is not None else req_from
        eff_to = asked_to if asked_to is not None else req_to
        clamped = False
        if req_from is not None and eff_from is not None and eff_from < req_from:
            eff_from, clamped = req_from, True
        if req_to is not None and eff_to is not None and eff_to > req_to:
            eff_to, clamped = req_to, True
        if eff_from is not None and eff_to is not None and eff_from > eff_to:
            raise ToolError(
                INVALID_ARGS,
                f"Year range {asked_from}-{asked_to} does not overlap the allowed range "
                f"{req_from or 'any'}-{req_to or 'any'}.",
            )
        if clamped:
            log.guard(
                "YEAR_CLAMPED", step,
                asked={"year_from": asked_from, "year_to": asked_to}, used={"year_from": eff_from, "year_to": eff_to},
            )
        return eff_from, eff_to

    def _limit(self, value: Any) -> int:
        asked = _opt_int("limit", value)
        cap = self._settings.search_page_size
        return cap if asked is None else max(1, min(asked, cap))

    def _consume_budget(self, state: RunState) -> None:
        if state.searches_remaining <= 0:
            raise ToolError(
                SEARCH_BUDGET_EXHAUSTED,
                f"Search budget exhausted ({state.settings.max_searches}/{state.settings.max_searches} used). "
                "Screen the remaining papers and call finish_research.",
                retryable=False,
                budget=state.budget(),
            )
        state.searches_used += 1

    def _search_result(self, state: RunState, outcome: SearchOutcome) -> Dict[str, Any]:
        state.register(outcome.new_papers)
        state.successful_searches += 1
        limit = self._settings.abstract_chars_for_model
        return {
            "status": "ok",
            "new_papers": [_paper_view(p, limit) for p in outcome.new_papers],
            "stats": outcome.stats.as_dict(),
            "budget": state.budget(),
            "progress": state.progress(),
        }

    @staticmethod
    def _search_summary(label: str, outcome: SearchOutcome) -> str:
        s = outcome.stats
        parts = [f"{s.returned} trả về", f"{len(outcome.new_papers)} mới"]
        if s.dropped_year:
            parts.append(f"{s.dropped_year} loại do năm")
        if s.dropped_duplicate:
            parts.append(f"{s.dropped_duplicate} trùng")
        if s.dropped_invalid:
            parts.append(f"{s.dropped_invalid} không hợp lệ")
        return f"{label} -> " + ", ".join(parts)

    # -- handlers ---------------------------------------------------------------------------
    def _search_papers(self, args: Dict[str, Any], state: RunState, step: int, log: RunLogger) -> ToolOutcome:
        query = args.get("query")
        if query is None or (isinstance(query, str) and not query.strip()):
            raise ToolError(EMPTY_QUERY, "'query' must not be empty.")
        query = _required_str("query", query)
        _required_str("rationale", args.get("rationale"))
        asked_from = _year("year_from", args.get("year_from"))
        asked_to = _year("year_to", args.get("year_to"))
        limit = self._limit(args.get("limit"))
        eff_from, eff_to = self._effective_years(state, asked_from, asked_to, step, log)

        repeated = normalize_title(query) in {normalize_title(q) for q in state.queries}
        self._consume_budget(state)  # a call that reaches the network always costs budget
        state.queries.append(query)
        outcome = search_openalex(
            self._backend, query, year_from=eff_from, year_to=eff_to, limit=limit, seen=state.seen
        )
        response = self._search_result(state, outcome)
        if repeated:
            response["note"] = "This query was already used earlier; prefer a different formulation."
        return ToolOutcome(response, self._search_summary(f"search_papers({query!r})", outcome))

    def _find_related_papers(self, args: Dict[str, Any], state: RunState, step: int, log: RunLogger) -> ToolOutcome:
        seed = _required_str("seed_paper_id", args.get("seed_paper_id"))
        _required_str("rationale", args.get("rationale"))
        limit = self._limit(args.get("limit"))
        if seed not in state.registry:
            raise ToolError(UNKNOWN_PAPER_ID, f"seed_paper_id '{seed}' is not a paper returned in this run.")
        assessment = state.assessments.get(seed)
        if assessment is None or assessment.label not in (RelevanceLabel.RELEVANT, RelevanceLabel.POSSIBLY_RELEVANT):
            raise ToolError(
                SEED_NOT_ELIGIBLE,
                f"Seed '{seed}' must first be labelled Relevant or Possibly Relevant via record_screening.",
            )
        eff_from, eff_to = self._effective_years(state, None, None, step, log)
        self._consume_budget(state)
        outcome = find_related(
            self._backend, seed, year_from=eff_from, year_to=eff_to, limit=limit, seen=state.seen
        )
        return ToolOutcome(self._search_result(state, outcome), self._search_summary(f"find_related_papers({seed})", outcome))

    def _record_screening(self, args: Dict[str, Any], state: RunState, step: int, log: RunLogger) -> ToolOutcome:
        from backend.agent.schemas import PaperAssessment  # local import keeps module header small

        _required_str("rationale", args.get("rationale"))
        items = args.get("assessments")
        if not isinstance(items, list) or not items:
            raise ToolError(INVALID_ARGS, "'assessments' must be a non-empty list.")
        accepted: List[str] = []
        rejected: List[Dict[str, str]] = []
        relabeled = 0

        def reject(pid: Any, code: str, message: str) -> None:
            rejected.append({"paper_id": str(pid), "code": code, "message": message})
            log.guard(code, step, tool="record_screening", paper_id=str(pid), message=message)

        for item in items:
            if not isinstance(item, dict):
                reject("?", INVALID_ARGS, "Each assessment must be an object {paper_id, label, reason}.")
                continue
            pid = item.get("paper_id")
            if not isinstance(pid, str) or pid not in state.registry:
                reject(pid, UNKNOWN_PAPER_ID, "paper_id was not returned by a search in this run.")
                continue
            try:
                label = RelevanceLabel(item.get("label"))
            except ValueError:
                allowed = ", ".join(l.value for l in RelevanceLabel)
                reject(pid, INVALID_LABEL, f"label must be one of: {allowed}.")
                continue
            reason = item.get("reason")
            if not isinstance(reason, str) or len(reason.strip()) < 5:
                reject(pid, INVALID_ARGS, "reason must be a non-empty string (at least 5 characters).")
                continue
            reason = reason.strip()[:MAX_REASON_CHARS]
            previous = state.assessments.get(pid)
            if previous is not None and previous.label != label:
                relabeled += 1
                log.log("relabel", step, paper_id=pid, old=previous.label.value, new=label.value)
            # only paper_id/label/reason are read; any extra field the model sent is ignored
            state.assessments[pid] = PaperAssessment(paper_id=pid, label=label, reason=reason)
            accepted.append(pid)

        summary = f"record_screening -> {len(accepted)} ghi nhận, {len(rejected)} từ chối"
        if relabeled:
            summary += f", {relabeled} gắn nhãn lại"
        if not accepted:
            first = rejected[0]["code"] if rejected else INVALID_ARGS
            return ToolOutcome(
                error_response(first, "No assessment was accepted; see 'rejected'.", True,
                               rejected=rejected, progress=state.progress()),
                summary, status="rejected", error_code=first,
            )
        response: Dict[str, Any] = {
            "status": "ok", "accepted": accepted, "rejected": rejected, "progress": state.progress()
        }
        return ToolOutcome(response, summary)

    def _finish_research(self, args: Dict[str, Any], state: RunState, step: int, log: RunLogger) -> ToolOutcome:
        reason = _required_str("reason", args.get("reason"))
        unscreened = state.unscreened_ids()
        if unscreened:
            shown = ", ".join(unscreened[:MAX_IDS_IN_MESSAGE])
            raise ToolError(
                UNSCREENED_PAPERS_REMAIN,
                f"{len(unscreened)} papers are not screened yet: {shown}. Call record_screening first.",
                unscreened_ids=unscreened,
            )
        progress = state.progress()
        if progress["relevant_so_far"] < progress["target"] and state.searches_remaining > 0 and not state.finish_confirm_asked:
            state.finish_confirm_asked = True
            raise ToolError(
                PREMATURE_FINISH_CONFIRM,
                f"{state.searches_remaining} searches remain and only {progress['relevant_so_far']}/"
                f"{progress['target']} Relevant papers were found. Finish only if you judge that searching "
                "more would not help: state why and call finish_research again to confirm.",
            )
        state.finished = True
        state.finish_reason = reason
        return ToolOutcome({"status": "ok", "message": "Research finished."}, "finish_research -> đã kết thúc")
