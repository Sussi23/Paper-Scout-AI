# PaperScout AI — Backend

A real AI agent (Gemini function calling through `google-genai`) that searches the OpenAlex scholarly database
and screens papers for relevance, exposed as a FastAPI service for the website. This folder is the **backend only**.

- **Gemini decides**: which tool to call, which keywords to use, how to change the query after seeing results,
  and when to stop.
- **Code decides**: executing tools, validating arguments, the search budget, the source of truth for metadata,
  error handling, the final status, and logging.
- Title, authors, year, DOI, abstract and URL come **only from OpenAlex** (kept in a registry keyed by `paper_id`).
  Gemini returns just `paper_id` + label + reason.

All commands below are run from the **repository root** (the folder that contains `backend/`), not from `backend/`.

<!-- BEGIN: FOR TV5 (copy into the root README) -->
## Installation

Requires Python 3.10 or newer (tested on 3.10.0).

```bash
python -m venv .venv
# Windows PowerShell:  .venv\Scripts\Activate.ps1      macOS/Linux:  source .venv/bin/activate
pip install -r backend/requirements.txt
cp .env.example .env        # Windows PowerShell: Copy-Item .env.example .env
```

Fill in `.env` (never commit it): `GEMINI_API_KEY`, `OPENALEX_API_KEY`, and `GEMINI_MODEL`. There is no default
model on purpose; run `python scripts/list_models.py` to see which models your key can use.

## How to Run

```bash
uvicorn backend.main:app --reload      # API on http://127.0.0.1:8000, interactive docs at /docs
```

Check it with `GET /health`. One research run takes about 1-2 minutes, so a client must wait at least 3 minutes.

## Testing

```bash
pip install -r backend/requirements-dev.txt
pytest                                  # offline: no network, no keys needed
```

Live tests that call the real Gemini/OpenAlex APIs (they use quota) are off by default; see "Testing" below.
<!-- END: FOR TV5 -->

## 1. Configuration

Read from environment variables, or from a `.env` file in the repository root (environment variables win).
The server starts without keys (so `/health` works), but `POST /api/research` returns 503 until the required
ones are set. `.env.example` lists every name.

| Variable | Required | Default | Meaning |
|---|---|---|---|
| `GEMINI_API_KEY` | yes | none | Google AI Studio API key |
| `GEMINI_MODEL` | yes | none | Gemini model name (see `scripts/list_models.py`) |
| `GEMINI_TEMPERATURE` | no | model default | Leave empty unless you are experimenting |
| `GEMINI_MIN_INTERVAL_SECONDS` | no | `2` | Minimum gap between Gemini calls (helps avoid 429) |
| `OPENALEX_API_KEY` | yes | none | OpenAlex key, sent as `Authorization: Bearer` |
| `MAX_SEARCHES` | no | `3` | Search budget per run; `search_papers` and `find_related_papers` both count |
| `MAX_MODEL_STEPS` | no | `12` | Hard cap on model turns per run |
| `SEARCH_PAGE_SIZE` | no | `25` | Results per search (also the cap on a tool's `limit`) |
| `ABSTRACT_CHARS_FOR_MODEL` | no | `1200` | Abstract characters shown to the model |
| `MAX_CONCURRENT_RUNS` | no | `1` | Parallel runs; others wait up to 90 s, then get HTTP 429 |
| `ALLOWED_ORIGINS` | no | empty | Comma-separated frontend origins allowed by CORS |
| `LOG_DIR` | no | `logs` | Where per-run JSONL logs are written |
| `RUNS_DIR` | no | `runs` | Where `result.json` and `results.csv` are saved |
| `CRITERIA_PATH` | no | `evaluation/screening_criteria.md` | Screening criteria, re-read on every run |

Keys are never printed: they are masked (`***`) in logs, error messages, `tool_history` and API output, and
`/health` only reports whether each key is configured.

## 2. API contract

Base URL in development: `http://127.0.0.1:8000`. Interactive documentation (Swagger UI): `/docs`.

### `POST /api/research`

Runs the agent synchronously (about 1-2 minutes) and returns the result.

Request body:

| Field | Type | Rules |
|---|---|---|
| `topic` | string | required, 3-300 characters after trimming |
| `year_from`, `year_to` | integer or null | 1900 to next year; `year_from <= year_to` |
| `num_papers` | integer | 1-30, default 10 |
| `criteria` | string or null | extra screening criteria, at most 1000 characters |
| `language` | `"vi"` or `"en"` | language of each paper's `reason`, default `"vi"` |

```json
{ "topic": "machine learning for medical image diagnosis", "num_papers": 8, "language": "en" }
```

Response (excerpt of [`demo/sample_result.json`](../demo/sample_result.json), a real run; long values shortened):

```json
{
  "run_id": "e29b2bbd33a4",
  "status": "completed",
  "message": "Found 16/8 Relevant papers after 1 searches.",
  "selected_ids": ["W3169509062", "W3173991394", "..."],
  "selected": [
    {
      "metadata": {
        "paper_id": "W3169509062",
        "title": "Machine learning for medical imaging-based COVID-19 detection and diagnosis",
        "authors": ["Rokaya Rehouma", "Michael Büchert"],
        "year": 2021,
        "doi": "https://doi.org/10.1002/int.22504",
        "source_url": "https://doi.org/10.1002/int.22504",
        "abstract": "The novel coronavirus disease 2019 (COVID-19) is considered to be ...",
        "source": "openalex"
      },
      "label": "Relevant",
      "reason": "The paper focuses on machine learning applications using medical imaging ..."
    }
  ],
  "label_counts": { "Relevant": 16, "Possibly Relevant": 1, "Not Relevant": 0, "Insufficient Information": 8 },
  "searches_used": 1,
  "queries": ["machine learning medical image diagnosis"],
  "tool_history": [
    {
      "step": 1, "tool": "search_papers",
      "args": { "query": "machine learning medical image diagnosis", "rationale": "..." },
      "rationale": "Initial search to find papers focusing on ...",
      "status": "ok", "error_code": null,
      "summary": "search_papers('machine learning medical image diagnosis') -> 25 trả về, 25 mới",
      "latency_ms": 1873, "timestamp": "2026-11-09T16:45:17.165387Z"
    }
  ],
  "errors": [],
  "terminated_by": "model_finish",
  "model": "gemini-3.5-flash",
  "started_at": "2026-11-09T16:45:10.953374Z",
  "finished_at": "2026-11-09T16:45:35.185462Z"
}
```

(The sample was recorded before the wording of `message` was changed for runs that exceed the target; such a
run now says "Target met: 16 Relevant papers found (target 8) ...". Do not parse `message`; use `status`.)

`status` is computed by code, not by the model:

| `status` | Meaning |
|---|---|
| `completed` | number of Relevant papers >= `num_papers` |
| `partial` | fewer Relevant papers than requested, but at least one search succeeded (`message` says the real `X/Y`) |
| `error` | no search succeeded, or Gemini failed and could not recover (see `errors`, `terminated_by`) |

**`completed`, `partial` and `error` are all returned with HTTP 200; the status is always inside the JSON.**
`selected` holds all Relevant papers, then Possibly Relevant ones, in discovery order, cut at `num_papers`.
`label_counts` counts every labelled paper, including Not Relevant and Insufficient Information.
`terminated_by` is one of `model_finish`, `model_text_stop`, `step_cap`, `budget`, `llm_error`.

Other HTTP codes: `422` invalid input (`{"detail": "Invalid request: ...", "errors": [{"field", "message"}]}`),
`429` the server is busy with another run (waited 90 s), `503` the server is missing required configuration
(the message names the missing variables), `500` unexpected error (no internal details).

**Timeouts:** a run usually takes 1-2 minutes (screening 25 papers can take 15-30 s of Gemini time). The frontend
must use a client timeout of at least 3 minutes and show a waiting state.

### `GET /api/runs/{run_id}`

Returns the saved `result.json` for a run, including `tool_history` for the Agent Activity screen. `404` if unknown.

### `GET /api/exports/{run_id}`

Returns the CSV of `selected` (`text/csv`, UTF-8 with BOM so Excel reads Vietnamese correctly, filename
`paperscout_{run_id}.csv`; columns Title, Authors, Year, DOI, Source URL, Abstract, Relevance, Relevance Reason).
`404` if unknown. `run_id` must match `^[a-f0-9]{12}$`; anything else is rejected (`422`, or `404` when the value
contains path separators and so does not match the route), which prevents path traversal.
The same CSV for the sample run is in [`demo/sample_results.csv`](../demo/sample_results.csv).

### `GET /health`

`{"status": "ok", "model": "...", "gemini_key_configured": true, "openalex_key_configured": true}`.

## 3. Testing

```bash
pytest                       # default: offline, uses FakeLLM / FakeOpenAlex and synthetic fixtures
```

The live smoke test (`tests/test_live_smoke.py`) calls the real APIs and uses quota. It needs real keys in `.env`:

```powershell
# Windows PowerShell
$env:RUN_LIVE = "1"; pytest -m live
```
```bash
# bash / zsh
RUN_LIVE=1 pytest -m live
```

Test files: `test_agent.py` (agent loop, state, tool registry, run logger, Gemini client), `test_tools.py`
(OpenAlex client, search, processing, CSV export), `test_api.py` (FastAPI). Smaller extras: `test_config.py`,
`test_schemas.py`, `test_structure.py`, `test_baseline.py`, `test_show_log.py`, `test_scan_secrets.py`,
`test_live_smoke.py`. Shared fakes and fixtures: `conftest.py`, `fakes.py`, `fixtures/`.

## 4. Layout

```
backend/
  main.py              FastAPI app (create_app, dependencies)
  config.py            settings from .env / environment, validated
  requirements.txt     runtime dependencies, pinned     requirements-dev.txt   + pytest
  agent/
    research_agent.py  run_research(): the agent loop and final result
    instructions.py    system instruction, task message, loading the screening criteria
    schemas.py         Pydantic models (request, metadata, assessment, result)
    state.py           per-run state: paper registry, assessments, budget
    llm.py             Gemini client (throttling, retry/backoff), LLMClient protocol
    tool_registry.py   the 4 tool declarations, validation, dispatch, error codes
    run_logger.py      JSONL logger with secret redaction
  tools/
    search.py          OpenAlex client and the search / related-works operations
    processing.py      normalize, filter by year, de-duplicate
    export.py          CSV export
scripts/               helper scripts, run from the repo root (see below)
demo/                  logs/ (3 real demo runs), sample_result.json, sample_results.csv
tests/                 offline test-suite
```

Scripts:

| Script | Purpose | Uses API quota |
|---|---|---|
| `scripts/list_models.py` | list Gemini models that support `generateContent` | no cost to speak of |
| `scripts/check_connection.py` | Gemini smoke test (text + function calling) | 2 tiny requests |
| `scripts/check_openalex.py` | OpenAlex smoke test (search + related works) | 1 search |
| `scripts/live_demo.py` | 3 real requests; writes `demo/logs/` and `demo/results/` | yes |
| `scripts/show_log.py` | print a readable timeline of a run log | no |
| `scripts/baseline_script.py` | non-agent baseline (fixed keywords, always 3 searches) for comparison | yes |
| `scripts/scan_secrets.py` | scan files/history for leaked secrets (never prints values) | no |
| `scripts/check_env_example.py` | compare variable names in `config.py` and `.env.example` | no |
| `scripts/verify_clean_clone.py` | clone the repo to a temp dir, install, test and smoke-start the server | no |

`python scripts/show_log.py demo/logs/e29b2bbd33a4.jsonl` shows the timeline of the sample run.

## 5. How the agent loop works

```
state = RunState(request); contents = [task message]
for step in 1..MAX_MODEL_STEPS:
    response = gemini.generate(contents, tools, system_instruction)     # throttled, retried on 429/503
    contents.append(response.content)       # the model's own Content object, unchanged (keeps thought signatures)
    for each function call (parallel calls allowed, handled in order):
        result = registry.dispatch(call, state)       # validate -> execute -> catch errors -> log
    contents.append(user Content: one function_response per call, same order)
    stop when finish_research is accepted, or the budget is spent and every paper is screened
result = build_result(state)      # status and metadata computed by code from the registry
```

Tools (manual function calling; automatic function calling is disabled, mode AUTO). Every tool takes a
`rationale` (why this step, based on earlier results), which is logged and shown in `tool_history`.

| Tool | Cost | Notes |
|---|---|---|
| `search_papers(query, rationale, year_from?, year_to?, limit?)` | 1 search | years are clamped to the request's range; only papers not seen before are returned |
| `find_related_papers(seed_paper_id, rationale, limit?)` | 1 search | the seed must be labelled Relevant or Possibly Relevant; uses OpenAlex `related_works` |
| `record_screening(assessments, rationale)` | free | per-item validation; bad items are rejected, good ones recorded; relabelling allowed |
| `finish_research(reason)` | free | rejected while papers are unscreened; the first early finish must be confirmed |

Error codes returned to the model: `SEARCH_BUDGET_EXHAUSTED`, `UNKNOWN_TOOL`, `INVALID_ARGS`, `EMPTY_QUERY`,
`UNKNOWN_PAPER_ID`, `INVALID_LABEL`, `SEED_NOT_ELIGIBLE`, `UNSCREENED_PAPERS_REMAIN`, `PREMATURE_FINISH_CONFIRM`,
`OPENALEX_AUTH`, `OPENALEX_RATE_LIMIT`, `OPENALEX_HTTP`, `OPENALEX_TIMEOUT`, and `INTERNAL_ERROR` for an unexpected
exception inside a tool.

Titles and abstracts are treated as untrusted data; the system instruction tells the model to ignore any
instructions found inside them. Screening criteria come from `CRITERIA_PATH` on every run (change the file, change
the behaviour); if the file is missing, a built-in default is used and a warning is logged.

Each run writes `logs/{run_id}.jsonl`, one event per line: `run_started`, `model_request`, `model_response`,
`tool_call`, `tool_result`, `guard` (budget refused, unknown tool, bad arguments, invented id, early finish, year
clamped), `run_finished`.

## 6. Deploying on Render

- Build command: `pip install -r backend/requirements.txt`
- Start command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
- `PYTHON_VERSION`: set it to a 3.10.x release (the version this backend was tested on; 3.11 was not tested).
- Health check path: `/health`
- Environment variables to declare in the Render dashboard: `GEMINI_API_KEY`, `GEMINI_MODEL`, `OPENALEX_API_KEY`,
  `ALLOWED_ORIGINS` (the deployed frontend origin, e.g. `https://your-site.example`); optional: `GEMINI_TEMPERATURE`,
  `GEMINI_MIN_INTERVAL_SECONDS`, `MAX_SEARCHES`, `MAX_MODEL_STEPS`, `SEARCH_PAGE_SIZE`, `ABSTRACT_CHARS_FOR_MODEL`,
  `MAX_CONCURRENT_RUNS`, `LOG_DIR`, `RUNS_DIR`, `CRITERIA_PATH`.
- The disk is ephemeral: `runs/` and `logs/` are lost on restart or redeploy, so `GET /api/runs/{id}` and
  `GET /api/exports/{id}` only work for recent runs. Free instances also sleep when idle (the first request after
  a pause is slow).

## 7. Known limits

- Tested with Python 3.10.0 only.
- A run takes about 1-2 minutes; the endpoint is synchronous, and with `MAX_CONCURRENT_RUNS=1` a second user waits
  (up to 90 s) or gets HTTP 429.
- Relevance labels are Gemini's judgement from title and abstract. There is no reference evaluation set yet, so
  label quality has not been measured.
- The rule "a paper without an abstract can never be Relevant" is only in the prompt and the screening criteria;
  the code does not enforce it. (In the three demo runs no Relevant paper lacked an abstract.)
- Free-tier quotas: Gemini limits depend on the model and key (on 429/503 the client backs off 4, 8, 16 s, at most
  3 retries, then the run ends with `status: "error"`); OpenAlex's free key gives a small daily budget (about
  USD 1/day; a keyword search costs about USD 0.001).
- `scripts/baseline_script.py` (the non-agent comparison) is covered by offline tests but has not been run against
  the real APIs yet, so there are no baseline results in `demo/`.
- `demo/logs/` holds the three real demo runs: popular topic (`e29b2bbd33a4`), narrow topic (`398069570191`) and
  tight year range (`f06d72185141`).
- `evaluation/screening_criteria.md` is a draft owned by the evaluation member and is not part of this repository
  snapshot; without it the built-in default criteria are used.

## 8. Sources

The code was written from scratch against the current Gemini function-calling documentation and the OpenAlex API
documentation (API key authentication, `search`, `select`, `related_works`). The reference repository
`academic-online-research-agent` was not used and no code was copied from it.
