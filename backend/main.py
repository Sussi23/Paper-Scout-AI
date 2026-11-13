"""FastAPI app. Run from the repo root:  uvicorn backend.main:app --reload"""
from __future__ import annotations

import json
import logging
import re
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from fastapi import Depends, FastAPI, Path as PathParam, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from backend.agent.research_agent import run_research
from backend.agent.schemas import ResearchRequest, ResearchResult
from backend.config import ConfigError, Settings, load_settings
from backend.tools.export import export_csv

logger = logging.getLogger(__name__)

RUN_ID_PATTERN = r"^[a-f0-9]{12}$"
_RUN_ID_RE = re.compile(RUN_ID_PATTERN)
QUEUE_TIMEOUT_SECONDS = 90.0

ResearchFn = Callable[..., ResearchResult]


# --- dependencies (overridable in tests via app.dependency_overrides) ----------------------------
def get_settings(request: Request) -> Settings:
    """Settings attached to the app at creation time."""
    return request.app.state.settings


def get_research_fn() -> ResearchFn:
    """The function that runs one research request. Tests override this with a fake."""
    return run_research


def _run_dir(settings: Settings, run_id: str) -> Path:
    if not _RUN_ID_RE.match(run_id):  # defence in depth; the route already enforces the pattern
        raise ValueError("invalid run id")
    return settings.runs_dir / run_id


def _error(status: int, message: str, **extra: Any) -> JSONResponse:
    return JSONResponse(status_code=status, content={"detail": message, **extra})


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    """Application factory (tests pass their own Settings)."""
    settings = settings or load_settings(strict=False)
    app = FastAPI(title="PaperScout AI", version="1.0.0")
    app.state.settings = settings
    app.state.semaphore = threading.BoundedSemaphore(settings.max_concurrent_runs)
    app.state.queue_timeout = QUEUE_TIMEOUT_SECONDS

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.allowed_origins),
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["Content-Disposition"],
    )

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        problems: List[Dict[str, str]] = []
        for err in exc.errors():
            field = ".".join(str(p) for p in err.get("loc", ()) if p != "body")
            message = str(err.get("msg", "invalid value")).removeprefix("Value error, ")
            problems.append({"field": field, "message": message})
        summary = "; ".join(f"{p['field']}: {p['message']}" if p["field"] else p["message"] for p in problems)
        return JSONResponse(status_code=422, content={"detail": f"Invalid request: {summary}", "errors": problems})

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error")
        return _error(500, "Internal server error.")

    @app.get("/health")
    def health(current: Settings = Depends(get_settings)) -> Dict[str, Any]:
        return {
            "status": "ok",
            "model": current.gemini_model,
            "gemini_key_configured": bool(current.gemini_key()),
            "openalex_key_configured": bool(current.openalex_key()),
        }

    @app.post("/api/research", response_model=ResearchResult)
    def research(
        body: ResearchRequest,
        request: Request,
        current: Settings = Depends(get_settings),
        run_fn: ResearchFn = Depends(get_research_fn),
    ):
        semaphore: threading.BoundedSemaphore = request.app.state.semaphore
        if not semaphore.acquire(timeout=request.app.state.queue_timeout):
            return _error(429, "The server is busy with another research run. Please try again shortly.")
        try:
            result = run_fn(body, settings=current)
        except ConfigError as exc:
            return _error(503, f"Server configuration incomplete: {exc}")
        except Exception:
            logger.exception("Research run failed unexpectedly")
            return _error(500, "Internal server error while running the research.")
        finally:
            semaphore.release()

        try:
            run_dir = _run_dir(current, result.run_id)
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "result.json").write_text(
                json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8"
            )
            export_csv(result, run_dir / "results.csv")
        except (OSError, ValueError):
            logger.exception("Could not persist run %s", result.run_id)
        return result

    @app.get("/api/runs/{run_id}")
    def get_run(
        run_id: str = PathParam(..., pattern=RUN_ID_PATTERN), current: Settings = Depends(get_settings)
    ):
        path = _run_dir(current, run_id) / "result.json"
        if not path.is_file():
            return _error(404, "Run not found.")
        return JSONResponse(content=json.loads(path.read_text(encoding="utf-8")))

    @app.get("/api/exports/{run_id}")
    def get_export(
        run_id: str = PathParam(..., pattern=RUN_ID_PATTERN), current: Settings = Depends(get_settings)
    ):
        path = _run_dir(current, run_id) / "results.csv"
        if not path.is_file():
            return _error(404, "Export not found.")
        return FileResponse(path, media_type="text/csv", filename=f"paperscout_{run_id}.csv")

    return app


app = create_app()
