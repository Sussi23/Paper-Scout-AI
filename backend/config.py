"""Application settings read from environment variables / `.env`, validated early.

Environment variables override values in `.env` (so Render's dashboard variables win).
Secrets are kept as `SecretStr` so they never show up in `repr()` or logs.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple, Union

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, SecretStr

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_FILE = REPO_ROOT / ".env"

ENV_NAMES = (
    "GEMINI_API_KEY",
    "GEMINI_MODEL",
    "GEMINI_TEMPERATURE",
    "GEMINI_MIN_INTERVAL_SECONDS",
    "OPENALEX_API_KEY",
    "MAX_SEARCHES",
    "MAX_MODEL_STEPS",
    "SEARCH_PAGE_SIZE",
    "ABSTRACT_CHARS_FOR_MODEL",
    "MAX_CONCURRENT_RUNS",
    "ALLOWED_ORIGINS",
    "LOG_DIR",
    "RUNS_DIR",
    "CRITERIA_PATH",
)

_MISSING_HINTS = {
    "GEMINI_API_KEY": "GEMINI_API_KEY is not set. Create a key at https://aistudio.google.com/apikey and put it in .env.",
    "GEMINI_MODEL": (
        "GEMINI_MODEL is not set (no default on purpose). Run `python scripts/list_models.py`, "
        "choose a model that supports generateContent and put it in .env, e.g. GEMINI_MODEL=<name>."
    ),
    "OPENALEX_API_KEY": "OPENALEX_API_KEY is not set. Get a free key at https://openalex.org/settings/api and put it in .env.",
}


class ConfigError(RuntimeError):
    """Raised when configuration is missing or invalid. Message is safe to show (no secrets)."""


class Settings(BaseModel):
    """Validated runtime settings."""

    model_config = ConfigDict(frozen=True)

    gemini_api_key: Optional[SecretStr] = None
    gemini_model: Optional[str] = None
    gemini_temperature: Optional[float] = None
    gemini_min_interval_seconds: float = 2.0
    openalex_api_key: Optional[SecretStr] = None
    max_searches: int = 3
    max_model_steps: int = 12
    search_page_size: int = 25
    abstract_chars_for_model: int = 1200
    max_concurrent_runs: int = 1
    allowed_origins: Tuple[str, ...] = ()
    log_dir: Path = REPO_ROOT / "logs"
    runs_dir: Path = REPO_ROOT / "runs"
    criteria_path: Path = REPO_ROOT / "evaluation" / "screening_criteria.md"

    # -- secret access -------------------------------------------------------------------
    def gemini_key(self) -> Optional[str]:
        """Return the raw Gemini key (or None)."""
        return self.gemini_api_key.get_secret_value() if self.gemini_api_key else None

    def openalex_key(self) -> Optional[str]:
        """Return the raw OpenAlex key (or None)."""
        return self.openalex_api_key.get_secret_value() if self.openalex_api_key else None

    def secret_values(self) -> List[str]:
        """All secret strings that must be redacted from logs and messages."""
        return [s for s in (self.gemini_key(), self.openalex_key()) if s]

    # -- readiness -----------------------------------------------------------------------
    def missing_required(self) -> List[str]:
        """Names of required variables that are not configured."""
        missing = []
        if not self.gemini_key():
            missing.append("GEMINI_API_KEY")
        if not self.gemini_model:
            missing.append("GEMINI_MODEL")
        if not self.openalex_key():
            missing.append("OPENALEX_API_KEY")
        return missing

    def require_ready(self) -> "Settings":
        """Raise ConfigError with actionable hints if a required variable is missing."""
        missing = self.missing_required()
        if missing:
            raise ConfigError("Configuration incomplete:\n- " + "\n- ".join(_MISSING_HINTS[m] for m in missing))
        return self


def _resolve_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else REPO_ROOT / path


def _parse_number(
    name: str, raw: Optional[str], default: Union[int, float, None], kind: type, lo: float, hi: float, errors: List[str]
) -> Union[int, float, None]:
    if raw is None or raw.strip() == "":
        return default
    try:
        value = kind(raw.strip())
    except ValueError:
        errors.append(f"{name}={raw!r} is not a valid {kind.__name__}.")
        return default
    if not (lo <= value <= hi):
        errors.append(f"{name}={value} must be between {lo} and {hi}.")
        return default
    return value


def load_settings(
    env_file: Union[str, Path, None] = DEFAULT_ENV_FILE,
    environ: Optional[Mapping[str, str]] = None,
    strict: bool = True,
) -> Settings:
    """Load settings from `env_file` then `environ` (environ wins).

    Args:
        env_file: path to a dotenv file, or None to skip it.
        environ: mapping to read instead of `os.environ` (useful in tests).
        strict: if True, raise ConfigError when required keys/model are missing.
    """
    values: Dict[str, str] = {}
    if env_file is not None and Path(env_file).is_file():
        for key, val in dotenv_values(env_file).items():
            if val is not None and val.strip() != "":
                values[key] = val.strip()
    env = os.environ if environ is None else environ
    for name in ENV_NAMES:
        val = env.get(name)
        if val is not None and val.strip() != "":
            values[name] = val.strip()

    errors: List[str] = []
    temperature = _parse_number("GEMINI_TEMPERATURE", values.get("GEMINI_TEMPERATURE"), None, float, 0.0, 2.0, errors)
    settings_kwargs = dict(
        gemini_api_key=SecretStr(values["GEMINI_API_KEY"]) if values.get("GEMINI_API_KEY") else None,
        gemini_model=values.get("GEMINI_MODEL") or None,
        gemini_temperature=temperature,
        gemini_min_interval_seconds=_parse_number(
            "GEMINI_MIN_INTERVAL_SECONDS", values.get("GEMINI_MIN_INTERVAL_SECONDS"), 2.0, float, 0.0, 60.0, errors
        ),
        openalex_api_key=SecretStr(values["OPENALEX_API_KEY"]) if values.get("OPENALEX_API_KEY") else None,
        max_searches=_parse_number("MAX_SEARCHES", values.get("MAX_SEARCHES"), 3, int, 1, 20, errors),
        max_model_steps=_parse_number("MAX_MODEL_STEPS", values.get("MAX_MODEL_STEPS"), 12, int, 1, 50, errors),
        search_page_size=_parse_number("SEARCH_PAGE_SIZE", values.get("SEARCH_PAGE_SIZE"), 25, int, 1, 100, errors),
        abstract_chars_for_model=_parse_number(
            "ABSTRACT_CHARS_FOR_MODEL", values.get("ABSTRACT_CHARS_FOR_MODEL"), 1200, int, 100, 10000, errors
        ),
        max_concurrent_runs=_parse_number(
            "MAX_CONCURRENT_RUNS", values.get("MAX_CONCURRENT_RUNS"), 1, int, 1, 10, errors
        ),
        allowed_origins=tuple(o.strip() for o in values.get("ALLOWED_ORIGINS", "").split(",") if o.strip()),
        log_dir=_resolve_path(values.get("LOG_DIR", "logs")),
        runs_dir=_resolve_path(values.get("RUNS_DIR", "runs")),
        criteria_path=_resolve_path(values.get("CRITERIA_PATH", "evaluation/screening_criteria.md")),
    )
    if errors:
        raise ConfigError("Invalid configuration:\n- " + "\n- ".join(errors))
    settings = Settings(**settings_kwargs)
    if strict:
        settings.require_ready()
    elif settings.missing_required():
        logger.warning("Configuration incomplete, missing: %s", ", ".join(settings.missing_required()))
    return settings
