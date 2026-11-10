"""JSONL run logger with secret redaction, plus an optional live event callback."""
from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

logger = logging.getLogger(__name__)

EventCallback = Callable[[Dict[str, Any]], None]


def redact_secrets(text: str, secrets: Iterable[str]) -> str:
    """Replace every secret value in `text` with `***`."""
    for secret in secrets:
        if secret:
            text = text.replace(secret, "***")
    return text


class RunLogger:
    """Writes one JSON object per line to `{log_dir}/{run_id}.jsonl`.

    Each line: {"ts", "run_id", "step", "event", "data": {...}}. Secrets are redacted from the
    serialised line before it is written or passed to `on_event`.
    """

    def __init__(
        self,
        run_id: str,
        log_dir: Path,
        secrets: Iterable[str] = (),
        on_event: Optional[EventCallback] = None,
    ) -> None:
        self.run_id = run_id
        self.path = Path(log_dir) / f"{run_id}.jsonl"
        self._secrets = [s for s in secrets if s]
        self._on_event = on_event
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            logger.warning("Cannot create log directory %s", self.path.parent)

    def redact(self, text: str) -> str:
        """Redact secrets from arbitrary text."""
        return redact_secrets(text, self._secrets)

    def log(self, event: str, step: Optional[int] = None, **data: Any) -> Dict[str, Any]:
        """Append one event and return the (redacted) record."""
        record = {
            "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
            "run_id": self.run_id,
            "step": step,
            "event": event,
            "data": data,
        }
        line = self.redact(json.dumps(record, ensure_ascii=False, default=str))
        try:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError:
            logger.warning("Cannot write to run log %s", self.path)
        safe = json.loads(line)
        if self._on_event is not None:
            try:
                self._on_event(safe)
            except Exception:  # a broken listener must never break the run
                logger.exception("on_event callback failed")
        return safe

    def guard(self, kind: str, step: Optional[int] = None, **data: Any) -> None:
        """Log a code-side guard decision (budget refused, bad args, invented id, early finish...)."""
        self.log("guard", step, kind=kind, **data)
