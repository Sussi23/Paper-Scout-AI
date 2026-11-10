"""Print a readable timeline of a run log (JSONL) — handy for demos and the defence.

Usage:  python scripts/show_log.py logs/<run_id>.jsonl      (or just the run id)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO_ROOT = Path(__file__).resolve().parent.parent


def _clip(text: Any, n: int = 160) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


def format_event(rec: Dict[str, Any]) -> List[str]:
    """Turn one log record into one or more display lines."""
    ts = str(rec.get("ts", ""))[11:19]
    step = rec.get("step")
    prefix = f"{ts}  step {step:>2}  " if step is not None else f"{ts}          "
    data = rec.get("data", {})
    event = rec.get("event")
    if event == "run_started":
        req = data.get("request", {})
        return [f"{prefix}RUN START  model={data.get('model')}  topic={req.get('topic')!r}  "
                f"years={req.get('year_from')}-{req.get('year_to')}  target={req.get('num_papers')}  "
                f"budget={data.get('max_searches')} searches"]
    if event == "model_request":
        return [f"{prefix}-> model ({data.get('messages')} messages)"]
    if event == "model_response":
        lines = []
        if data.get("text"):
            lines.append(f"{prefix}<- model says: {_clip(data['text'])}")
        calls = ", ".join(c["name"] for c in data.get("function_calls", [])) or "(no tool call)"
        usage = data.get("usage", {})
        lines.append(f"{prefix}<- model calls: {calls}  [tokens in={usage.get('prompt_token_count')} "
                     f"out={usage.get('candidates_token_count')}]")
        return lines
    if event == "tool_call":
        args = {k: v for k, v in data.get("args", {}).items() if k not in ("rationale", "assessments")}
        lines = [f"{prefix}TOOL  {data.get('name')}({_clip(json.dumps(args, ensure_ascii=False), 120)})"]
        if data.get("rationale"):
            lines.append(f"{prefix}      why: {_clip(data['rationale'])}")
        return lines
    if event == "tool_result":
        mark = {"ok": "ok", "rejected": "REJECTED", "error": "ERROR"}.get(data.get("status"), "?")
        return [f"{prefix}      {mark}: {_clip(data.get('summary'))}  ({data.get('latency_ms')} ms)"]
    if event == "guard":
        return [f"{prefix}GUARD {data.get('kind')}: {_clip(data.get('message') or data)}"]
    if event == "relabel":
        return [f"{prefix}RELABEL {data.get('paper_id')}: {data.get('old')} -> {data.get('new')}"]
    if event == "model_error":
        return [f"{prefix}MODEL ERROR {data.get('code')}: {_clip(data.get('message'))}"]
    if event == "run_finished":
        return [f"{prefix}RUN END  status={data.get('status')}  terminated_by={data.get('terminated_by')}  "
                f"relevant={data.get('relevant')}  searches_used={data.get('searches_used')}"]
    return [f"{prefix}{event} {_clip(data)}"]


def show(path: Path) -> None:
    """Print the timeline of the JSONL file at `path`."""
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            for out in format_event(json.loads(line)):
                print(out)


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    arg = Path(sys.argv[1])
    if not arg.is_file():
        arg = REPO_ROOT / "logs" / f"{sys.argv[1]}.jsonl"
    if not arg.is_file():
        print(f"Log not found: {sys.argv[1]}")
        return 1
    show(arg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
