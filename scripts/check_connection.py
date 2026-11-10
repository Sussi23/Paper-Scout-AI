"""Smoke-test the Gemini connection with a tiny prompt and a tiny function-calling round.

Usage (from repo root):  python scripts/check_connection.py
Costs 2 small Gemini requests. Never prints the API key.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from google import genai  # noqa: E402
from google.genai import types  # noqa: E402

from backend.config import ConfigError, load_settings  # noqa: E402


def _safe(msg: str, secrets: list) -> str:
    for secret in secrets:
        msg = msg.replace(secret, "***")
    return msg[:600]


def main() -> int:
    try:
        settings = load_settings(strict=False)
    except ConfigError as exc:
        print(exc)
        return 2
    if not settings.gemini_key() or not settings.gemini_model:
        print("Need GEMINI_API_KEY and GEMINI_MODEL in .env (run scripts/list_models.py to choose a model).")
        return 2
    client = genai.Client(api_key=settings.gemini_key())
    secrets = settings.secret_values()
    print(f"Model: {settings.gemini_model}")
    try:
        resp = client.models.generate_content(model=settings.gemini_model, contents="Reply with exactly: OK")
        print(f"[1/2] Plain text call -> {resp.text!r}")

        ping = types.FunctionDeclaration(
            name="ping",
            description="Echo a word back. Call this tool when asked to ping.",
            parameters_json_schema={
                "type": "object",
                "properties": {"word": {"type": "string"}},
                "required": ["word"],
            },
        )
        config = types.GenerateContentConfig(
            tools=[types.Tool(function_declarations=[ping])],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode=types.FunctionCallingConfigMode.AUTO)
            ),
        )
        resp = client.models.generate_content(
            model=settings.gemini_model, contents="Please ping with the word 'hello'.", config=config
        )
        calls = resp.function_calls or []
        print(f"[2/2] Function calling -> {[(c.name, dict(c.args or {})) for c in calls]}")
        if not calls:
            print("WARNING: model did not return a function call.")
            return 1
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {_safe(str(exc), secrets)}")
        return 1
    print("Gemini connection OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
