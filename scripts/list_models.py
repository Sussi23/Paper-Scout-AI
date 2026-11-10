"""List Gemini models available to GEMINI_API_KEY that support generateContent.

Usage (from repo root):  python scripts/list_models.py
Never prints the API key.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from google import genai  # noqa: E402

from backend.config import ConfigError, load_settings  # noqa: E402


def main() -> int:
    try:
        settings = load_settings(strict=False)
    except ConfigError as exc:
        print(exc)
        return 2
    key = settings.gemini_key()
    if not key:
        print("GEMINI_API_KEY is missing in .env")
        return 2
    client = genai.Client(api_key=key)
    rows = []
    try:
        for model in client.models.list():
            actions = model.supported_actions or []
            if "generateContent" not in actions:
                continue
            name = (model.name or "").replace("models/", "")
            thinking = "thinking" if getattr(model, "thinking", None) else ""
            rows.append((name, model.display_name or "", model.input_token_limit or 0, thinking))
    except Exception as exc:  # show a safe message only
        msg = str(exc)
        for secret in settings.secret_values():
            msg = msg.replace(secret, "***")
        print(f"Failed to list models: {type(exc).__name__}: {msg[:500]}")
        return 1
    rows.sort()
    print(f"{'MODEL (use as GEMINI_MODEL)':<45} {'DISPLAY NAME':<40} {'INPUT TOKENS':>12}  NOTE")
    for name, display, limit, thinking in rows:
        print(f"{name:<45} {display[:40]:<40} {limit:>12}  {thinking}")
    print(f"\n{len(rows)} models support generateContent.")
    print(f"Current GEMINI_MODEL in config: {settings.gemini_model or '(not set)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
