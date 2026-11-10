"""Repository layout and secret hygiene (must also pass in a fresh clone: no .env, no logs/, no runs/)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_required_dirs_exist():
    for rel in ["backend/agent", "backend/tools", "scripts", "tests/fixtures", "demo/logs"]:
        assert (ROOT / rel).is_dir(), rel


def test_gitignore_protects_secrets_and_outputs():
    lines = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    for pattern in [".env", ".env.*", "!.env.example", "/logs/", "/runs/", ".venv/"]:
        assert pattern in lines, pattern
    assert "logs/" not in lines and "*.jsonl" not in lines  # unanchored/broad rules would hide demo/logs


def test_env_example_lists_all_settings_without_secrets():
    from backend.config import ENV_NAMES

    lines = (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
    entries = dict(line.split("=", 1) for line in lines if "=" in line and not line.startswith("#"))
    assert set(ENV_NAMES) <= set(entries)
    assert entries["GEMINI_API_KEY"] == "" and entries["OPENALEX_API_KEY"] == "" and entries["GEMINI_MODEL"] == ""
