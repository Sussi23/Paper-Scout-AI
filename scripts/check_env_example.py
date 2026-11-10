"""Compare the environment variable names read by backend/config.py with those in .env.example.

Usage (from repo root):  python scripts/check_env_example.py
Prints only variable NAMES. Exit code 1 if the two sets differ or .env.example holds a non-empty secret.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NAME = re.compile(r"^[A-Z][A-Z0-9_]{2,}$")
SENSITIVE = re.compile(r"KEY|TOKEN|SECRET|PASSWORD")


def names_read_by_config() -> set:
    """All UPPER_CASE string constants in config.py (the env var names it looks up)."""
    tree = ast.parse((ROOT / "backend" / "config.py").read_text(encoding="utf-8"))
    return {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str) and NAME.match(n.value)}


def example_entries() -> dict:
    entries = {}
    for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("#") and "=" in line:
            name, value = line.split("=", 1)
            entries[name.strip()] = value.strip()
    return entries


def main() -> int:
    read, listed = names_read_by_config(), example_entries()
    problems = []
    for name in sorted(read - set(listed)):
        problems.append(f"read by config.py but missing from .env.example: {name}")
    for name in sorted(set(listed) - read):
        problems.append(f"in .env.example but never read by config.py: {name}")
    for name, value in listed.items():
        if SENSITIVE.search(name) and value:
            problems.append(f"secret-looking variable has a value in .env.example: {name}")
    print(f"config.py reads {len(read)} names; .env.example lists {len(listed)}.")
    if problems:
        print("\n".join("  " + p for p in problems))
        return 1
    print("OK: names match and no secret values are present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
