"""Scan what WOULD be committed (or what IS committed) for leaked secrets. Never prints secret values.

Usage (from repo root, any OS):
    python scripts/scan_secrets.py              # working tree: files from `git ls-files -co --exclude-standard`
    python scripts/scan_secrets.py --committed  # content of HEAD plus all commit messages

Checks (reports only file, line number and the NAME of the matching variable/pattern):
  1. values of .env variables whose name contains KEY, TOKEN, SECRET or PASSWORD
  2. Google API key shapes, `api_key=<value>` in URLs, Bearer tokens
  3. personal absolute paths (Windows user profile, macOS /Users/..., Linux /home/...)
  4. history: `git log --all -S<value>` for every secret value (value passed to git, never printed)
Exit code 1 if anything is found, 0 if clean, 2 on usage/setup errors.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent.parent
SELF = Path(__file__).resolve().relative_to(ROOT).as_posix()
SENSITIVE_NAME = re.compile(r"KEY|TOKEN|SECRET|PASSWORD", re.IGNORECASE)
MIN_SECRET_LEN = 8  # shorter values would cause meaningless matches

# Patterns are assembled from pieces so this file does not match its own checks.
_SLASH = "/"
PATTERNS: List[Tuple[str, "re.Pattern[str]"]] = [
    ("google-api-key-shape", re.compile("AI" + r"za[0-9A-Za-z_\-]{35}")),
    ("api_key-in-url", re.compile(r"[?&]api_?key=(?!YOUR|<|\{|\$|%)[A-Za-z0-9_\-\.]{12,}", re.IGNORECASE)),
    ("bearer-token", re.compile(r"Bearer\s+(?!YOUR|<|\{|\$|%)[A-Za-z0-9_\-\.=]{20,}")),
    ("personal-path-windows", re.compile(r"[A-Za-z]:\\+Users\\+(?!<local>|Public|\*)[^\\\s\"']+", re.IGNORECASE)),
    ("personal-path-unix", re.compile("(?:" + _SLASH + "Users" + _SLASH + "|" + _SLASH + "home" + _SLASH + r")(?!<local>)[A-Za-z0-9_.\-]+")),
]


def git(*args: str, check: bool = True) -> bytes:
    """Run git in the repo root and return stdout bytes."""
    proc = subprocess.run(["git", *args], cwd=ROOT, capture_output=True)
    if check and proc.returncode != 0:
        raise RuntimeError(f"git {args[0]} failed")
    return proc.stdout


def load_secrets() -> Dict[str, str]:
    """name -> value for sensitive variables in .env (values stay in memory only)."""
    env_path = ROOT / ".env"
    if not env_path.is_file():
        return {}
    return {
        name: value.strip()
        for name, value in dotenv_values(env_path).items()
        if value and SENSITIVE_NAME.search(name) and len(value.strip()) >= MIN_SECRET_LEN
    }


def has_commits() -> bool:
    return subprocess.run(["git", "rev-parse", "--verify", "-q", "HEAD"], cwd=ROOT, capture_output=True).returncode == 0


def list_working_files() -> List[Tuple[str, bytes]]:
    names = git("ls-files", "-co", "--exclude-standard", "-z").split(b"\0")
    out = []
    for raw in names:
        if not raw:
            continue
        rel = raw.decode("utf-8", "replace")
        if rel == ".env" or rel.startswith(".git/"):
            continue
        path = ROOT / rel
        if path.is_file():
            out.append((rel, path.read_bytes()))
    return out


def list_committed_files() -> List[Tuple[str, bytes]]:
    names = git("ls-tree", "-r", "--name-only", "-z", "HEAD").split(b"\0")
    return [(n.decode("utf-8", "replace"), git("show", f"HEAD:{n.decode('utf-8', 'replace')}")) for n in names if n]


def scan_content(rel: str, data: bytes, secrets: Dict[str, str]) -> List[str]:
    if b"\0" in data[:4096]:
        return []  # binary
    findings = []
    text = data.decode("utf-8", "replace")
    for lineno, line in enumerate(text.splitlines(), start=1):
        for name, value in secrets.items():
            if value in line:
                findings.append(f"{rel}:{lineno}: contains the VALUE of variable {name}")
        if rel == SELF:
            continue
        for label, pattern in PATTERNS:
            if pattern.search(line):
                findings.append(f"{rel}:{lineno}: matches pattern {label}")
    return findings


def scan_history(secrets: Dict[str, str]) -> List[str]:
    if not has_commits():
        return []
    findings = []
    messages = git("log", "--all", "--format=%B").decode("utf-8", "replace")
    for name, value in secrets.items():
        hits = git("log", "--all", "--oneline", f"-S{value}").decode("utf-8", "replace").strip().splitlines()
        if hits:
            findings.append(f"history: value of {name} appears in {len(hits)} commit(s) (see `git log --all -S<value>`)")
        if value in messages:
            findings.append(f"history: value of {name} appears in a commit message")
    return findings


def main(argv: Optional[List[str]] = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    committed = "--committed" in args
    try:
        git("rev-parse", "--is-inside-work-tree")
    except (RuntimeError, FileNotFoundError):
        print("Not a git repository (run `git init` first).")
        return 2
    secrets = load_secrets()
    print(f"Sensitive variables checked (names only): {', '.join(sorted(secrets)) or '(none found in .env)'}")
    if committed and not has_commits():
        print("No commits yet; nothing committed to scan.")
        return 2
    files = list_committed_files() if committed else list_working_files()
    print(f"Scanning {len(files)} files ({'HEAD' if committed else 'working tree, respecting .gitignore'}).")
    findings: List[str] = []
    for rel, data in files:
        findings += scan_content(rel, data, secrets)
    findings += scan_history(secrets)
    if findings:
        print(f"\nLEAKS FOUND ({len(findings)}):")
        for line in findings:
            print("  " + line)
        return 1
    print("CLEAN: no secret values, key-shaped strings, bearer tokens or personal paths found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
