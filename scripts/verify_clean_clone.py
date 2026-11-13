"""Verify that a clean clone of the repository installs, passes its tests and starts without a .env file.

Usage (from anywhere inside the repo, any OS):
    python scripts/verify_clean_clone.py [--branch backend] [--expect-passed 124]

Steps: clone the branch into a temp dir OUTSIDE the repo -> create a venv -> install
backend/requirements-dev.txt -> run pytest with every GEMINI_*/OPENALEX_* variable removed ->
start uvicorn on a free port -> check /health, /docs and that POST /api/research returns a clear JSON error
(HTTP 503 naming the missing variables) -> stop the server -> delete the temp dir this script created.
Needs network access for `pip install`. Uses no API quota and never prints secrets.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
CONFIG_PREFIXES = ("GEMINI_", "OPENALEX_")
CONFIG_NAMES = {
    "MAX_SEARCHES", "MAX_MODEL_STEPS", "SEARCH_PAGE_SIZE", "ABSTRACT_CHARS_FOR_MODEL", "MAX_CONCURRENT_RUNS",
    "ALLOWED_ORIGINS", "LOG_DIR", "RUNS_DIR", "CRITERIA_PATH",
}
EXPECTED_MISSING = ("GEMINI_API_KEY", "GEMINI_MODEL", "OPENALEX_API_KEY")


def venv_python(venv: Path) -> Path:
    """Interpreter inside a venv, by platform."""
    return venv / ("Scripts/python.exe" if sys.platform.startswith("win") else "bin/python")


def clean_env() -> Dict[str, str]:
    """Current environment without any app configuration variable."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(CONFIG_PREFIXES) and k not in CONFIG_NAMES}
    env["PYTHONIOENCODING"] = "utf-8"
    env.pop("VIRTUAL_ENV", None)
    return env


def run(cmd: List[str], cwd: Path, env: Optional[Dict[str, str]] = None, timeout: int = 900) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def http(method: str, url: str, body: Optional[dict] = None, timeout: int = 20) -> Tuple[int, str]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")


def rmtree(path: Path) -> None:
    def onerror(func, p, _exc):  # git objects are read-only on Windows
        os.chmod(p, stat.S_IWRITE)
        func(p)

    shutil.rmtree(path, onerror=onerror)


def check(ok: bool, label: str, failures: List[str], detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  ({detail})" if detail and not ok else ""))
    if not ok:
        failures.append(label)


def verify(branch: str, expect_passed: int) -> int:
    tmp = Path(tempfile.mkdtemp(prefix="paperscout-verify-"))
    failures: List[str] = []
    server: Optional[subprocess.Popen] = None
    try:
        clone = tmp / "paperscout-check"
        print(f"1. cloning branch '{branch}' into a temp dir outside the repo")
        r = run(["git", "clone", "--quiet", "--branch", branch, "--single-branch", str(REPO), str(clone)], REPO)
        check(r.returncode == 0, "git clone", failures, r.stderr.strip()[-300:])
        if failures:
            return 1
        check(not (clone / ".env").exists(), "clone contains no .env", failures)

        print("2. creating venv and installing backend/requirements-dev.txt (needs network)")
        r = run([sys.executable, "-m", "venv", str(tmp / "venv")], clone)
        check(r.returncode == 0, "create venv", failures, r.stderr[-300:])
        py = venv_python(tmp / "venv")
        r = run([str(py), "-m", "pip", "install", "--quiet", "--disable-pip-version-check",
                 "-r", "backend/requirements-dev.txt"], clone, clean_env(), timeout=1200)
        check(r.returncode == 0, "pip install -r backend/requirements-dev.txt", failures, (r.stdout + r.stderr)[-500:])
        if failures:
            return 1

        print("3. running pytest without any GEMINI_*/OPENALEX_* variable")
        r = run([str(py), "-m", "pytest", "-p", "no:cacheprovider"], clone, clean_env())
        out = r.stdout + r.stderr
        passed = re.search(r"(\d+) passed", out)
        count = int(passed.group(1)) if passed else -1
        check(r.returncode == 0, "pytest exit code 0", failures, out[-800:])
        check(count == expect_passed, f"{expect_passed} tests passed (got {count})", failures)

        print("4. starting uvicorn without .env and probing the API")
        port = free_port()
        server = subprocess.Popen(
            [str(py), "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", str(port)],
            cwd=clone, env=clean_env(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        base = f"http://127.0.0.1:{port}"
        health = None
        deadline = time.time() + 30
        while time.time() < deadline and server.poll() is None:
            try:
                status, body = http("GET", base + "/health", timeout=3)
                health = (status, body)
                break
            except (urllib.error.URLError, ConnectionError, OSError):
                time.sleep(0.5)
        check(health is not None and health[0] == 200, "/health returns 200 within 30 s", failures)
        if health and health[0] == 200:
            data = json.loads(health[1])
            check(data.get("gemini_key_configured") is False and data.get("openalex_key_configured") is False,
                  "/health reports both keys as not configured", failures, health[1])
            check(set(data) == {"status", "model", "gemini_key_configured", "openalex_key_configured"},
                  "/health exposes only the documented fields", failures, health[1])
            status, _ = http("GET", base + "/docs")
            check(status == 200, "/docs returns 200", failures)
            started = time.time()
            status, body = http("POST", base + "/api/research", {"topic": "adaptive learning", "num_papers": 3})
            elapsed = time.time() - started
            try:
                detail = json.loads(body).get("detail", "")
            except ValueError:
                detail = ""
            check(status == 503, "POST /api/research -> 503 JSON", failures, f"status {status}")
            check(all(name in detail for name in EXPECTED_MISSING),
                  "error message names every missing variable", failures, detail[:200])
            check("Traceback" not in body and elapsed < 15, "no stack trace, no hang", failures)
    finally:
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
        rmtree(tmp)
        print(f"5. temp dir removed: {not tmp.exists()}")
    if failures:
        print(f"\nFAILED ({len(failures)}): " + "; ".join(failures))
        return 1
    print("\nALL CHECKS PASSED")
    return 0


def main(argv: List[str]) -> int:
    branch, expect = "backend", 124
    for i, arg in enumerate(argv):
        if arg == "--branch":
            branch = argv[i + 1]
        if arg == "--expect-passed":
            expect = int(argv[i + 1])
    return verify(branch, expect)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
