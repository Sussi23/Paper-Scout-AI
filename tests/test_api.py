import csv
import re
import threading
import uuid

import pytest
from fastapi.testclient import TestClient

from backend.config import Settings
from backend.main import create_app, get_research_fn
from tests.conftest import FAKE_GEMINI_KEY, FAKE_OPENALEX_KEY, make_result, make_selected

VALID = {"topic": "adaptive learning", "year_from": 2020, "year_to": 2024, "num_papers": 3}


def fake_run(request, settings=None, **kw):
    selected = [make_selected("W1", "Nghiên cứu học tập thích ứng", authors=["Nguyễn Văn An"], abstract=None)]
    return make_result(selected, run_id=uuid.uuid4().hex[:12], request=request)


@pytest.fixture
def api(settings):
    app = create_app(settings.model_copy(update={"allowed_origins": ("http://localhost:3000",)}))
    app.dependency_overrides[get_research_fn] = lambda: fake_run
    return app, TestClient(app)


def test_health_reports_config_without_leaking_keys(api):
    _, client = api
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ok", "model": "fake-model", "gemini_key_configured": True, "openalex_key_configured": True,
    }
    assert FAKE_GEMINI_KEY not in resp.text and FAKE_OPENALEX_KEY not in resp.text


def test_health_when_keys_missing(tmp_path):
    client = TestClient(create_app(Settings(runs_dir=tmp_path)))
    body = client.get("/health").json()
    assert body["gemini_key_configured"] is False and body["openalex_key_configured"] is False


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"topic": ""},
        {"topic": "   "},
        {"topic": "ok topic", "year_from": 2024, "year_to": 2020},
        {"topic": "ok topic", "num_papers": 99},
        {"topic": "ok topic", "language": "fr"},
    ],
)
def test_invalid_input_is_422_with_clear_message(api, payload):
    _, client = api
    resp = client.post("/api/research", json=payload)
    assert resp.status_code == 422
    body = resp.json()
    assert body["detail"].startswith("Invalid request") and body["errors"]


def test_year_order_message_mentions_years(api):
    _, client = api
    resp = client.post("/api/research", json={"topic": "ok topic", "year_from": 2024, "year_to": 2020})
    assert "year_from must be <= year_to" in resp.text


def test_post_valid_request_persists_and_returns_result(api, settings):
    _, client = api
    resp = client.post("/api/research", json=VALID)
    assert resp.status_code == 200
    body = resp.json()
    assert re.fullmatch(r"[a-f0-9]{12}", body["run_id"]) and body["status"] == "completed"
    assert body["selected"][0]["metadata"]["title"] == "Nghiên cứu học tập thích ứng"
    run_dir = settings.runs_dir / body["run_id"]
    assert (run_dir / "result.json").is_file() and (run_dir / "results.csv").is_file()


def test_get_run_and_export(api):
    _, client = api
    run_id = client.post("/api/research", json=VALID).json()["run_id"]
    run = client.get(f"/api/runs/{run_id}")
    assert run.status_code == 200 and run.json()["run_id"] == run_id and "tool_history" in run.json()

    export = client.get(f"/api/exports/{run_id}")
    assert export.status_code == 200
    assert export.headers["content-type"].startswith("text/csv")
    assert f'filename="paperscout_{run_id}.csv"' in export.headers["content-disposition"]
    assert export.content.startswith(b"\xef\xbb\xbf")
    rows = list(csv.reader(export.content.decode("utf-8-sig").splitlines()))
    assert rows[0][0] == "Title" and rows[1][0] == "Nghiên cứu học tập thích ứng" and rows[1][5] == ""


def test_unknown_run_is_404(api):
    _, client = api
    assert client.get("/api/runs/aaaaaaaaaaaa").status_code == 404
    assert client.get("/api/exports/aaaaaaaaaaaa").status_code == 404


@pytest.mark.parametrize("bad", ["..%2Fx", "%2e%2e%2fsecret", "abc", "ABCDEF012345", "aaaaaaaaaaaaa", "..%5Cx"])
def test_malformed_or_traversal_run_ids_rejected(api, bad):
    _, client = api
    for route in ("/api/runs/", "/api/exports/"):
        resp = client.get(route + bad)
        assert resp.status_code in (404, 422) and "secret" not in resp.text.lower().replace("%2fsecret", "")


def test_unexpected_error_is_500_json_without_details(api):
    app, _ = api

    def boom(request, settings=None, **kw):
        raise RuntimeError(f"internal detail {FAKE_GEMINI_KEY}")

    app.dependency_overrides[get_research_fn] = lambda: boom
    resp = TestClient(app).post("/api/research", json=VALID)
    assert resp.status_code == 500 and resp.json()["detail"]
    assert "internal detail" not in resp.text and FAKE_GEMINI_KEY not in resp.text


def test_incomplete_server_config_is_503(tmp_path):
    client = TestClient(create_app(Settings(runs_dir=tmp_path)))  # real run_research, no keys
    resp = client.post("/api/research", json=VALID)
    assert resp.status_code == 503 and "GEMINI" in resp.text


def test_concurrency_limit_returns_429(api):
    app, client = api
    app.state.queue_timeout = 0.05
    started, release = threading.Event(), threading.Event()

    def slow(request, settings=None, **kw):
        started.set()
        release.wait(5)
        return fake_run(request)

    app.dependency_overrides[get_research_fn] = lambda: slow
    first = {}
    worker = threading.Thread(target=lambda: first.update(r=TestClient(app).post("/api/research", json=VALID)))
    worker.start()
    assert started.wait(5)
    try:
        second = client.post("/api/research", json=VALID)
        assert second.status_code == 429
    finally:
        release.set()
        worker.join(5)
    assert first["r"].status_code == 200


def test_cors_allows_configured_origin_only(api):
    _, client = api
    ok = client.get("/health", headers={"Origin": "http://localhost:3000"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:3000"
    other = client.get("/health", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in other.headers
