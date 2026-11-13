"""Tools tests: OpenAlex client, search pipeline, processing (normalize/filter/dedup) and CSV export."""
import httpx
import pytest
from backend.tools.search import OpenAlexClient, OpenAlexError
from backend.tools.processing import SeenIndex
from backend.tools.search import OpenAlexError, find_related, search_openalex
from tests.fakes import FakeOpenAlex
from backend.tools.processing import (
    SeenIndex,
    filter_papers,
    normalize_doi,
    normalize_openalex_work,
    reconstruct_abstract,
    short_openalex_id,
)
import csv
from backend.agent.schemas import RelevanceLabel
from backend.tools.export import CSV_HEADER, export_csv
from tests.conftest import make_result, make_selected


# ----- from test_openalex_client.py ----------------------------------------
KEY = "oa-secret-key-999"


def make_client(handler, **kw):
    sleeps = []
    client = OpenAlexClient(KEY, transport=httpx.MockTransport(handler), sleep=sleeps.append, **kw)
    return client, sleeps


def test_search_request_shape_and_auth_header():
    seen = {}

    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"results": [{"id": "https://openalex.org/W1", "title": "t"}]})

    client, _ = make_client(handler)
    out = client.search_works("adaptive learning", year_from=2020, year_to=2024, limit=25)
    assert out[0]["id"].endswith("W1")
    assert seen["auth"] == f"Bearer {KEY}"
    assert KEY not in seen["url"]  # key only in header, never in URL
    assert "search=adaptive+learning" in seen["url"] or "search=adaptive%20learning" in seen["url"]
    assert "from_publication_date%3A2020-01-01" in seen["url"] or "from_publication_date:2020-01-01" in seen["url"]
    assert "per_page=25" in seen["url"] and "select=" in seen["url"]


def test_retry_on_429_then_success():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429, headers={"Retry-After": "1"}, json={"message": "slow down"})
        return httpx.Response(200, json={"results": []})

    client, sleeps = make_client(handler)
    assert client.search_works("x", year_from=None, year_to=None, limit=5) == []
    assert calls["n"] == 3 and sleeps == [1.0, 1.0]


def test_daily_budget_429_not_retried():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(429, headers={"Retry-After": "40000"}, json={"message": "budget"})

    client, _ = make_client(handler)
    with pytest.raises(OpenAlexError) as exc:
        client.search_works("x", year_from=None, year_to=None, limit=5)
    assert exc.value.code == "OPENALEX_RATE_LIMIT" and calls["n"] == 1


@pytest.mark.parametrize("status", [401, 403])
def test_auth_error_not_retried_and_key_not_leaked(status):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(status, json={"message": f"bad key {KEY}"})

    client, _ = make_client(handler)
    with pytest.raises(OpenAlexError) as exc:
        client.search_works("x", year_from=None, year_to=None, limit=5)
    assert exc.value.code == "OPENALEX_AUTH" and calls["n"] == 1
    assert KEY not in exc.value.message


def test_server_errors_retry_three_times_then_fail():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(503, text="down")

    client, sleeps = make_client(handler)
    with pytest.raises(OpenAlexError) as exc:
        client.search_works("x", year_from=None, year_to=None, limit=5)
    assert exc.value.code == "OPENALEX_HTTP" and exc.value.retryable
    assert calls["n"] == 4 and len(sleeps) == 3


def test_client_error_400_not_retried():
    client, sleeps = make_client(lambda r: httpx.Response(400, json={"error": "Invalid filter", "message": "bad"}))
    with pytest.raises(OpenAlexError) as exc:
        client.search_works("x", year_from=None, year_to=None, limit=5)
    assert exc.value.code == "OPENALEX_HTTP" and not exc.value.retryable and sleeps == []


def test_timeout_maps_to_code():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    client, sleeps = make_client(handler)
    with pytest.raises(OpenAlexError) as exc:
        client.search_works("x", year_from=None, year_to=None, limit=5)
    assert exc.value.code == "OPENALEX_TIMEOUT" and len(sleeps) == 3


def test_related_works_two_step_fetch():
    urls = []

    def handler(request: httpx.Request):
        urls.append(request.url.path)
        if request.url.path == "/works/W1":
            return httpx.Response(
                200, json={"id": "https://openalex.org/W1",
                           "related_works": ["https://openalex.org/W10", "https://openalex.org/W11"]}
            )
        assert "W10|W11" in request.url.params["filter"]
        return httpx.Response(200, json={"results": [{"id": "https://openalex.org/W10"}, {"id": "https://openalex.org/W11"}]})

    client, _ = make_client(handler)
    assert len(client.related_works("W1", limit=10)) == 2
    assert urls == ["/works/W1", "/works"]


def test_related_works_empty_seed_skips_second_request():
    urls = []

    def handler(request):
        urls.append(request.url.path)
        return httpx.Response(200, json={"id": "https://openalex.org/W1", "related_works": []})

    client, _ = make_client(handler)
    assert client.related_works("W1", limit=10) == [] and urls == ["/works/W1"]


# ----- from test_search_tools.py ----------------------------------------
def test_search_pipeline_stats():
    backend, seen = FakeOpenAlex(), SeenIndex()
    out = search_openalex(backend, "adaptive learning", year_from=2019, year_to=2024, limit=25, seen=seen)
    assert [p.paper_id for p in out.new_papers] == [f"W90000000{n:02d}" for n in (1, 2, 3, 4, 5)]
    s = out.stats
    assert (s.returned, s.dropped_invalid, s.dropped_year, s.dropped_duplicate) == (8, 1, 2, 0)


def test_second_search_only_returns_new_papers():
    backend, seen = FakeOpenAlex(), SeenIndex()
    search_openalex(backend, "adaptive learning", year_from=2019, year_to=2024, limit=25, seen=seen)
    out = search_openalex(backend, "learning analytics", year_from=2019, year_to=2024, limit=25, seen=seen)
    assert [p.paper_id for p in out.new_papers] == ["W9000000010", "W9000000011", "W9000000012"]
    assert out.stats.dropped_duplicate == 3  # by DOI (W09), by id (W02), by title+year (W14)


def test_find_related_uses_same_pipeline():
    backend, seen = FakeOpenAlex(), SeenIndex()
    out = find_related(backend, "W9000000001", year_from=2021, year_to=2024, limit=10, seen=seen)
    assert [p.paper_id for p in out.new_papers] == ["W9000000010", "W9000000011"]  # W12 is 2020 -> dropped
    assert out.stats.dropped_year == 1
    assert backend.calls[-1] == {"op": "related", "seed_id": "W9000000001", "limit": 10}


def test_unknown_query_returns_empty():
    out = search_openalex(FakeOpenAlex(), "no such topic", year_from=None, year_to=None, limit=5, seen=SeenIndex())
    assert out.new_papers == [] and out.stats.returned == 0


def test_backend_errors_propagate_as_openalex_error():
    backend = FakeOpenAlex(fail_with=OpenAlexError("OPENALEX_TIMEOUT", "t", retryable=True))
    with pytest.raises(OpenAlexError) as exc:
        search_openalex(backend, "adaptive learning", year_from=None, year_to=None, limit=5, seen=SeenIndex())
    assert exc.value.code == "OPENALEX_TIMEOUT"


# ----- from test_processing.py ----------------------------------------
def _normalize_all(raw_works):
    return [p for p in (normalize_openalex_work(w) for w in raw_works) if p is not None]


def test_doi_normalisation():
    assert normalize_doi("10.1/ABC") == "https://doi.org/10.1/ABC"
    assert normalize_doi("doi:10.1/abc") == "https://doi.org/10.1/abc"
    assert normalize_doi("http://dx.doi.org/10.1/abc") == "https://doi.org/10.1/abc"
    assert normalize_doi("not a doi") is None and normalize_doi(None) is None


def test_short_id():
    assert short_openalex_id("https://openalex.org/W123") == "W123"
    assert short_openalex_id("w456") == "W456"
    assert short_openalex_id("https://openalex.org/A123") is None


def test_abstract_reconstruction():
    assert reconstruct_abstract({"hello": [0], "world": [1, 3], "big": [2]}) == "hello world big world"
    assert reconstruct_abstract(None) is None and reconstruct_abstract({}) is None


def test_normalize_work_fields(raw_by_id):
    p = normalize_openalex_work(raw_by_id["W9000000001"])
    assert p.paper_id == "W9000000001" and p.year == 2021 and p.source == "openalex"
    assert p.doi == "https://doi.org/10.5555/syn.0001" and p.source_url == p.doi
    assert p.authors == ["Alice Example", "Bob Sample"]
    assert p.abstract.startswith("SYNTHETIC TEST DATA.")


def test_source_url_fallbacks(raw_by_id):
    no_doi = normalize_openalex_work(raw_by_id["W9000000004"])
    assert no_doi.doi is None and no_doi.source_url == "https://example.org/syn/0004"
    bare = dict(raw_by_id["W9000000004"], primary_location=None)
    assert normalize_openalex_work(bare).source_url == "https://openalex.org/W9000000004"


def test_missing_abstract_not_fabricated(raw_by_id):
    assert normalize_openalex_work(raw_by_id["W9000000004"]).abstract is None


def test_missing_title_or_id_returns_none(raw_by_id):
    assert normalize_openalex_work(raw_by_id["W9000000007"]) is None
    assert normalize_openalex_work({"title": "x"}) is None
    assert normalize_openalex_work("junk") is None


def test_dedup_by_doi_and_year_filter(raw_works):
    papers = _normalize_all(raw_works)
    first_batch = [p for p in papers if p.paper_id in {"W9000000001", "W9000000003", "W9000000006", "W9000000008"}]
    seen = SeenIndex()
    kept, stats = filter_papers(first_batch, 2019, 2024, seen)
    assert [p.paper_id for p in kept] == ["W9000000001", "W9000000003"]
    assert stats.dropped_year == 2  # 2018 and unknown year
    # second batch: DOI duplicate (different id, different DOI case) and title+year duplicate
    dup_doi = next(p for p in papers if p.paper_id == "W9000000009")
    dup_title = next(p for p in papers if p.paper_id == "W9000000014")
    same_id = next(p for p in papers if p.paper_id == "W9000000001")
    kept2, stats2 = filter_papers([dup_doi, dup_title, same_id], 2019, 2024, seen)
    assert kept2 == [] and stats2.dropped_duplicate == 3


def test_no_year_bounds_keeps_unknown_year(raw_works):
    papers = _normalize_all(raw_works)
    kept, stats = filter_papers(papers, None, None, SeenIndex())
    assert any(p.year is None for p in kept) and stats.dropped_year == 0


def test_filter_preserves_metadata_and_counts_invalid(raw_by_id):
    p = normalize_openalex_work(raw_by_id["W9000000005"])
    blank = p.model_copy(update={"title": "  "})
    kept, stats = filter_papers([p, blank], 2020, 2020, SeenIndex())
    assert kept == [p] and kept[0].doi == "https://doi.org/10.5555/SYN.0005"
    assert stats.dropped_invalid == 1 and stats.returned == 2


# ----- from test_export.py ----------------------------------------
def test_export_csv(tmp_path):
    sel = [
        make_selected(
            "W1",
            "Nghiên cứu về học tập thích ứng",
            authors=["Nguyễn Văn An", "Trần Thị Bình"],
            doi="https://doi.org/10.5555/x",
            abstract="Tóm tắt, có dấu phẩy và \"ngoặc kép\".\nXuống dòng.",
            reason="Phù hợp chủ đề.",
        ),
        make_selected("W2", "No abstract paper", abstract=None, doi=None, label=RelevanceLabel.POSSIBLY_RELEVANT),
    ]
    path = export_csv(make_result(sel), tmp_path / "out" / "r.csv")
    raw = path.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")  # UTF-8 BOM for Excel
    with path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == CSV_HEADER == [
        "Title", "Authors", "Year", "DOI", "Source URL", "Abstract", "Relevance", "Relevance Reason"
    ]
    assert rows[1][0] == "Nghiên cứu về học tập thích ứng"
    assert rows[1][1] == "Nguyễn Văn An; Trần Thị Bình"
    assert rows[1][5] == "Tóm tắt, có dấu phẩy và \"ngoặc kép\".\nXuống dòng."
    assert rows[1][6] == "Relevant" and rows[1][7] == "Phù hợp chủ đề."
    # missing abstract / DOI stay empty, never fabricated
    assert rows[2][3] == "" and rows[2][5] == "" and rows[2][6] == "Possibly Relevant"


def test_export_empty_result_has_header_only(tmp_path):
    path = export_csv(make_result([]), tmp_path / "empty.csv")
    with path.open(encoding="utf-8-sig", newline="") as fh:
        assert list(csv.reader(fh)) == [CSV_HEADER]


# ----- per-search identity used by the funnel ---------------------------------------------------
def test_search_stats_identity_for_every_search():
    backend, seen = FakeOpenAlex(), SeenIndex()
    for query in ("adaptive learning", "learning analytics", "virtual reality education"):
        out = search_openalex(backend, query, year_from=2019, year_to=2024, limit=25, seen=seen)
        s = out.stats
        assert s.returned == s.dropped_invalid + s.dropped_year + s.dropped_duplicate + len(out.new_papers)


def test_run_state_register_keeps_first_origin_and_accumulates_totals(settings):
    from backend.agent.schemas import ResearchRequest
    from backend.agent.state import RunState

    state = RunState(request=ResearchRequest(topic="adaptive learning"), settings=settings, run_id="abcdef012345")
    first = search_openalex(FakeOpenAlex(), "adaptive learning", year_from=None, year_to=None, limit=25, seen=SeenIndex())
    state.register(first.new_papers, step=1, tool="search_papers", via="adaptive learning")
    state.add_search_stats(first.stats)
    state.register(first.new_papers[:1], step=4, tool="find_related_papers", via="related:X")  # already known
    assert state.origins[first.new_papers[0].paper_id].step == 1
    assert state.search_totals.returned == 8
    state.register(first.new_papers)  # old call style (no origin arguments) still works
    assert len(state.registry) == len(first.new_papers)
