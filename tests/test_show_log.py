from scripts.show_log import show
from tests.fakes import FakeLLM, FakeOpenAlex, fc, screen_all, turn
from backend.agent.research_agent import run_research
from backend.agent.schemas import ResearchRequest


def test_show_log_prints_timeline(settings, capsys):
    llm = FakeLLM([
        turn(fc("search_papers", query="adaptive learning", rationale="Initial broad query on the topic.")),
        screen_all("Relevant"),
        turn(fc("finish_research", reason="Done.")),
    ])
    req = ResearchRequest(topic="adaptive learning", year_from=2019, year_to=2024, num_papers=3)
    result = run_research(req, llm=llm, search_backend=FakeOpenAlex(), settings=settings)
    show(settings.log_dir / f"{result.run_id}.jsonl")
    out = capsys.readouterr().out
    assert "RUN START" in out and "TOOL  search_papers" in out and "why: Initial broad query" in out
    assert "RUN END  status=completed" in out
