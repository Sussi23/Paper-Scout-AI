import pytest
from pydantic import ValidationError

from backend.agent.schemas import PaperAssessment, PaperMetadata, RelevanceLabel, ResearchRequest, max_year


def test_request_defaults_and_trim():
    r = ResearchRequest(topic="  machine learning  ")
    assert r.topic == "machine learning" and r.num_papers == 10 and r.language == "vi"
    assert r.criteria is None


@pytest.mark.parametrize("topic", ["", "  ", "ab", "x" * 301])
def test_topic_invalid(topic):
    with pytest.raises(ValidationError):
        ResearchRequest(topic=topic)


def test_year_validation():
    with pytest.raises(ValidationError):
        ResearchRequest(topic="abc", year_from=2022, year_to=2020)
    with pytest.raises(ValidationError):
        ResearchRequest(topic="abc", year_from=1899)
    with pytest.raises(ValidationError):
        ResearchRequest(topic="abc", year_to=max_year() + 1)
    assert ResearchRequest(topic="abc", year_from=2020, year_to=max_year()).year_to == max_year()


def test_num_papers_and_language_and_criteria():
    for bad in (0, 31):
        with pytest.raises(ValidationError):
            ResearchRequest(topic="abc", num_papers=bad)
    with pytest.raises(ValidationError):
        ResearchRequest(topic="abc", language="fr")
    with pytest.raises(ValidationError):
        ResearchRequest(topic="abc", criteria="x" * 1001)
    assert ResearchRequest(topic="abc", criteria="   ").criteria is None


def test_label_enum_exact_values():
    assert [l.value for l in RelevanceLabel] == [
        "Relevant",
        "Possibly Relevant",
        "Not Relevant",
        "Insufficient Information",
    ]


def test_assessment_ignores_extra_fields_and_bounds_reason():
    a = PaperAssessment.model_validate(
        {"paper_id": "W1", "label": "Relevant", "reason": "Good match.", "doi": "fake", "title": "fake"}
    )
    assert not hasattr(a, "doi") and not hasattr(a, "title")
    with pytest.raises(ValidationError):
        PaperAssessment(paper_id="W1", label="Relevant", reason="no")
    with pytest.raises(ValidationError):
        PaperAssessment(paper_id="W1", label="Maybe", reason="valid reason")


def test_metadata_frozen():
    m = PaperMetadata(paper_id="W1", title="t", source_url="https://x")
    with pytest.raises(ValidationError):
        m.title = "other"
