import pytest

from backend.config import ConfigError, load_settings

FULL = {"GEMINI_API_KEY": "gk-secret-123", "GEMINI_MODEL": "some-model", "OPENALEX_API_KEY": "ok-secret-456"}


def test_defaults_and_required():
    s = load_settings(env_file=None, environ=FULL)
    assert s.max_searches == 3 and s.max_model_steps == 12 and s.search_page_size == 25
    assert s.abstract_chars_for_model == 1200 and s.max_concurrent_runs == 1
    assert s.gemini_temperature is None and s.gemini_min_interval_seconds == 2.0
    assert s.gemini_model == "some-model"


def test_missing_model_has_helpful_message():
    env = {k: v for k, v in FULL.items() if k != "GEMINI_MODEL"}
    with pytest.raises(ConfigError) as exc:
        load_settings(env_file=None, environ=env)
    assert "list_models.py" in str(exc.value)


def test_missing_keys_lenient_mode():
    s = load_settings(env_file=None, environ={}, strict=False)
    assert set(s.missing_required()) == {"GEMINI_API_KEY", "GEMINI_MODEL", "OPENALEX_API_KEY"}


def test_invalid_numbers_rejected():
    with pytest.raises(ConfigError):
        load_settings(env_file=None, environ={**FULL, "MAX_SEARCHES": "abc"})
    with pytest.raises(ConfigError):
        load_settings(env_file=None, environ={**FULL, "GEMINI_TEMPERATURE": "9"})


def test_origins_parsed_and_secrets_hidden():
    s = load_settings(env_file=None, environ={**FULL, "ALLOWED_ORIGINS": "http://a.com, http://b.com ,"})
    assert s.allowed_origins == ("http://a.com", "http://b.com")
    assert "gk-secret-123" not in repr(s) and "ok-secret-456" not in repr(s)
    assert set(s.secret_values()) == {"gk-secret-123", "ok-secret-456"}
