"""The leak scanner must detect leaks (positive controls use fake values only).

Fake values are assembled from pieces so that this file does not match the scanner's own patterns.
"""
from scripts.scan_secrets import scan_content

FAKE = {"DEMO_API_KEY": "fake-value-0123456789"}
BS = "\\"


def test_detects_secret_value_without_echoing_it():
    out = scan_content("a.py", b'x = "fake-value-0123456789"\n', FAKE)
    assert out == ["a.py:1: contains the VALUE of variable DEMO_API_KEY"]
    assert "fake-value" not in " ".join(out)


def test_detects_key_shapes_tokens_and_personal_paths():
    google_shape = "AI" + "za" + "A" * 35
    url = "https://x.test/w" + "?api_" + "key=" + "abcdef1234567890"
    bearer = "Authorization: " + "Bea" + "rer " + "abcdefghijklmnopqrstuvwxyz"
    win_path = "C:" + BS + "Us" + "ers" + BS + "someone" + BS + "proj"
    unix_path = "/" + "ho" + "me" + "/" + "someone/proj"
    text = "\n".join([f"k = '{google_shape}'", f"url = '{url}'", f"h = '{bearer}'", f"p = '{win_path}'", f"q = '{unix_path}'"])
    kinds = {line.split("pattern ")[1] for line in scan_content("b.txt", text.encode(), {})}
    assert kinds == {
        "google-api-key-shape", "api_key-in-url", "bearer-token", "personal-path-windows", "personal-path-unix",
    }


def test_placeholders_and_redacted_paths_are_not_flagged():
    text = "\n".join(["api_key=YOUR_KEY", "Bearer {token}", "<local>" + BS + "proj", "Authorization: Bearer $TOKEN"])
    assert scan_content("c.md", text.encode(), {}) == []
