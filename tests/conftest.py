"""Shared test fixtures."""

import urllib.request

import pytest


def ollama_available() -> bool:
    """Check if Ollama server is reachable."""
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


@pytest.fixture
def skip_no_ollama():
    """Skip test if Ollama is not running."""
    if not ollama_available():
        pytest.skip("Ollama not running")
