"""Layer 0 environment sanity checks."""

import locale
import sys

import pytest


@pytest.mark.smoke
def test_utf8_encoding():
    """Preferred encoding UTF-8 olmali — ChromaDB .env bug'i tekrarlamasin."""
    enc = locale.getpreferredencoding(False)
    assert enc.lower().replace("-", "") == "utf8", (
        f"Preferred encoding {enc}, UTF-8 bekleniyor. "
        "PYTHONUTF8=1 environment variable ayarla."
    )


@pytest.mark.smoke
def test_venv_active():
    """Global pip'e dokunmayi engelle — venv aktif olmali."""
    assert ".venv" in sys.executable, (
        f"Python: {sys.executable} — .venv aktif degil!"
    )


@pytest.mark.smoke
def test_python_version():
    """Python 3.11+ gerekli."""
    assert sys.version_info >= (3, 11)
