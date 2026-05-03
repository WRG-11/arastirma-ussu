"""Timeout wrapper for crew execution."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError  # noqa: F401
from typing import Callable, TypeVar

T = TypeVar("T")


def run_with_timeout(func: Callable[..., T], *args, timeout: int = 600) -> T:
    """Run *func(*args)* with a timeout.

    Raises ``TimeoutError`` if the function does not complete within
    *timeout* seconds.  Note: the underlying thread is **not** killed
    (CPython limitation) — the Ollama-side ``request_timeout`` provides
    the real safety net.
    """
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args)
        return future.result(timeout=timeout)
