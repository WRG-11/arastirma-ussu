#!/usr/bin/env python
"""dolphin-mistral tool-calling sanity test.

Sonuc:
  PASS (exit 0) -> bind_tools calisiyor, Layer 1 native tool calling kullanir
  FAIL (exit 1) -> bind_tools calismiyor, Layer 1 manuel ReAct parser kullanir

Bu karar Layer 1 mimarisini BELIRLER (fallback degil, default secim).
"""

import sys

from langchain_core.tools import tool
from langchain_ollama import ChatOllama


@tool
def test_add(a: int, b: int) -> int:
    """Iki sayiyi topla."""
    return a + b


def main():
    print("=== dolphin-mistral tool-calling sanity test ===")
    print()

    llm = ChatOllama(model="dolphin-mistral", base_url="http://localhost:11434")
    llm_with_tools = llm.bind_tools([test_add])

    try:
        resp = llm_with_tools.invoke("3 ile 5'i topla")
        calls = resp.tool_calls if hasattr(resp, "tool_calls") else []

        if calls and calls[0].get("name") == "test_add":
            print("PASS: bind_tools calisiyor")
            print("KARAR: Layer 1 native tool calling kullanilacak")
            sys.exit(0)
        else:
            content_preview = resp.content[:200] if resp.content else "(bos)"
            print(f"FAIL: tool_calls bos veya yanlis.")
            print(f"  Response: {content_preview}")
            print(f"  tool_calls: {calls}")
            print()
            print("KARAR: Layer 1 manuel ReAct parser ile kurulacak")
            sys.exit(1)

    except Exception as e:
        print(f"FAIL: Exception — {e}")
        print()
        print("KARAR: Layer 1 manuel ReAct parser ile kurulacak")
        sys.exit(1)


if __name__ == "__main__":
    main()
