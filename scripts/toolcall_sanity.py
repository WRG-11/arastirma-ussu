#!/usr/bin/env python
"""dolphin-mistral tool-calling sanity test.

Outcome:
  PASS (exit 0) -> bind_tools works, Layer 1 uses native tool calling
  FAIL (exit 1) -> bind_tools broken, Layer 1 uses the manual ReAct parser

This decision DETERMINES the Layer 1 architecture (not a fallback - the default choice).
"""

import sys

from langchain_core.tools import tool
from langchain_ollama import ChatOllama


@tool
def test_add(a: int, b: int) -> int:
    """Add two numbers."""
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
            print("PASS: bind_tools works")
            print("DECISION: Layer 1 will use native tool calling")
            sys.exit(0)
        else:
            content_preview = resp.content[:200] if resp.content else "(empty)"
            print(f"FAIL: tool_calls empty or wrong.")
            print(f"  Response: {content_preview}")
            print(f"  tool_calls: {calls}")
            print()
            print("DECISION: Layer 1 will be wired with the manual ReAct parser")
            sys.exit(1)

    except Exception as e:
        print(f"FAIL: Exception - {e}")
        print()
        print("DECISION: Layer 1 will be wired with the manual ReAct parser")
        sys.exit(1)


if __name__ == "__main__":
    main()
