"""
Day 8 test - verify guardrails are working.
Run with: python test_day8.py
"""

import os
from dotenv import load_dotenv

load_dotenv()


def test_max_read_page():
    from agent.graph import research_graph
    from langchain_core.messages import HumanMessage

    print(" Testing read_page cap...")
    print(" Question: What is the GIL in Python?")
    print(" (Previously used 5 read_page calls - should now be capped at 3)\n")

    result = research_graph.invoke(
        {
            "messages": [HumanMessage(content="What is the GIL in Python?")],
            "sources": [],
            "iterations": 0,
            "read_page_count": 0,
        }
    )

    messages = result["messages"]
    read_page_calls = sum(
        1
        for msg in messages
        if hasattr(msg, "tool_calls") and msg.tool_calls
        for tc in msg.tool_calls
        if tc["name"] == "read_page"
    )

    print(f" read_page calls used: {read_page_calls} (max allowed: 3)")
    print(f" Total messages: {len(messages)}")
    assert (
        read_page_calls <= 3
    ), f"Guardrail failed - used {read_page_calls} read_page_calls"
    print(" read_page cap working correctly.")


def test_tool_error_handling():
    from agent.tools import read_page

    print("\n Testing greaceful failure on bad URL...")
    result = read_page.invoke({"url": "https://this-url-does-not-exist-xyz123.com"})
    assert (
        "Could not read" in result or "error" in result.lower()
    ), "Tool should return error message, not raise exception"
    print(f" Graceful error returned: {result[:80]}...")
    print(" Error handling working correctly.")


if __name__ == "__main__":
    print("Day 8 test\n")
    test_tool_error_handling()
    test_max_read_page()
    print("\nDay 8 complete. Ready for Day 9.")
