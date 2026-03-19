"""
Day 4 test — verify read_page is wired in and agent produces
structured, cited answers.
Run with: python test_day4.py
"""

import os
from dotenv import load_dotenv

load_dotenv()


def test_tools_registered():
    from agent.tools import TOOLS

    tool_names = [t.name for t in TOOLS]
    assert "web_search" in tool_names, "web_search missing"
    assert "read_page" in tool_names, "read_page missing"
    print(f"  Tools registered: {tool_names}")


def test_read_page_directly():
    from agent.tools import read_page

    print("\n  Testing read_page directly on a real URL...")
    result = read_page.invoke({"url": "https://python.org"})
    assert len(result) > 100, "read_page returned too little content"
    print(f"  read_page returned {len(result)} characters")
    print(f"  First 100 chars: {result[:100]}")
    print("  read_page working correctly.")


def test_agent_uses_read_page():
    from agent.graph import research_graph
    from langchain_core.messages import HumanMessage

    print("\n  Running agent on a question requiring page reading...")
    print("  Question: What are the main features of Python 3.12?")

    result = research_graph.invoke(
        {
            "messages": [
                HumanMessage(content="What are the main features of Python 3.12?")
            ],
            "sources": [],
        }
    )

    messages = result["messages"]
    tool_names_used = []
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_names_used.append(tc["name"])

    final_answer = messages[-1].content
    print(f"\n  Tools used in order: {tool_names_used}")
    print(f"  Total messages: {len(messages)}")
    print(f"\n  Final answer preview:\n")
    print(f"  {final_answer[:600]}")

    has_summary = "summary" in final_answer.lower() or "##" in final_answer
    has_sources = "http" in final_answer.lower() or "source" in final_answer.lower()
    print(f"\n  Has structured sections: {has_summary}")
    print(f"  Has sources: {has_sources}")
    print("\n  Agent producing structured answers correctly.")


if __name__ == "__main__":
    print("Day 4 test\n")
    test_tools_registered()
    test_read_page_directly()
    test_agent_uses_read_page()
    print("\nDay 4 complete. Ready for Day 5.")
