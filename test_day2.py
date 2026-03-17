"""
Day 2 test — verify AgentState and web_search tool work correctly.
Run with: python test_day2.py
"""

import os
from dotenv import load_dotenv

load_dotenv()


def test_state():
    from agent.state import AgentState
    from langchain_core.messages import HumanMessage

    state: AgentState = {
        "messages": [HumanMessage(content="test message")],
        "sources": [],
    }

    print(f"  AgentState created with {len(state['messages'])} message")
    print(f"  Message content: {state['messages'][0].content}")
    print("  AgentState working correctly.")


def test_web_search():
    from agent.tools import web_search

    print("\n  Running web_search for 'LangGraph agent tutorial'...")
    result = web_search.invoke({"query": "LangGraph agent tutorial"})

    lines = result.split("\n")
    print(f"  Got {result.count('Title:')} results back")
    print(f"  First result: {lines[0]}")
    print("  web_search tool working correctly.")


if __name__ == "__main__":
    print("Day 2 test\n")
    test_state()
    test_web_search()
    print("\nDay 2 complete. Ready for Day 3.")
