"""
Day 3 test — the full agent loop running end to end.
Run with: python test_day3.py
"""

import os
from dotenv import load_dotenv

load_dotenv()


def test_graph_builds():
    from agent.graph import research_graph

    print("  Graph compiled successfully.")
    print(f"  Nodes: {list(research_graph.nodes.keys())}")


def test_agent_runs():
    from agent.graph import research_graph
    from langchain_core.messages import HumanMessage

    print("\n  Running agent on a real question...")
    print("  Question: What is LangGraph and what is it used for?")
    print("  (This will take 10-20 seconds — the agent is searching)\n")

    result = research_graph.invoke(
        {
            "messages": [
                HumanMessage(content="What is LangGraph and what is it used for?")
            ],
            "sources": [],
        }
    )

    final_message = result["messages"][-1].content
    total_messages = len(result["messages"])

    print(f"  Agent completed in {total_messages} messages (turns)")
    print(f"\n  Final answer preview:\n")
    print(f"  {final_message[:500]}...")
    print("\n  Agent loop working correctly.")


if __name__ == "__main__":
    print("Day 3 test\n")
    test_graph_builds()
    test_agent_runs()
    print("\nDay 3 complete. Ready for Day 4.")
