"""
Day 5 test — verify LangSmith tracing is active and working.
Run with: python test_day5.py
"""

import os
from dotenv import load_dotenv

load_dotenv()


def test_langsmith_config():
    tracing = os.getenv("LANGCHAIN_TRACING_V2")
    api_key = os.getenv("LANGCHAIN_API_KEY")
    project = os.getenv("LANGCHAIN_PROJECT")

    assert tracing == "true", f"LANGCHAIN_TRACING_V2 is '{tracing}', should be 'true'"
    assert api_key and api_key != "your-langsmith-key-here", "LANGCHAIN_API_KEY not set"
    assert project, "LANGCHAIN_PROJECT not set"

    print(f"  Tracing enabled: {tracing}")
    print(f"  Project: {project}")
    print(f"  API key: found ({api_key[:12]}...)")
    print("  LangSmith config correct.")


def test_agent_with_tracing():
    from agent.graph import research_graph
    from langchain_core.messages import HumanMessage

    print("\n  Running agent — this run will appear in LangSmith...")
    print("  Question: What is the capital of the UAE and what is it known for?")

    result = research_graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content="What is the capital of the UAE and what is it known for?"
                )
            ],
            "sources": [],
        }
    )

    messages = result["messages"]
    tool_calls_made = []
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls_made.append(tc["name"])

    final_answer = messages[-1].content

    print(f"  Tools called: {tool_calls_made}")
    print(f"  Total messages: {len(messages)}")
    print(f"\n  Answer preview: {final_answer[:300]}")
    print("\n  Run complete — check smith.langchain.com for the trace.")


if __name__ == "__main__":
    print("Day 5 test\n")
    test_langsmith_config()
    test_agent_with_tracing()
    print("\nDay 5 complete. Ready for Day 6.")
