import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition

from agent.state import AgentState
from agent.tools import TOOLS

load_dotenv()

MAX_ITERATIONS = 10
MAX_READ_PAGE_CALLS = 3

SYSTEM_PROMPT = """You are a thorough research assistant. When given a question:

STEP 1 — Search first
Use web_search to find 3-5 relevant sources on the topic.

STEP 2 — Read the best sources
Use read_page on the 2-3 most promising URLs from your search results.
Do not read more than 3 pages — be selective about which ones are worth reading.

STEP 3 — Synthesise a structured answer
Write your final answer in this format:

## Summary
2-3 sentence overview of the key finding.

## Detail
The full explanation with specifics, numbers, and examples where relevant.

## Sources
- [Title](URL)
- [Title](URL)

IMPORTANT RULES:
- Never make up facts. If you cannot find something, say so.
- Do not call read_page more than 3 times — choose the best sources.
- Keep your answer focused on what was actually asked."""


def build_graph():
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    ).bind_tools(TOOLS)

    def agent_node(state: AgentState) -> dict:
        iterations = state.get("iterations", 0)

        if iterations >= MAX_ITERATIONS:
            return {
                "messages": [
                    AIMessage(
                        content="I've reached the maximum number of research steps. "
                        "Here is what I found so far based on my research."
                    )
                ],
                "iterations": iterations + 1,
            }

        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response = llm.invoke(messages)

        if hasattr(response, "tool_calls") and response.tool_calls:
            read_page_count = state.get("read_page_count", 0)
            filtered_calls = []
            for tc in response.tool_calls:
                if tc["name"] == "read_page":
                    if read_page_count < MAX_READ_PAGE_CALLS:
                        filtered_calls.append(tc)
                        read_page_count += 1
                else:
                    filtered_calls.append(tc)

            if len(filtered_calls) != len(response.tool_calls):
                response.tool_calls = filtered_calls

            return {
                "messages": [response],
                "iterations": iterations + 1,
                "read_page_count": read_page_count,
            }

        return {
            "messages": [response],
            "iterations": iterations + 1,
        }

    tool_node = ToolNode(TOOLS)

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")

    return graph.compile()


research_graph = build_graph()
