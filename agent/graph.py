import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition


from agent.state import AgentState
from agent.tools import TOOLS

load_dotenv()

SYSTEM_PROMPT = """You are a research assistant. Your job is to answer
questions thoroughly and accurately by searching the web.

When given a research question:
1. Use web_search to find relevant sources
2. Use read_page to read the most promising pages in detail
3. Synthesise everything into a clear, well-structured answer
4. Always mention the sources you used at the end

Be concise but complete. If search results are insufficient, say so
clearly rather than guessing."""


def build_graph():
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    ).bind_tools(TOOLS)

    def agent_node(state: AgentState) -> dict:
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response = llm.invoke(messages)
        return {"messages": [response]}

    tool_node = ToolNode(TOOLS)

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")

    return graph.compile()


research_graph = build_graph()
