import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition


from agent.state import AgentState
from agent.tools import TOOLS

load_dotenv()

SYSTEM_PROMPT = """You are a research assistant. When given a question:

STEP 1 - Search first
Use web_search to find 3-5 relevant sources on the topic. 

STEP 2 - Read the best sources
Use read-page to read 2-3 of the most promising URLs returned from the web search 
So not skip this step for complex questions - snippets alone are not enough

STEP 3 - Synthesise a structured answer 
Write your final answer in this format:

## Summary
2-3 sentence overview of the key finding. 

## Detail
The full explanation with the specifics, numbers and examples where relevant. 

## Sources
- [Title](URL)
- [Title](URL)

IMPORTANT RULES 
- Never make up facts. If you cannot find something, say no
- Always read one full page before answering complex questions 
- Keep your answer focused on what was actually asked."""


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
