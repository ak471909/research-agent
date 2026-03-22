import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()

if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]
    os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
    os.environ["LANGCHAIN_TRACING_V2"] = st.secrets["LANGCHAIN_TRACING_V2"]
    os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGCHAIN_PROJECT"]

st.set_page_config(
    page_title="Research Agent",
    page_icon="🔍",
    layout="centered",
)

st.title("Research Agent")
st.caption("Powered by GPT-4o-mini + LangGraph — searches the web and reads sources")

query = st.text_input(
    "Research question",
    placeholder="e.g. What are the latest developments in AI agents?",
)

if st.button("Research", disabled=not query):
    from agent.graph import research_graph

    tool_calls_made = []
    status_placeholder = st.empty()
    answer_placeholder = st.empty()

    with st.spinner(""):
        status_placeholder.caption("Searching the web...")

        result = research_graph.invoke(
            {
                "messages": [HumanMessage(content=query)],
                "sources": [],
            }
        )

        messages = result["messages"]

        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls_made.append(tc["name"])
                    if tc["name"] == "web_search":
                        status_placeholder.caption(
                            f"Searched for: {tc['args'].get('query', '')}"
                        )
                    elif tc["name"] == "read_page":
                        status_placeholder.caption(
                            f"Reading: {tc['args'].get('url', '')[:60]}..."
                        )

        final_answer = messages[-1].content
        status_placeholder.empty()

    answer_placeholder.markdown(final_answer)

    with st.expander("How this answer was produced"):
        st.write(f"**Total turns:** {len(messages)}")
        st.write(f"**Tools used:** {tool_calls_made}")
        st.write(f"**Tool call count:** {len(tool_calls_made)}")

    st.divider()
    st.caption("Every run is traced in LangSmith — smith.langchain.com")
