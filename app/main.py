import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()

# creating A UI for the page
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

    with st.spinner("Searching and reading sources..."):
        result = research_graph.invoke(
            {
                "messages": [HumanMessage(content=query)],
                "sources": [],
            }
        )

    messages = result["messages"]
    final_answer = messages[-1].content

    tool_calls_made = []
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls_made.append(tc["name"])

    st.markdown(final_answer)

    with st.expander("How this answer was produced"):
        st.write(f"**Total turns:** {len(messages)}")
        st.write(f"**Tools used:** {tool_calls_made}")
        st.write(f"**Tool call count:** {len(tool_calls_made)}")

    st.divider()
    st.caption("Every run is traced in LangSmith — smith.langchain.com")
