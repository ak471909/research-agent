# Research Agent

An AI agent that researches any topic by searching the web, reading relevant pages, and synthesising a cited answer — built with LangGraph and GPT-4o-mini.

**Live demo:** https://research-agent-5sntfnpuafd7esxuitkwsh.streamlit.app/

---

## What it does

Ask it any research question. The agent:
1. Searches the web using Tavily
2. Reads the most relevant pages in full
3. Synthesises a structured answer with cited sources

---

## Architecture
```
User query
    |
    v
[LangGraph StateGraph]
    |
    +-- agent_node (GPT-4o-mini with tools bound)
    |       |
    |       +-- calls web_search (Tavily)
    |       +-- calls read_page (httpx + BeautifulSoup)
    |       |
    |       v
    +-- tools_condition (did the model call a tool?)
    |       |
    |       yes --> tools node --> back to agent
    |       no  --> END (final answer shown in UI)
```

---

## Architecture decisions

### Why LangGraph over LangChain AgentExecutor?
LangGraph models the agent as an explicit state machine — each node is a pure function, edges are inspectable, and conditional logic is clear. AgentExecutor is a black box that makes debugging and extending difficult. For a production agent, the graph model is the right foundation.

### Why GPT-4o-mini?
Strong tool calling support, low cost (~$0.001 per agent run), and reliable structured output. The LangChain abstraction means switching to a different model is two lines of code if needed.

### Why Tavily over raw search?
Tavily is purpose-built for LLM agents — returns clean markdown snippets rather than raw HTML, handles rate limiting gracefully, and the free tier covers development and demos comfortably.

### Why Streamlit over FastAPI + React?
Streamlit deploys directly from a GitHub repo to Streamlit Cloud in minutes — no server config, no build pipeline. For a portfolio project where the goal is a live demo URL, Streamlit is the right trade-off. A production version would use FastAPI with a proper frontend.

### Guardrails
- Max 10 agent iterations per run to prevent infinite loops
- Max 3 read_page calls per run to prevent excessive API usage
- Graceful error handling on all tool calls — failures return informative messages rather than crashing the agent

---

## Setup
```bash
git clone https://github.com/ak471909/research-agent
cd research-agent
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Add your API keys to .env
python test_day1.py
```

## API keys needed

| Key | Where to get | Cost |
|-----|-------------|------|
| `OPENAI_API_KEY` | platform.openai.com | Pay per use (~$0.001/run) |
| `TAVILY_API_KEY` | app.tavily.com | Free tier |
| `LANGCHAIN_API_KEY` | smith.langchain.com | Free tier |

## Run locally
```bash
streamlit run app/main.py
```

## Run with Docker
```bash
docker build -t research-agent .
docker run -p 8501:8501 --env-file .env research-agent
```

---

## Observability

Every agent run is traced in LangSmith — tool calls, latency, token usage, and full message history are visible per run at smith.langchain.com.

---

## Build log

| Day | What was built |
|-----|---------------|
| 1 | Repo setup, virtual environment, API connections verified |
| 2 | AgentState TypedDict, web_search and read_page tools |
| 3 | LangGraph agent loop — full ReAct pattern working end to end |
| 4 | read_page integration, improved synthesis prompt with structured output |
| 5 | LangSmith tracing — full observability on every run |
| 6 | Streamlit UI — agent accessible via browser |
| 7 | Progressive status updates showing live tool calls during research |
| 8 | Guardrails — max iterations, read_page cap, graceful error handling |
| 9 | Dockerfile and .dockerignore — agent containerised |
| 10 | Deployed to Streamlit Cloud — live public URL |
| 11 | Evaluation golden dataset |
| 12 | LLM-as-judge eval script |
| 13 | README finalised |
| 14 | LinkedIn post and CV update |