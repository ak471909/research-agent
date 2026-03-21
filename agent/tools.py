import os
import httpx
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from tavily import TavilyClient


@tool
def web_search(query: str) -> str:
    """
    Search the web for current information on a topic.
    Returns the top results with titles, URLs, and snippets.
    Use this when you need up-to-date facts or information.
    """
    try:
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        response = client.search(query=query, max_results=5)

        if not response.get("results"):
            return "No results found."

        formatted = []
        for r in response["results"]:
            formatted.append(
                f"Title: {r['title']}\n"
                f"URL: {r['url']}\n"
                f"Snippet: {r['content']}\n"
            )

        return "\n---\n".join(formatted)
    except Exception as e:
        return f"Search failed: {str(e)}. Ty a different search query."


@tool
def read_page(url: str) -> str:
    """
    Fetch and read the main text content from a URL.
    Use this to read the full content of a page found in search results.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; ResearchAgent/1.0)"}
        response = httpx.get(url, headers=headers, timeout=10, follow_redirects=True)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        lines = [line for line in text.splitlines() if len(line.strip()) > 40]
        cleaned = "\n".join(lines)

        return cleaned[:4000] if cleaned else "Could not extract text from this page."

    except httpx.TimeoutException:
        return f"Timeout reading {url} — page took too long to load, skip this source."
    except httpx.HTTPStatusError as e:
        return f"HTTP {e.response.status_code} error reading {url} - skip this source."
    except Exception as e:
        return f"Could not read {url}: {str(e)} - skip this source and try another."


TOOLS = [web_search, read_page]
