import os
import requests
from typing import Optional, List, Annotated, Literal, Any
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field
from fastmcp import FastMCP
from utils import tavily_client, fetch_webpage_content

# LangChain imports
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# Local imports
try:
    from mongdbclient import collection, index_name
except ImportError:
    # Fallback if mongdbclient is not properly configured
    collection = None
    index_name = "vector_index"

# Initialize FastMCP for Basic Tools
mcp = FastMCP("BasicTools")


# Models for YouTube Tool
class VideoSearchResult(BaseModel):
    title: str
    video_url: str


class YoutubeVideoSearchToolInput(BaseModel):
    """Input for YoutubeVideoSearchTool."""

    keyword: str = Field(..., description="The search keyword.")
    max_results: int = Field(10, description="The maximum number of results to return.")


# Vector Store Setup
_vectorstore = None
_embeddings = None


def get_vectorstore():
    global _vectorstore, _embeddings
    if _vectorstore is None:
        if collection is None:
            raise ValueError(
                "MongoDB collection not configured. Please check mongdbclient.py and .env"
            )
        _embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        _vectorstore = MongoDBAtlasVectorSearch(
            embedding=_embeddings, collection=collection, index_name=index_name
        )
    return _vectorstore


@mcp.tool()
def retrieve_mercedes_info(query: str) -> str:
    """Useful for when you need to answer questions about Mercedes-Benz company using the internal blog database."""
    try:
        vs = get_vectorstore()
        retriever = vs.as_retriever()
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant information found in the Mercedes blog database."
        return "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        return f"Error accessing vector store: {str(e)}"


@mcp.tool()
def get_weather(city: str, country: Optional[str] = None) -> str:
    """Retrieves current weather information for a given location."""
    api_key = os.getenv("OPENWEATHERMAP_API_KEY", "b947d60a697680e632ac0c2b5e9f4c0e")
    base_url = "https://api.openweathermap.org/data/2.5/weather"

    params = {
        "q": f"{city},{country}" if country else city,
        "appid": api_key,
        "units": "metric",
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        temp = data["main"]["temp"]
        weather_desc = data["weather"][0]["description"]
        return f"Current weather in {city}, {country or 'unknown'} is: {temp}\u00b0C, {weather_desc}"
    except Exception as e:
        return f"Failed to retrieve weather for {city}. Error: {str(e)}"


@mcp.tool()
def double_number(number: float) -> float:
    """Doubles the given number."""
    return number * 2


@mcp.tool()
def divide_numbers(number1: float, number2: float) -> str:
    """Divides the first number by the second number."""
    if number2 == 0:
        return "Error: Division by zero"
    return str(number1 / number2)


@mcp.tool()
def scrape_web_page(url: str) -> str:
    """Parses web content from a URL using BeautifulSoup and returns the text."""
    try:
        response = requests.get(url=url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        # Clean up some common clutter
        for script in soup(["script", "style"]):
            script.extract()
        text = soup.get_text(separator="\n", strip=True)
        return text[:10000]  # Cap output to 10k chars
    except Exception as e:
        return f"Error scraping {url}: {str(e)}"


@mcp.tool()
def search_youtube(keyword: str, max_results: int = 5) -> List[dict]:
    """Searches YouTube videos based on a keyword and returns a list of video search results."""
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        return [{"error": "YOUTUBE_API_KEY not found in environment"}]

    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": keyword,
        "maxResults": max_results,
        "type": "video",
        "key": api_key,
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        items = response.json().get("items", [])

        results = []
        for item in items:
            title = item["snippet"]["title"]
            video_id = item["id"]["videoId"]
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            results.append({"title": title, "url": video_url})
        return results
    except Exception as e:
        return [{"error": f"Error searching YouTube: {str(e)}"}]


@mcp.tool()
def create_document(content_html: str, output_format: str, filename: str) -> str:
    """
    Creates a document (docx, pdf, or txt) from HTML content using a Gotenberg instance.
    Args:
        content_html: The full content for the document, formatted as HTML.
        output_format: The file extension ('docx', 'pdf', or 'txt').
        filename: The name of the file (e.g., 'report').
    """
    gotenberg_url = os.getenv(
        "GOTENBERG_URL", "http://localhost:3000/forms/libreoffice/convert"
    )
    files = {"files": ("index.html", content_html)}
    full_filename = f"{filename}.{output_format}"

    try:
        response = requests.post(gotenberg_url, files=files, timeout=60)
        response.raise_for_status()

        # Save to local output directory
        output_dir = os.path.join(os.getcwd(), "output")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, full_filename)

        with open(output_path, "wb") as f:
            f.write(response.content)

        return (
            f"Successfully created {output_format.upper()} document at: {output_path}"
        )
    except Exception as e:
        return f"Failed to create document: {str(e)}. (Ensure Gotenberg is running at {gotenberg_url})"


@mcp.tool()
def wikipedia_search(query: str) -> str:
    """Searches Wikipedia for a given query and returns a summary."""
    try:
        wiki_wrapper = WikipediaAPIWrapper(
            top_k_results=2, lang="en", doc_content_chars_max=3000
        )
        wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)
        return wiki_tool.invoke(query)
    except Exception as e:
        return f"Wikipedia search failed: {str(e)}"


@mcp.tool()
async def crawl_web_page(url: str) -> str:
    """
    Crawls a single URL using Crawl4AI and returns its content in clean markdown format.
    Advanced version of scrape_web_page that handles JavaScript-heavy sites.
    """
    return await fetch_webpage_content(url)


@mcp.tool()
async def tavily_search(
    query: str,
    max_results: int = 3,
    topic: str = "general",
) -> str:
    """
    Advanced web search using Tavily.
    Discovers relevant URLs, then fetches and returns full webpage content as markdown using Crawl4AI.

    Args:
        query: Search query to execute
        max_results: Maximum number of results to return (default: 1)
        topic: Topic filter - 'general', 'news', or 'finance' (default: 'general')
    """
    try:
        # Use Tavily to discover URLs
        search_results = tavily_client.search(
            query,
            max_results=max_results,
            topic=topic,
        )

        # Fetch full content for each URL
        result_texts = []
        for result in search_results.get("results", []):
            url = result["url"]
            title = result["title"]

            # Fetch webpage content using Crawl4AI
            content = await fetch_webpage_content(url)

            result_text = f"## {title}\n**URL:** {url}\n\n{content}\n\n---\n"
            result_texts.append(result_text)

        # Format final response
        if not result_texts:
            return f"No results found for '{query}'."

        response = (
            f"\ud83d\udd0d Found {len(result_texts)} result(s) for '{query}':\n\n"
            + "\n".join(result_texts)
        )
        return response
    except Exception as e:
        return f"Search failed: {str(e)}"


@mcp.tool()
def think_tool(
    analysis: Annotated[str, "Detailed analysis of current findings"],
    gaps: Annotated[str, "Identification of missing information"],
    next_steps: Annotated[str, "Plan for the next research phase"],
) -> str:
    """
    A strategic reflection tool to analyze research progress and plan next steps.
    Helps organize thoughts and ensure high-quality decision making during research.
    """
    reflection = f"""
### \ud83e\udde0 Strategic Reflection
**Current Analysis:**
{analysis}

**Information Gaps:**
{gaps}

**Plan of Action:**
{next_steps}
"""
    return reflection
