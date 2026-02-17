# Professional MCP Server

This is a high-performance MCP (Model Context Protocol) server built with `FastMCP`, integrating advanced tools for search, retrieval, and document generation.

## Features

- **Vector Search**: Mercedes-Benz blog post retrieval via MongoDB Atlas.
- **Web Tools**: Tavily Search, Wikipedia, and Web Scraping.
- **Media**: YouTube Video Search.
- **Utilities**: Weather data, Math (Double/Divide).
- **Document Generation**: HTML to PDF/DOCX/TXT conversion via Gotenberg.

## Prerequisites

- [uv](https://github.com/astral-sh/uv) for dependency management.
- MongoDB Atlas cluster for the vector store.
- API Keys for Tavily, YouTube, and OpenWeatherMap.
- [Gotenberg](https://gotenberg.dev/) running locally for document conversion.

```bash
# Run Gotenberg with Docker
docker run -p 3000:3000 gotenberg/gotenberg:8
```

## Setup

1. **Clone the environment file**:
   ```bash
   cp .env.example .env
   ```
2. **Configure your API keys** in `.env`.

3. **Install dependencies**:
   ```bash
   uv sync
   ```

## Running the Server

To start the MCP server locally:

```bash
uv run server.py
```

## Tools Included

| Tool | Description |
| --- | --- |
| `retrieve_mercedes_info` | Vector search in MongoDB for Mercedes-Benz info. |
| `web_search` | Real-time web search via Tavily. |
| `tavily_search` | Advanced search that crawls full results into markdown. |
| `crawl_web_page` | High-fidelity web crawling using Crawl4AI. |
| `think_tool` | Strategic reflection for complex research tasks. |
| `wikipedia_search` | Search and retrieve summaries from Wikipedia. |
| `search_youtube` | Find relevant YouTube videos. |
| `scrape_web_page` | Extract text from any URL (Basic). |
| `get_weather` | Get current weather for any city. |
| `create_document_tool` | Convert HTML to PDF/DOCX/TXT. |
| `double_number` | Simple math utility. |
| `divide_numbers` | Simple math utility. |

## Note on Crawl4AI

The advanced crawling tools require Playwright browsers:

```bash
uv run playwright install chromium
```
