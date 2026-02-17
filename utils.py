import os
from tavily import TavilyClient
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

# Initialize Tavily Client
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


async def fetch_webpage_content(url: str) -> str:
    """Helper to fetch webpage content as markdown using Crawl4AI."""
    browser_config = BrowserConfig(headless=True, verbose=False)
    run_config = CrawlerRunConfig(cache_mode=CacheMode.ENABLED)

    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(url=url, config=run_config)
        if result.success:
            return result.markdown
        return f"Error crawling {url}: {result.error_message}"
