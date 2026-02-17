import os
import requests
from typing import Literal, Annotated
from fastmcp import FastMCP
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from utils import tavily_client, fetch_webpage_content

# Initialize FastMCP for Financial Tools
mcp = FastMCP("FinancialTools")


async def tavily_search_limited(query: str, max_results: int = 1) -> str:
    """Simplified tavily search for financial tools."""
    try:
        search_results = tavily_client.search(query, max_results=max_results)
        result_texts = []
        for result in search_results.get("results", []):
            url = result["url"]
            title = result["title"]
            content = await fetch_webpage_content(url)
            result_texts.append(f"## {title}\n**URL:** {url}\n\n{content}")
        return "\n\n".join(result_texts) if result_texts else "No results found."
    except Exception as e:
        return f"Search failed: {str(e)}"


@mcp.tool()
async def crawl_for_finance(
    url: str, extraction_goal: str = "Extract key financial metrics and sentiment"
) -> str:
    """
    FOR ALL AGENTS: Uses Crawl4AI to scrape dynamic, JS-heavy financial sites.
    Converts the page to clean Markdown with an emphasis on your 'extraction_goal'.
    """
    browser_config = BrowserConfig(headless=True)
    # We use a 'Fit' strategy to remove headers/footers and keep financial data
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.ENABLED,
        word_count_threshold=10,
        exclude_external_links=True,
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(url=url, config=run_config)
        if result.success:
            # We return a focused slice of the markdown to keep it lean
            return f"Crawl successful for {url}:\n\n{result.markdown[:12000]}"
        return f"Crawl failed: {result.error_message}"


@mcp.tool()
def get_detailed_ratios(ticker: str) -> str:
    """
    FOR QUANT_AUDITOR: Fetches P/E, PEG, ROE, and Debt-to-Equity directly.
    Requires FMP_API_KEY in .env.
    """
    api_key = os.getenv("FMP_API_KEY")
    url = f"https://financialmodelingprep.com/api/v3/key-metrics-ttm/{ticker}?apikey={api_key}"
    try:
        r = requests.get(url)
        data = r.json()
        if not data:
            return "No data found for this ticker."
        # Filter for the most important metrics only
        m = data[0]
        summary = {
            "PE": m.get("peRatioTTM"),
            "PEG": m.get("pegRatioTTM"),
            "ROE": m.get("returnOnEquityTTM"),
            "Debt_Equity": m.get("debtToEquityTTM"),
            "FCF_Yield": m.get("freeCashFlowYieldTTM"),
        }
        return str(summary)
    except Exception as e:
        return f"Ratio fetch failed: {str(e)}"


@mcp.tool()
async def get_insider_signals(ticker: str) -> str:
    """
    FOR SENTIMENT_PROCESSOR: Checks if CEOs/CFOs are buying or selling stock.
    High insider buying is a strong 'Bullish' signal.
    """
    api_key = os.getenv("FMP_API_KEY")
    url = f"https://financialmodelingprep.com/api/v3/insider-trading/{ticker}?limit=10&apikey={api_key}"
    try:
        r = requests.get(url)
        data = r.json()
        # Summarize the last 10 trades
        trades = [
            f"{t['type']} by {t['reportingName']} ({t['securitiesTransacted']} shares)"
            for t in data
        ]
        return "\n".join(trades) if trades else "No recent insider trades."
    except Exception as e:
        return f"Insider data failed: {str(e)}"


@mcp.tool()
def get_company_financials(
    ticker: str, period: Literal["annual", "quarter"] = "annual"
) -> str:
    """
    FOR QUANT_AUDITOR: Fetches Income Statement, Balance Sheet, and Cash Flow.
    Use this to calculate Margins, FCF, and Debt-to-Equity.
    """
    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        return "Error: FMP_API_KEY missing."

    base_url = f"https://financialmodelingprep.com/api/v3/income-statement/{ticker}"
    params = {"limit": 4, "period": period, "apikey": api_key}

    try:
        r = requests.get(base_url, params=params)
        data = r.json()
        return str(data[:2])  # Return last 2 periods for comparison
    except Exception as e:
        return f"Financial fetch failed: {str(e)}"


@mcp.tool()
def get_valuation_multiples(ticker: str) -> str:
    """
    FOR QUANT_AUDITOR: Fetches P/E, PEG, EV/EBITDA, and Price-to-Book ratios.
    Essential for determining if a stock is over or undervalued.
    """
    api_key = os.getenv("FMP_API_KEY")
    url = (
        f"https://financialmodelingprep.com/api/v3/ratios-ttm/{ticker}?apikey={api_key}"
    )
    try:
        r = requests.get(url)
        return str(r.json())
    except Exception as e:
        return f"Valuation fetch failed: {str(e)}"


@mcp.tool()
async def search_sec_filings(ticker: str, filing_type: str = "10-K") -> str:
    """
    FOR RISK_FORENSIC: Searches for the most recent SEC filings.
    Use this to identify 'Legal Proceedings' and 'Risk Factors'.
    """
    query = f"site:sec.gov {ticker} {filing_type} filing interactive data"
    return await tavily_search_limited(query=query, max_results=1)


@mcp.tool()
async def get_earnings_transcript(ticker: str, quarter: int, year: int) -> str:
    """
    FOR SENTIMENT_PROCESSOR: Retrieves the full text of an earnings call.
    Analyze this for 'Management Tone' and 'Hedging Language'.
    """
    api_key = os.getenv("FMP_API_KEY")
    url = f"https://financialmodelingprep.com/api/v3/earning_call_transcript/{ticker}?quarter={quarter}&year={year}&apikey={api_key}"
    try:
        r = requests.get(url)
        transcript_data = r.json()
        if transcript_data:
            return transcript_data[0].get("content", "No transcript content found.")[
                :15000
            ]
        return "No transcript available for this period."
    except Exception as e:
        return f"Transcript fetch failed: {str(e)}"


@mcp.tool()
async def analyze_competitors(ticker: str) -> str:
    """
    FOR MOAT_STRATEGIST: Finds top competitors and their recent news.
    Used for Porter's Five Forces analysis.
    """
    api_key = os.getenv("FMP_API_KEY")
    peers_url = f"https://financialmodelingprep.com/api/v3/stock_peers?symbol={ticker}&apikey={api_key}"
    try:
        peers_response = requests.get(peers_url).json()
        peers = peers_response[0].get("peersList", []) if peers_response else []
        # Step 2: Search for recent market share shifts
        search_query = (
            f"Market share comparison {ticker} vs {' vs '.join(peers[:3])} 2025 2026"
        )
        return await tavily_search_limited(query=search_query, max_results=2)
    except Exception as e:
        return f"Competitor analysis failed: {str(e)}"


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
