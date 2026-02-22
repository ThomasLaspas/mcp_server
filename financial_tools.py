import os
import requests
import json
from typing import Literal, Annotated, Optional
from fastmcp import FastMCP
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from utils import tavily_client, fetch_webpage_content
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

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


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 6 — run_pandas_query
# ─────────────────────────────────────────────────────────────────────────────
@mcp.tool()
def run_pandas_query(
    csv_path: str,
    query: str,
    output_path: Optional[str] = None,
) -> dict:
    """
    Load a CSV and run a pandas query expression or a groupby/agg operation,
    returning the result as a JSON string.

    Args:
        csv_path:    Path to CSV, e.g. /workspace/clean_data.csv
        query:       A pandas query string, e.g. "Revenue > 1000 and Year == 2023"
                     OR a JSON groupby spec:
                     '{"groupby": ["Year","Segment"], "agg": {"Revenue": "sum"}}'
        output_path: Optional path to save the filtered result as CSV.

    Returns:
        {"result": str (JSON), "rows": int, "columns": [str], "success": bool}
    """
    try:
        import pandas as pd

        df = pd.read_csv(csv_path)

        if query.strip().startswith("{"):
            # Groupby spec
            spec = json.loads(query)
            result = df.groupby(spec["groupby"]).agg(spec["agg"]).reset_index()
        else:
            # Standard pandas query expression
            result = df.query(query)

        if output_path:
            result.to_csv(output_path, index=False)

        return {
            "result": result.to_json(orient="records", date_format="iso"),
            "rows": len(result),
            "columns": list(result.columns),
            "success": True,
        }
    except Exception as e:
        return {
            "result": "[]",
            "rows": 0,
            "columns": [],
            "success": False,
            "error": str(e),
        }


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 7 — run_financial_ratio_calculator
# ─────────────────────────────────────────────────────────────────────────────
@mcp.tool()
def run_financial_ratio_calculator(
    financials_json: str,
    company_name: str = "Company",
    output_path: str = "/workspace/financial_ratios.csv",
) -> dict:
    """
    Compute a comprehensive 20-ratio financial dashboard from raw statement data
    and save the results to a CSV.

    Args:
        financials_json: JSON string with keys:
            revenue, gross_profit, ebitda, ebit, net_income, interest_expense,
            total_assets, total_equity, total_debt, cash, current_assets,
            current_liabilities, inventory, capex, operating_cash_flow,
            shares_outstanding, market_cap, enterprise_value
            Each key maps to a list of values (most recent period first).
        company_name: Used in output labels.
        output_path:  Where to save the ratios CSV (default /workspace/financial_ratios.csv)

    Returns:
        {"ratios": dict, "saved_to": str, "success": bool}
    """
    try:
        import pandas as pd

        d = json.loads(financials_json)

        def safe_div(a, b, scale=1):
            try:
                return round((a / b) * scale, 4) if b and b != 0 else None
            except Exception:
                return None

        # Most recent period values
        rev = d.get("revenue", [None])[0]
        gp = d.get("gross_profit", [None])[0]
        ebitda = d.get("ebitda", [None])[0]
        ebit = d.get("ebit", [None])[0]
        ni = d.get("net_income", [None])[0]
        int_e = d.get("interest_expense", [None])[0]
        ta = d.get("total_assets", [None])[0]
        eq = d.get("total_equity", [None])[0]
        debt = d.get("total_debt", [None])[0]
        cash = d.get("cash", [None])[0]
        ca = d.get("current_assets", [None])[0]
        cl = d.get("current_liabilities", [None])[0]
        inv = d.get("inventory", [None])[0]
        capex = d.get("capex", [None])[0]
        ocf = d.get("operating_cash_flow", [None])[0]
        mc = d.get("market_cap", [None])[0]
        ev = d.get("enterprise_value", [None])[0]

        ratios = {
            "company": company_name,
            # Profitability
            "gross_margin_pct": safe_div(gp, rev, 100),
            "ebitda_margin_pct": safe_div(ebitda, rev, 100),
            "ebit_margin_pct": safe_div(ebit, rev, 100),
            "net_margin_pct": safe_div(ni, rev, 100),
            "roa_pct": safe_div(ni, ta, 100),
            "roe_pct": safe_div(ni, eq, 100),
            "roic_pct": safe_div(ebit, (debt + eq), 100) if debt and eq else None,
            # Liquidity
            "current_ratio": safe_div(ca, cl),
            "quick_ratio": safe_div((ca - inv) if ca and inv else ca, cl),
            "cash_ratio": safe_div(cash, cl),
            # Leverage
            "debt_to_equity": safe_div(debt, eq),
            "net_debt_to_ebitda": safe_div(
                (debt - cash) if debt and cash else debt, ebitda
            ),
            "interest_coverage": safe_div(ebit, int_e),
            # Cash Flow
            "fcf": (ocf - capex) if ocf and capex else None,
            "capex_to_revenue_pct": safe_div(capex, rev, 100),
            "ocf_to_net_income": safe_div(ocf, ni),
            # Valuation
            "pe_ratio": safe_div(mc, ni),
            "ev_to_ebitda": safe_div(ev, ebitda),
            "ev_to_revenue": safe_div(ev, rev),
            "price_to_book": safe_div(mc, eq),
            "price_to_fcf": safe_div(mc, (ocf - capex)) if ocf and capex else None,
        }

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        pd.DataFrame([ratios]).to_csv(output_path, index=False)

        return {
            "ratios": ratios,
            "saved_to": output_path,
            "success": True,
        }
    except Exception as e:
        return {"ratios": {}, "saved_to": "", "success": False, "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 8 — run_monte_carlo_dcf
# ─────────────────────────────────────────────────────────────────────────────
@mcp.tool()
def run_monte_carlo_dcf(
    base_revenue: float,
    revenue_growth_mean: float,
    revenue_growth_std: float,
    ebitda_margin_mean: float,
    ebitda_margin_std: float,
    wacc_mean: float,
    wacc_std: float,
    terminal_growth_mean: float = 0.025,
    terminal_growth_std: float = 0.005,
    capex_pct: float = 0.05,
    tax_rate: float = 0.21,
    shares_outstanding: float = 1_000_000,
    current_price: Optional[float] = None,
    n_simulations: int = 10_000,
    forecast_years: int = 5,
    output_path: str = "/workspace/monte_carlo_results.csv",
) -> dict:
    """
    Run a Monte Carlo DCF simulation and return the intrinsic value distribution.

    All rate parameters are in decimal form (0.10 = 10%).

    Returns:
        {
            "intrinsic_value_p10": float,
            "intrinsic_value_p50": float,
            "intrinsic_value_p90": float,
            "prob_above_current_price": float,  (only if current_price provided)
            "prob_upside_20pct": float,
            "prob_downside_20pct": float,
            "mean": float,
            "std": float,
            "saved_to": str,
            "success": bool
        }
    """
    try:
        import numpy as np
        import pandas as pd

        rng = np.random.default_rng(42)

        # Vectorised simulation — draw all N samples at once
        gr = rng.normal(revenue_growth_mean, revenue_growth_std, n_simulations)
        margin = rng.normal(ebitda_margin_mean, ebitda_margin_std, n_simulations)
        wacc = np.clip(rng.normal(wacc_mean, wacc_std, n_simulations), 0.05, None)
        tg = np.clip(
            rng.normal(terminal_growth_mean, terminal_growth_std, n_simulations),
            0.01,
            None,
        )

        # Build FCF streams for each simulation (shape: n_simulations × forecast_years)
        years = np.arange(1, forecast_years + 1)  # [1, 2, ..., 5]
        rev = base_revenue * np.cumprod(
            1 + np.tile(gr, (forecast_years, 1)).T,  # (N, years)
            axis=1,
        )
        fcf = rev * margin * (1 - tax_rate) - rev * capex_pct

        # PV of explicit period FCFs
        discount = (1 + wacc[:, None]) ** years[None, :]  # (N, years)
        pv_fcf = (fcf / discount).sum(axis=1)  # (N,)

        # Terminal value (Gordon Growth on final year FCF)
        terminal_fcf = fcf[:, -1] * (1 + tg)
        tv = np.where(wacc > tg, terminal_fcf / (wacc - tg), 0)
        pv_tv = tv / (1 + wacc) ** forecast_years

        value_per_share = (pv_fcf + pv_tv) / shares_outstanding

        results = {
            "intrinsic_value_p10": round(float(np.percentile(value_per_share, 10)), 2),
            "intrinsic_value_p25": round(float(np.percentile(value_per_share, 25)), 2),
            "intrinsic_value_p50": round(float(np.percentile(value_per_share, 50)), 2),
            "intrinsic_value_p75": round(float(np.percentile(value_per_share, 75)), 2),
            "intrinsic_value_p90": round(float(np.percentile(value_per_share, 90)), 2),
            "mean": round(float(value_per_share.mean()), 2),
            "std": round(float(value_per_share.std()), 2),
            "n_simulations": n_simulations,
        }

        if current_price is not None:
            results["prob_above_current_price"] = round(
                float((value_per_share > current_price).mean()), 4
            )
            results["prob_upside_20pct"] = round(
                float((value_per_share > current_price * 1.2).mean()), 4
            )
            results["prob_downside_20pct"] = round(
                float((value_per_share < current_price * 0.8).mean()), 4
            )

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        pd.DataFrame({"intrinsic_value_per_share": value_per_share}).to_csv(
            output_path, index=False
        )

        return {**results, "saved_to": output_path, "success": True}

    except Exception as e:
        return {"success": False, "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 9 — generate_pandas_profile_report
# ─────────────────────────────────────────────────────────────────────────────
@mcp.tool()
def generate_pandas_profile_report(
    csv_path: str = "/workspace/clean_data.csv",
    output_path: str = "/workspace/profile_report.html",
) -> dict:
    """
    Generate a full ydata-profiling HTML report for a CSV file.

    Returns:
        {"success": bool, "saved_to": str}
    """
    try:
        import pandas as pd
        from ydata_profiling import ProfileReport

        df = pd.read_csv(csv_path)
        profile = ProfileReport(df, title="Data Profile Report", explorative=True)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        profile.to_file(output_path)

        return {"success": True, "saved_to": output_path}
    except Exception as e:
        return {"success": False, "saved_to": "", "error": str(e)}
