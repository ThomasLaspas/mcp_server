import uvicorn
from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the MCP instances from our separate tool components
from basic_tools import mcp as basic_mcp
from financial_tools import mcp as financial_mcp

# Create the main MCP server instance
mcp = FastMCP("ProAgent")


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> PlainTextResponse:
    return PlainTextResponse("OK")


# Get the Starlette app from our main MCP instance
# We'll use this as the foundation for our multi-route server
app = mcp.http_app(transport="sse")

# Mount the basic and financial toolsets as sub-apps
# Using path="/" for the sub-apps allows them to handle SSE at the mounted root
app.mount("/mcp/basic", basic_mcp.http_app(transport="sse", path="/"))
app.mount("/mcp/financial", financial_mcp.http_app(transport="sse", path="/"))

if __name__ == "__main__":
    print("🚀 Starting ProAgent MCP Server on port 8001...")
    print(f"📡 Health Check:     http://127.0.0.1:8001/health")
    print(f"🛠️ Basic Tools:      http://127.0.0.1:8001/mcp/basic")
    print(f"💰 Financial Tools:  http://127.0.0.1:8001/mcp/financial")

    uvicorn.run(app, host="127.0.0.1", port=8001)
