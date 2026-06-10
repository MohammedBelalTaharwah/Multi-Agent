# mcp_tools.py - Model Context Protocol tool definitions
# Each tool is defined with a name, description, input schema, and handler

from typing import Dict, Any, Callable
from scraper import try_trafilatura


class MCPTool:
    """A tool exposed via MCP with name, schema, and handler."""

    def __init__(self, name: str, description: str, input_schema: Dict, handler: Callable):
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.handler = handler

    def execute(self, params: Dict[str, Any]) -> str:
        try:
            result = self.handler(**params)
            return str(result)
        except Exception as e:
            return f"Error executing {self.name}: {e}"


def _search_tavily(query: str, max_results: int = 5):
    """Search using Tavily (handled by search tool in agents.py)."""
    return f"[SEARCH] Query: {query} (max: {max_results})"


def _scrape_web(url: str):
    """Scrape a web page for full content."""
    return try_trafilatura(url)


def _classify_text(text: str, categories: str):
    """Placeholder for classification via LLM."""
    return f"[CLASSIFY] Categories: {categories}"


def _extract_entities(text: str):
    """Placeholder for NER via LLM."""
    return f"[NER] Extracting entities from text ({len(text)} chars)"


# Define all MCP tools
MCP_TOOLS = [
    MCPTool(
        name="web_search",
        description="Search the web for information on a query",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {"type": "integer", "default": 5}
            },
            "required": ["query"]
        },
        handler=_search_tavily
    ),
    MCPTool(
        name="web_scrape",
        description="Scrape a web page for full article content",
        input_schema={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to scrape"}
            },
            "required": ["url"]
        },
        handler=_scrape_web
    ),
    MCPTool(
        name="classify",
        description="Classify text into given categories",
        input_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to classify"},
                "categories": {"type": "string", "description": "Comma-separated categories"}
            },
            "required": ["text", "categories"]
        },
        handler=_classify_text
    ),
    MCPTool(
        name="extract_entities",
        description="Extract named entities from text",
        input_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to analyze"}
            },
            "required": ["text"]
        },
        handler=_extract_entities
    ),
]


def get_mcp_tool(name: str) -> MCPTool:
    """Get a tool by name."""
    for tool in MCP_TOOLS:
        if tool.name == name:
            return tool
    raise ValueError(f"MCP tool '{name}' not found")


def list_mcp_tools() -> str:
    """List all available MCP tools."""
    lines = []
    for tool in MCP_TOOLS:
        lines.append(f"- {tool.name}: {tool.description}")
    return "\n".join(lines)
