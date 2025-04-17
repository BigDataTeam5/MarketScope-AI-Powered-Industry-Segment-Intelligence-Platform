from typing import Callable, Dict, List
from functools import wraps
from mcp.server.fastmcp import FastMCP
from langgraph import Tool
from langgraph.prebuilt import create_react_agent
class ToolConverter:
    @staticmethod
    async def convert_mcp_to_langgraph(mcp_tools: Dict[str, Callable]) -> List[Tool]:
        """Convert MCP tools to LangGraph tools"""
        langgraph_tools = []
        for name, func in mcp_tools.items():
            # Convert to LangGraph tool format
            tool = Tool(
                name=name,
                func=func,
                description=func.__doc__ or ""
            )
            langgraph_tools.append(tool)
        return langgraph_tools