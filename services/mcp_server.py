"""
MCP Server for MarketScope AI
Main entry point for the Model Context Protocol server
"""
import os
import importlib
import inspect
import sys
from mcp.server.fastmcp import FastMCP
from typing import Dict, Any, Callable, List

from config import Config, litellm_service
from agents import MarketingManagementAgent

def load_tool_functions(module_path: str) -> Dict[str, Callable]:
    """
    Dynamically load tool functions from a specified module path.
    The module should have a 'tool_functions' dictionary mapping tool names to functions.
    """
    try:
        module = importlib.import_module(module_path)
        if hasattr(module, 'tool_functions') and isinstance(module.tool_functions, dict):
            return module.tool_functions
        else:
            raise AttributeError(f"Module {module_path} does not have a 'tool_functions' dictionary")
    except Exception as e:
        print(f"Error loading tools from {module_path}: {str(e)}")
        return {}

def register_tools(mcp: FastMCP, tools: Dict[str, Callable]) -> List[str]:
    """Register tools with the MCP server and return list of registered tools."""
    registered = []
    for name, func in tools.items():
        # Only register tools that are in the enabled list
        if name in Config.ENABLED_TOOLS:
            print(f"Registering tool: {name}")
            mcp.tool(name=name)(func)
            registered.append(name)
        else:
            print(f"Skipping disabled tool: {name}")
    
    return registered

def validate_environment():
    """Validate environment before starting the server."""
    missing = Config.validate_config()
    if missing:
        print(f"ERROR: Missing required environment variables: {', '.join(missing)}")
        print("Please add these variables to your .env file")
        sys.exit(1)

def main():
    """Main entry point for the MCP server."""
    # Validate environment variables
    validate_environment()
    
    # Create MCP server instance
    mcp = FastMCP("MarketScope")
    
    # Track registered tools
    registered_tools = []
    
    # Load and register marketing tools
    marketing_tools = load_tool_functions("agents.marketing_management_book.marketing_tools")
    registered_tools.extend(register_tools(mcp, marketing_tools))
    
   
    # Print registered tools
    print(f"Registered tools: {registered_tools}")
    
    # Run the server - don't use port parameter if not supported
    print(f"Starting server on address: http://0.0.0.0:{Config.MCP_PORT}")
    # Note: MCP server seems to use a fixed port, so we're not passing it as a parameter
    mcp.run(transport="sse")

if __name__ == "__main__":
    print("Starting MarketScope MCP Server...")
    try:
        main()
    except KeyboardInterrupt:
        print("\nShutting down MCP server...")
    except Exception as e:
        print(f"Error starting MCP server: {str(e)}")
        sys.exit(1)
