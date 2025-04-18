"""
Utility functions for MarketScope agents
"""
from typing import Callable, Dict, List, Any, Awaitable, Optional
import inspect
import functools
import logging
import json
from langchain_core.tools import BaseTool, tool

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tools_utils")

class ToolConverter:
    """
    Converts MCP tools to LangGraph-compatible tools
    """
    
    @staticmethod
    async def convert_mcp_to_langgraph(mcp_tools: Dict[str, Callable]) -> List[BaseTool]:
        """
        Convert MCP tools to LangGraph tools
        
        Args:
            mcp_tools: Dictionary of MCP tool functions
            
        Returns:
            List of LangGraph-compatible tools
        """
        langgraph_tools = []
        
        for name, func in mcp_tools.items():
            try:
                # Get description from docstring
                description = func.__doc__ or f"Tool for {name}"
                
                # Check if function is async
                is_async = inspect.iscoroutinefunction(func)
                
                if is_async:
                    # Create async LangChain tool
                    @tool(name=name, description=description)
                    async def async_tool_wrapper(**kwargs):
                        try:
                            result = await func(**kwargs)
                            return ToolConverter.format_tool_output(result)
                        except Exception as e:
                            logger.error(f"Error in {name}: {str(e)}")
                            return f"Error executing {name}: {str(e)}"
                    
                    # Add attributes to preserve the original function's metadata
                    async_tool_wrapper.__name__ = name
                    async_tool_wrapper.__qualname__ = name
                    async_tool_wrapper.__module__ = func.__module__ if hasattr(func, "__module__") else "__main__"
                    
                    langgraph_tools.append(async_tool_wrapper)
                    
                else:
                    # Create synchronous LangChain tool
                    @tool(name=name, description=description)
                    def sync_tool_wrapper(**kwargs):
                        try:
                            result = func(**kwargs)
                            return ToolConverter.format_tool_output(result)
                        except Exception as e:
                            logger.error(f"Error in {name}: {str(e)}")
                            return f"Error executing {name}: {str(e)}"
                    
                    # Add attributes to preserve the original function's metadata
                    sync_tool_wrapper.__name__ = name
                    sync_tool_wrapper.__qualname__ = name
                    sync_tool_wrapper.__module__ = func.__module__ if hasattr(func, "__module__") else "__main__"
                    
                    langgraph_tools.append(sync_tool_wrapper)
                
            except Exception as e:
                logger.error(f"Error converting tool {name}: {str(e)}")
        
        return langgraph_tools

    @staticmethod
    def format_tool_output(output: Any) -> str:
        """
        Format tool output for display to the user
        
        Args:
            output: Output from MCP tool invocation
            
        Returns:
            Formatted string representation of the output
        """
        # If output is None, return empty string
        if output is None:
            return ""
            
        # If output is already a string, return it
        if isinstance(output, str):
            return output
            
        # If output is a dict, format it based on content
        if isinstance(output, dict):
            # If the output has a 'status' and it's 'error', return the error message
            if output.get('status') == 'error':
                return f"Error: {output.get('message', 'Unknown error')}"
            
            # For visualizations, return info and file path
            if 'image_base64' in output:
                title = output.get('title', 'Visualization')
                file_path = output.get('file_path', '')
                return f"{title}\n![Visualization]({file_path})\n\nCreated visualization for {title}"
            
            # For dataframes or data outputs, return a formatted version
            if 'data' in output:
                if isinstance(output['data'], dict):
                    return f"Data: {json.dumps(output['data'], indent=2)}"
                else:
                    return f"Data: {str(output['data'])}"
            
            # For analysis results, return the full output
            if 'analysis' in output:
                return f"Analysis: {output['analysis']}"
                
            # For general success messages
            if output.get('status') == 'success':
                if 'message' in output:
                    return f"Success: {output['message']}"
                elif 'result' in output:
                    return f"Result: {output['result']}"
                    
            # For product insights
            if 'product_performance' in output:
                products = output.get('product_performance', {})
                result = "Product Performance Summary:\n"
                for product, metrics in products.items():
                    result += f"- {product}: Revenue: ${metrics.get('REVENUE', 0):,.2f}, "
                    result += f"Units: {metrics.get('UNITS_SOLD', 0):,}\n"
                return result
                
        # If we can't determine a specific format, convert to string
        try:
            import json
            return json.dumps(output, indent=2)
        except:
            return str(output)
