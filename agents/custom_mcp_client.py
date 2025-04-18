"""
Custom MCP Client for connecting to MCP servers
"""
import logging
import json
import aiohttp
import asyncio
from typing import Dict, Any, List, Optional, Union
import os

# Import Config
try:
    from config.config import Config
except ImportError:
    # Default config if import fails
    class Config:
        MCP_PORT = 8000
        API_PORT = 8001

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("custom_mcp_client")

class CustomMCPClient:
    """
    Custom client for connecting to MCP servers.
    Provides functionality for tool discovery and invocation.
    """
    
    def __init__(self, base_url: str, timeout: int = 30):
        """
        Initialize the MCP client
        
        Args:
            base_url: URL of the MCP server
            timeout: Request timeout in seconds
        """
        self.base_url = base_url
        self.timeout = timeout
        self.tools_cache = None
        self.session = None
    
    async def _ensure_session(self):
        """Ensure aiohttp session is initialized"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
    
    async def get_tools(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Get list of available tools from the MCP server
        
        Args:
            force_refresh: If True, bypass cache and query the server
            
        Returns:
            List of tool definitions
        """
        try:
            # Return cached tools if available and not forced to refresh
            if self.tools_cache is not None and not force_refresh:
                return self.tools_cache
            
            # Ensure session is initialized
            await self._ensure_session()
            
            # Make request to tools endpoint
            tools_url = f"{self.base_url}/mcp/tools"
            logger.info(f"Fetching tools from: {tools_url}")
            async with self.session.get(tools_url) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Error getting tools: {error_text}")
                    return []
                
                # Parse response
                tools_data = await response.json()
                
                # Expected format: {"tools": [...]}
                tools = tools_data.get("tools", [])
                
                # Cache tools
                self.tools_cache = tools
                
                return tools
                
        except Exception as e:
            logger.error(f"Error getting tools: {str(e)}")
            return []
    
    async def invoke(self, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """
        Invoke a tool with specified parameters
        
        Args:
            tool_name: Name of the tool to invoke
            parameters: Tool parameters
            
        Returns:
            Tool result
        """
        try:
            # Ensure session is initialized
            await self._ensure_session()
            
            # Make request to invoke endpoint
            invoke_url = f"{self.base_url}/mcp/tools/{tool_name}/invoke"
            
            # Prepare request payload
            payload = {
                "parameters": parameters
            }
            
            logger.info(f"Invoking tool {tool_name} at {invoke_url} with parameters: {parameters}")
            
            # Make POST request
            async with self.session.post(invoke_url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Error invoking tool {tool_name}: {error_text}")
                    return f"Error invoking tool {tool_name}: {error_text}"
                
                # Parse response
                result_data = await response.json()
                
                # Extract content from result
                content = result_data.get("content")
                
                return content
                
        except Exception as e:
            logger.error(f"Error invoking tool {tool_name}: {str(e)}")
            return f"Error invoking tool {tool_name}: {str(e)}"
    
    def invoke_sync(self, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """
        Synchronous version of invoke for non-async contexts
        
        Args:
            tool_name: Name of the tool to invoke
            parameters: Tool parameters
            
        Returns:
            Tool result
        """
        # Create event loop if not existing
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        # Run the async invoke method
        return loop.run_until_complete(self.invoke(tool_name, parameters))
    
    async def close(self):
        """Close the HTTP session"""
        if self.session and not self.session.closed:
            await self.session.close()
            
    async def __aenter__(self):
        """Async context manager enter"""
        await self._ensure_session()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()


class MCPClient:
    """
    MCP Client wrapper for different services.
    Provides easy access to different MCP servers based on service name.
    """
    
    # Map of service names to their base URLs with specific ports
    SERVICE_URLS = {
        "snowflake": "http://localhost:8004",
        "market_analysis": "http://localhost:8001",
        "segment": "http://localhost:8003",
        "sales_analytics": "http://localhost:8002",
        "unified": f"http://localhost:{Config.MCP_PORT}",
        "marketscope": f"http://localhost:{Config.MCP_PORT}"
    }
    
    def __init__(self, service_name: str):
        """
        Initialize an MCP client for a specific service
        
        Args:
            service_name: Name of the service (snowflake, market_analysis, etc.)
        """
        if service_name not in self.SERVICE_URLS:
            raise ValueError(f"Unknown service: {service_name}")
        
        # Get service URL
        service_url = self.SERVICE_URLS[service_name]
        
        # Override from environment if available
        env_url = os.environ.get(f"{service_name.upper()}_MCP_URL")
        if env_url:
            service_url = env_url
            
        logger.info(f"Initializing {service_name} MCP client with URL: {service_url}")
            
        # Create CustomMCPClient
        self.client = CustomMCPClient(service_url)
        self.service_name = service_name
        
    async def get_tools(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Get tools from the service
        
        Args:
            force_refresh: Whether to force refresh tool cache
            
        Returns:
            List of tool definitions
        """
        return await self.client.get_tools(force_refresh)
        
    async def invoke(self, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """
        Invoke a tool on the service
        
        Args:
            tool_name: Name of the tool to invoke
            parameters: Tool parameters
            
        Returns:
            Tool result
        """
        return await self.client.invoke(tool_name, parameters)
        
    def invoke_sync(self, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """
        Synchronous version of invoke
        
        Args:
            tool_name: Name of the tool to invoke
            parameters: Tool parameters
            
        Returns:
            Tool result
        """
        return self.client.invoke_sync(tool_name, parameters)
        
    async def close(self):
        """Close the client"""
        await self.client.close()
