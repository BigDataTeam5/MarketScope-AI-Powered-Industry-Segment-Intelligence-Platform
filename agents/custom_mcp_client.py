"""
Custom MCP Client for connecting to MCP servers
"""
import logging
import json
import aiohttp
import asyncio
from typing import Dict, Any, List, Optional, Union
import os
import requests

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

class EventLoopManager:
    """Singleton to manage event loops across threads"""
    _instance = None
    _lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EventLoopManager, cls).__new__(cls)
            cls._instance.loop = None
        return cls._instance
    
    async def get_loop(self):
        """Get or create an event loop for the current context"""
        async with self._lock:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                self.loop = loop
                return loop
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                self.loop = loop
                return loop

class CustomMCPClient:
    """
    Custom client for connecting to MCP servers.
    Provides functionality for tool discovery and invocation.
    """
    
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url
        self.timeout = timeout
        self.tools_cache = None
        self.session = None
        self.health_checked = False
        self._loop_manager = EventLoopManager()
        self._init_lock = asyncio.Lock()
    
    async def _ensure_loop(self):
        """Ensure we have a valid event loop"""
        return await self._loop_manager.get_loop()
    
    async def _ensure_session(self):
        """Ensure aiohttp session is initialized"""
        async with self._init_lock:
            if self.session is None or self.session.closed:
                await self._ensure_loop()
                self.session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                )
    
    async def _check_server_health(self) -> bool:
        """Check if the server is healthy"""
        if self.health_checked:
            return True
        
        try:
            await self._ensure_session()
            async with self.session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    self.health_checked = True
                    return True
                return False
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False
    
    async def get_tools(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """Get available tools from the MCP server"""
        if not force_refresh and self.tools_cache is not None:
            return self.tools_cache
        
        await self._ensure_session()
        await self._ensure_loop()
        
        # First try /mcp/tools
        try:
            logger.info(f"Fetching tools from: {self.base_url}/mcp/tools")
            async with self.session.get(f"{self.base_url}/mcp/tools") as response:
                if response.status == 200:
                    self.tools_cache = await response.json()
                    return self.tools_cache
        except Exception as e:
            logger.warning(f"Error fetching tools from /mcp/tools: {str(e)}")
        
        # Fall back to /tools
        try:
            logger.info(f"Falling back to: {self.base_url}/tools")
            async with self.session.get(f"{self.base_url}/tools") as response:
                if response.status == 200:
                    self.tools_cache = await response.json()
                    return self.tools_cache
        except Exception as e:
            logger.error(f"Error getting tools: {str(e)}")
        
        return []
    
    async def invoke(self, tool_name: str, parameters: Dict[str, Any] = None) -> Any:
        """Invoke a tool on the MCP server"""
        if not await self._check_server_health():
            raise Exception("Server is not healthy")
        
        await self._ensure_session()
        await self._ensure_loop()
        
        # Prepare request payload
        payload = {
            "name": tool_name,
            "parameters": parameters or {}
        }
        
        # First try /mcp/tools/{tool_name}/invoke
        try:
            invoke_url = f"{self.base_url}/mcp/tools/{tool_name}/invoke"
            logger.info(f"Invoking tool at {invoke_url}")
            
            async with self.session.post(invoke_url, json=payload) as response:
                if response.status == 200:
                    return await response.json()
        except Exception as e:
            logger.warning(f"Error invoking tool at {invoke_url}: {str(e)}")
        
        # Fall back to /tools/{tool_name}/invoke
        try:
            invoke_url = f"{self.base_url}/tools/{tool_name}/invoke"
            logger.info(f"Falling back to: {invoke_url}")
            
            async with self.session.post(invoke_url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("content", result)
                
                error_text = await response.text()
                logger.error(f"Error invoking tool {tool_name}: {error_text}")
                return f"Error invoking tool {tool_name}: {error_text}"
        except Exception as e:
            logger.error(f"Error invoking tool {tool_name}: {str(e)}")
            return f"Error invoking tool {tool_name}: {str(e)}"
    
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
    """Synchronous client for MCP servers"""
    
    def __init__(self, base_url):
        self.base_url = base_url
        logger.info(f"Initialized MCPClient with URL: {base_url}")
        
    def invoke(self, tool_name, params=None):
        """Invoke a tool synchronously"""
        try:
            response = requests.post(
                f"{self.base_url}/invoke/{tool_name}",
                json=params or {},
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Request failed with status {response.status_code}: {response.text}")
                return {
                    "status": "error", 
                    "message": f"Request failed with status {response.status_code}"
                }
        except Exception as e:
            logger.error(f"Exception invoking MCP tool: {str(e)}")
            return {
                "status": "error",
                "message": f"Error invoking MCP tool: {str(e)}"
            }
    
    # Add an alias for compatibility
    invoke_sync = invoke


class SimpleMCPClient:
    """Simple client for interacting with MCP servers"""
    
    def __init__(self, base_url):
        import logging
        self.base_url = base_url
        self.logger = logging.getLogger("mcp_client")
        print(f"Initialized SimpleMCPClient with base_url: {base_url}")
    
    def invoke(self, tool_name, params=None):
        """Invoke an MCP tool on the server"""
        import requests
        try:
            # The correct endpoint structure for FastMCP
            url = f"{self.base_url}/invoke/{tool_name}"
            print(f"Invoking tool: {tool_name} at URL: {url}")
            
            response = requests.post(
                url,
                json=params or {},
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                error_msg = f"Request failed with status {response.status_code}: {response.text}"
                print(f"Error: {error_msg}")
                return {
                    "status": "error",
                    "message": error_msg
                }
        except Exception as e:
            print(f"Exception calling MCP: {str(e)}")
            return {
                "status": "error",
                "message": f"Error invoking MCP tool: {str(e)}"
            }
    
    # Add an alias for compatibility with different client implementations
    invoke_sync = invoke
