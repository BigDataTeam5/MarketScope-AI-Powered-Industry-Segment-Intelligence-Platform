"""
Reusable MCP Service client for MarketScope AI agents.
"""
from mcp import ClientSession

class MCPService:
    def __init__(self):
        self.session = None

    async def connect(self):
        try:
            self.session = await ClientSession().connect()
            return True
        except Exception as e:
            print(f"Error connecting to MCP: {e}")
            return False

    async def cleanup(self):
        if self.session:
            await self.session.close()
            self.session = None

# Singleton instance for import
mcp_service = MCPService()