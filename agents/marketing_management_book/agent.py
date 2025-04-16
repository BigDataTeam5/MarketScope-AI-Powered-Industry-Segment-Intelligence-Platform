"""
Marketing Management Agent using LangGraph for healthcare market segmentation.
"""
from typing import Dict, Any, Optional
import sys
import os
from langgraph.prebuilt import create_react_agent
from mcp import ClientSession
from mcp.client.sse import sse_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langsmith import traceable

# Import unified config and services
from config import Config, litellm_service
from services.mcp_service import mcp_service

class MarketingManagementAgent:
    """Marketing Management Agent using LangGraph"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.agent = None
        self.tools = None

    @traceable(name="setup_agent", run_type="chain")
    async def setup(self):
        """Setup agent with tools"""
        try:
            # Connect to MCP server using the shared service
            if not await mcp_service.connect():
                return False
                
            # Load tools from MCP server
            self.tools = await load_mcp_tools(mcp_service.session)
            print(f"Loaded marketing tools: {[tool.name for tool in self.tools]}")
            
            # Create React agent with tools
            self.agent = create_react_agent(self.tools)
            return True

        except Exception as e:
            print(f"Error setting up agent: {str(e)}")
            return False

    @traceable(name="process_query", run_type="chain")
    async def process_query(self, query: str) -> str:
        """Process query using LangGraph agent"""
        try:
            # Setup agent if not ready
            if not self.agent and not await self.setup():
                return "Error: Could not initialize marketing agent"

            # System message for marketing context
            system_message = """You are an AI assistant specializing in healthcare market segmentation.
            
            You have access to tools that can search Philip Kotler's Marketing Management book and industry reports.
            
            Follow these steps when answering questions:
            1. Use pinecone_search to find relevant chunks
            2. For each chunk_id returned, use fetch_s3_chunk to retrieve the content
            3. Review all retrieved content before formulating your answer
            4. For segment analysis, use analyze_market_segment tool
            5. For strategy generation, use generate_segment_strategy tool
            
            Always cite specific information from the retrieved chunks when possible."""

            # Get model configuration from settings
            model_name = self.config.get("model", Config.DEFAULT_MODEL)
            temperature = self.config.get("temperature", Config.DEFAULT_TEMPERATURE)

            # Use the unified llm_service
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": query}
            ]
            
            return await litellm_service.chat_completion(
                messages=messages,
                model_name=model_name,
                temperature=temperature
            )

        except Exception as e:
            return f"Error processing query: {str(e)}"
        
    async def cleanup(self):
        """Cleanup resources"""
        await mcp_service.cleanup()
        self.tools = None
        self.agent = None

# Create a singleton instance for easy access
marketing_agent = MarketingManagementAgent()