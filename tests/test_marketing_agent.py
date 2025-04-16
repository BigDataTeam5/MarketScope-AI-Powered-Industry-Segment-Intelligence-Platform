# tests/test_marketing_agent.py
import pytest
from unittest.mock import AsyncMock
from agents.marketing_management_book.agent import MarketingManagementAgent

# Simple mock data
MOCK_CONTENT = {
    "chunk1": "Healthcare market segmentation involves demographic and behavioral factors.",
    "chunk2": "Key marketing strategies for healthcare products include targeting and positioning."
}

@pytest.mark.asyncio
async def test_marketing_agent():
    """Basic test for marketing agent functionality"""
    
    # Mock MCP tools
    mock_search = AsyncMock(return_value=["chunk1", "chunk2"])
    mock_fetch = AsyncMock(side_effect=lambda x: MOCK_CONTENT.get(x, ""))
    
    # Mock MCP session
    mock_session = AsyncMock()
    mock_session.list_tools.return_value = [
        AsyncMock(name="pinecone_search", run=mock_search),
        AsyncMock(name="fetch_s3_chunk", run=mock_fetch)
    ]
    
    # Mock MCP service
    mock_mcp = AsyncMock()
    mock_mcp.session = mock_session
    mock_mcp.connect.return_value = True
    
    # Create agent
    agent = MarketingManagementAgent()
    agent._mcp_service = mock_mcp  # Inject mock service
    
    # Test query processing
    response = await agent.process_query("How to segment healthcare market?")
    
    # Basic assertions
    assert response is not None
    assert isinstance(response, str)
    assert mock_search.called
    assert mock_fetch.called

@pytest.mark.asyncio
async def test_error_handling():
    """Test basic error handling"""
    agent = MarketingManagementAgent()
    agent._mcp_service = AsyncMock(connect=AsyncMock(return_value=False))
    
    response = await agent.process_query("test")
    assert "Error" in response