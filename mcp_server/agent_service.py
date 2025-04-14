import asyncio
import os
from mcp import ClientSession
from mcp.client.sse import sse_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from config import OPENAI_API_KEY, MCP_SERVER_URL, setup_langsmith

# Replace current LangSmith setup with:
setup_langsmith()

# Import LangSmith tracing utilities
from langsmith.wrappers import wrap_openai
from langsmith import traceable
import openai

# Create wrapped OpenAI client
openai_client = wrap_openai(openai.Client())

class AgentService:
    """Service for handling agent queries to the MCP server."""
    
    def __init__(self, server_url="http://localhost:8000/sse"):
        self.server_url = server_url
        
    @traceable(name="process_query", run_type="chain")
    async def process_query(self, query: str):
        """Process a query through the agent system."""
        # Connect to the LLM with LangSmith tracing
        model = ChatOpenAI(model="gpt-4o", temperature=0)
        
        # Connect to running SSE server
        client = sse_client(self.server_url)
        read_stream, write_stream = await client.__aenter__()
        
        try:
            session = ClientSession(read_stream, write_stream)
            await session.__aenter__()
            
            try:
                # Initialize the session
                await session.initialize()
                
                # Load tools from the MCP server
                tools = await load_mcp_tools(session)
                print(f"Loaded tools: {[tool.name for tool in tools]}")

                # Get the system message
                system_message = """You are an AI assistant specializing in pharmaceutical and healthcare market segmentation.

                You have access to tools that can search a knowledge base containing marketing textbooks and industry reports.

                Follow these steps when answering questions:
                1. First use pinecone_search to find relevant chunks
                2. For each chunk_id returned, use fetch_s3_chunk to retrieve the content
                3. Review all retrieved content before formulating your answer
                4. If the tools return errors or no useful information, rely on your general knowledge but mention the limitations

                Always cite specific information from the retrieved chunks when possible."""

                # Create agent without using system_message parameter
                agent = create_react_agent(model, tools)

                # Execute query with system message in the messages list
                response = await agent.ainvoke({
                    "messages": [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": query}
                    ]
                })
                return response
            finally:
                await session.__aexit__(None, None, None)
        finally:
            await client.__aexit__(None, None, None)
    
    @traceable(name="direct_openai_query", run_type="llm")
    async def direct_openai_query(self, query: str):
        """Make a direct OpenAI query for testing LangSmith tracing."""
        result = openai_client.chat.completions.create(
            messages=[{"role": "user", "content": query}],
            model="gpt-4o"
        )
        return result.choices[0].message.content

# Singleton instance for reuse
agent_service = AgentService()