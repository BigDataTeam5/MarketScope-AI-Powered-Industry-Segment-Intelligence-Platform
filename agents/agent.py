import asyncio
from mcp import ClientSession
from mcp.client.sse import sse_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

async def main():
    # Connect to the LLM
    model = ChatOpenAI(model="gpt-4o", temperature=0)
    
    try:
        # Connect to running SSE server
        client = sse_client("http://localhost:8000/sse")
        read_stream, write_stream = await client.__aenter__()
        
        try:
            session = ClientSession(read_stream, write_stream)
            await session.__aenter__()
            
            try:
                # Initialize the session
                await session.initialize()
                
                # Load tools and create agent
                tools = await load_mcp_tools(session)
                agent = create_react_agent(model, tools)
                
                # Execute query
                query = "What does Kotler suggest about segmentation strategies for pharma?"
                response = await agent.ainvoke({"messages": query})
                print("Agent response:\n", response)
            finally:
                await session.__aexit__(None, None, None)
        finally:
            await client.__aexit__(None, None, None)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())