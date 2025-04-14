import asyncio
import uuid
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import os
from mcp.client.sse import sse_client
from config import setup_langsmith, MCP_SERVER_URL
# Change the import to use a relative import
setup_langsmith()
from agent_service import agent_service  

app = FastAPI(title="MarketScope AI API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store responses in memory for simple persistence
response_store: Dict[str, Dict[str, Any]] = {}

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    direct: Optional[bool] = False  # Whether to use direct OpenAI query (for testing)

class QueryResponse(BaseModel):
    session_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

async def process_query(query: str, session_id: str, direct: bool = False):
    """Background task to process queries with the agent service."""
    try:
        # Choose which method to use
        if direct:
            # Direct OpenAI query for testing
            result = await agent_service.direct_openai_query(query)
            # For direct queries, just use the text directly
            formatted_result = result
        else:
            # Full agent pipeline
            result = await agent_service.process_query(query)
            print(f"Raw agent result: {result}")
            
            # Try to extract the text content directly
            formatted_result = result
        
        # Store successful result with simplified format
        response_store[session_id] = {
            "status": "completed",
            "result": formatted_result
        }
        print(f"Stored result for session {session_id}: {formatted_result}")
    except Exception as e:
        # Store error result
        import traceback
        error_details = traceback.format_exc()
        print(f"Error processing query: {str(e)}\n{error_details}")
        response_store[session_id] = {
            "status": "error",
            "error": str(e)
        }

@app.post("/query", response_model=QueryResponse)
async def create_query(request: QueryRequest, background_tasks: BackgroundTasks):
    """Submit a new query to be processed."""
    # Generate session ID if not provided
    session_id = request.session_id or str(uuid.uuid4())
    
    # Create initial entry in the response store
    response_store[session_id] = {"status": "processing"}
    
    # Process query in the background
    background_tasks.add_task(process_query, request.query, session_id, request.direct)
    
    return {"session_id": session_id, "status": "processing"}

@app.get("/query/{session_id}", response_model=QueryResponse)
async def get_query_result(session_id: str):
    """Get the result of a previously submitted query."""
    if session_id not in response_store:
        raise HTTPException(status_code=404, detail="Session not found")
    
    result = response_store[session_id]
    return {"session_id": session_id, **result}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    # Check LangSmith configuration
    langsmith_config = {
        "LANGSMITH_TRACING": os.getenv("LANGSMITH_TRACING"),
        "LANGSMITH_PROJECT": os.getenv("LANGSMITH_PROJECT"),
        "LANGSMITH_API_KEY_SET": bool(os.getenv("LANGSMITH_API_KEY")),
        "LANGSMITH_ENDPOINT": os.getenv("LANGSMITH_ENDPOINT")
    }
    
    # Check MCP server connection
    mcp_status = "Unknown"
    try:
        # Try to connect to the MCP server
        client = sse_client(MCP_SERVER_URL)
        read_stream, write_stream = await client.__aenter__()
        await client.__aexit__(None, None, None)
        mcp_status = "Connected"
    except Exception as e:
        mcp_status = f"Error: {str(e)}"
    
    return {
        "status": "healthy",
        "version": "1.0.0",
        "langsmith_config": langsmith_config,
        "mcp_server": mcp_status,
        "environment": os.environ.get("ENVIRONMENT", "development")
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)