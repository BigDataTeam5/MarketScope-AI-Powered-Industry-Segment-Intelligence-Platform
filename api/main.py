from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uuid
import asyncio
from agents.marketing_management_book.agent import marketing_agent

app = FastAPI(title="MarketScope AI")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store responses with lock for thread safety
response_store: Dict[str, Any] = {}
response_lock = asyncio.Lock()

class QueryRequest(BaseModel):
    query: str
    segment: Optional[str] = None
    model: str = "gpt-4"  # Default to GPT-4
    agent_type: str = "marketing_management"

@app.post("/bookquery")
async def create_query(request: QueryRequest, background_tasks: BackgroundTasks):
    session_id = str(uuid.uuid4())
    async with response_lock:
        response_store[session_id] = {"status": "processing"}
    
    background_tasks.add_task(
        process_query,
        session_id,
        request.query,
        request.segment,
        request.model
    )
    
    return {"session_id": session_id, "status": "processing"}

@app.get("/bookquery/{session_id}")
async def get_query_result(session_id: str):
    async with response_lock:
        if session_id not in response_store:
            raise HTTPException(status_code=404, detail="Query not found")
        return response_store[session_id]

async def process_query(session_id: str, query: str, segment: Optional[str], model: str):
    """Process a query using the marketing agent"""
    try:
        print(f"Processing query for session {session_id}")
        
        # Process query through marketing agent
        result = await marketing_agent.process_query(query, segment)
        
        # Update response store with result
        async with response_lock:
            response_store[session_id] = {
                "status": "completed",
                "result": result,
                "segment": segment
            }
            print(f"Updated response store for session {session_id}: {response_store[session_id]}")
            
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        async with response_lock:
            response_store[session_id] = {
                "error": str(e),
                "status": "error",
                "segment": segment
            }
        print(f"Updated response store for session {session_id}: {response_store[session_id]}")

# Add startup event to initialize agent
@app.on_event("startup")
async def startup_event():
    """Initialize the marketing agent on startup"""
    try:
        await marketing_agent.setup()
        print("Marketing agent initialized successfully")
    except Exception as e:
        print(f"Error initializing marketing agent: {str(e)}")

# Add this route to your FastAPI app

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}