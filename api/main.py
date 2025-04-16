from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uuid
import sys
import os

# Add the root directory to the Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config
from agents.marketing_management_book.agent import MarketingManagementAgent
from contextlib import asynccontextmanager

# Import additional agents here as you add them
AGENT_REGISTRY = {
    "marketing_management": MarketingManagementAgent
}

# Store responses and agent instances in memory
response_store: Dict[str, Dict[str, Any]] = {}
agent_instances: Dict[str, MarketingManagementAgent] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app"""
    # Validate required configuration
    missing_config = Config.validate_config()
    if missing_config:
        print(f"WARNING: Missing required configuration: {', '.join(missing_config)}")
    yield
    # Cleanup on shutdown
    for session_id, agent in agent_instances.items():
        await agent.cleanup()
    agent_instances.clear()
    response_store.clear()

app = FastAPI(
    title="MarketScope AI API",
    lifespan=lifespan,
    servers=[{"url": f"http://localhost:{Config.API_PORT}"}]
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    agent_type: str = "marketing_management"
    model: Optional[str] = None
    config: Optional[Dict] = None
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    session_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

async def process_query(query: str, session_id: str, agent_type: str, model: Optional[str] = None, config: Optional[Dict] = None):
    """Background task to process queries"""
    try:
        agent_class = AGENT_REGISTRY.get(agent_type)
        if not agent_class:
            raise ValueError(f"Unknown agent type: {agent_type}")

        selected_model = model or Config.DEFAULT_MODEL
        if selected_model not in Config.get_available_models():
            raise ValueError(f"Model '{selected_model}' is not available")

        if not config:
            config = {}
        config["model"] = selected_model

        if session_id not in agent_instances:
            agent_instances[session_id] = agent_class(config)

        agent = agent_instances[session_id]
        result = await agent.process_query(query)

        response_store[session_id] = {
            "status": "completed",
            "result": {"output": result}
        }
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error processing query: {str(e)}\n{error_details}")
        response_store[session_id] = {
            "status": "error",
            "result": {"error": str(e)},
            "error": str(e)
        }
    finally:
        if session_id in agent_instances:
            await agent_instances[session_id].cleanup()
            del agent_instances[session_id]

@app.get("/agents")
async def list_agents():
    """List available agent types"""
    return {"agents": list(AGENT_REGISTRY.keys())}

@app.get("/models")
async def list_models():
    """List available models"""
    models = {}
    for model_name in Config.get_available_models():
        model_config = Config.get_model_config(model_name)
        models[model_name] = {
            "name": model_config.get("name"),
            "supports_images": model_config.get("supports_images", False),
            "provider": model_config.get("provider", "")
        }
    return {"models": models, "default_model": Config.DEFAULT_MODEL}

@app.post("/query", response_model=QueryResponse)
async def create_query(request: QueryRequest, background_tasks: BackgroundTasks):
    """Submit a new query to be processed."""
    session_id = request.session_id or str(uuid.uuid4())
    response_store[session_id] = {"status": "processing"}

    background_tasks.add_task(
        process_query,
        request.query,
        session_id,
        request.agent_type,
        request.model,
        request.config
    )

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
    try:
        test_agent = MarketingManagementAgent()
        await test_agent.setup()
        await test_agent.cleanup()
        agent_status = "healthy"
    except Exception as e:
        agent_status = f"error: {str(e)}"

    missing_config = Config.validate_config()
    config_status = "valid" if not missing_config else f"missing: {', '.join(missing_config)}"

    return {
        "status": "healthy",
        "version": "1.0.0",
        "agent_status": agent_status,
        "config_status": config_status
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=Config.API_PORT)