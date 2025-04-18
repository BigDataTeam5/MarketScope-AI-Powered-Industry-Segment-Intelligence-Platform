"""
Simplified API server for MarketScope platform
"""
import os
import sys
import logging
import argparse
import socket
from typing import Dict, Any, List, Optional, Union
import json
import asyncio
import uuid
import traceback
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger("api_server")

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    logger.info(f"Added {project_root} to Python path")

# Try to import Config
try:
    from config.config import Config
    api_port = Config.API_PORT
    logger.info(f"Using API port {api_port}")
except (ImportError, AttributeError):
    api_port = int(os.getenv("API_PORT", 8003))
    logger.info(f"Config not found, using API port from env or default: {api_port}")

# Set up dummy unified agent for simplified implementation
class DummyUnifiedAgent:
    async def register_all_mcp_servers(self):
        logger.info("Using dummy unified agent")
        return True
        
    async def process_query(self, query, use_case=None, segment=None, context=None):
        return {
            "status": "success",
            "response": f"This is a sample response for your query: {query}\n\nSegment: {segment}\nUse case: {use_case}"
        }
        
    async def process_csv_data(self, csv_data, segment=None, schema_name=None, table_name=None, query=None):
        return {
            "status": "success",
            "response": f"Processed CSV data for segment: {segment}\nSchema: {schema_name}\nTable: {table_name}"
        }

# Use dummy agent
unified_agent = DummyUnifiedAgent()

# Try importing real agent if available
try:
    from config.litellm_service import get_llm_model
    from agents.unified_agent import unified_agent
    logger.info("Successfully imported real unified_agent")
except Exception as e:
    logger.warning(f"Using dummy agent because of import error: {str(e)}")
    # Continue with dummy agent

# Create FastAPI app
app = FastAPI(
    title="MarketScope API",
    description="API for MarketScope AI-Powered Industry Segment Intelligence Platform",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store responses with lock for thread safety
response_store: Dict[str, Any] = {}
response_lock = asyncio.Lock()

class QueryRequest(BaseModel):
    query: str = Field(..., description="The query to process")
    segment: Optional[str] = Field(None, description="Segment to analyze")
    options: Optional[Dict[str, Any]] = Field(None, description="Additional options")

class QueryResponse(BaseModel):
    status: str
    response: str
    details: Optional[Dict[str, Any]] = None

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a query using the unified agent"""
    try:
        logger.info(f"Processing query: {request.query}")
        
        # Process query with unified agent
        result = await unified_agent.process_query(
            query=request.query,
            segment=request.segment,
            **(request.options or {})
        )
        
        # Return response
        return {
            "status": "success",
            "response": result.get("response", "No response"),
            "details": result
        }
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/query/{session_id}")
async def get_query_result(session_id: str):
    """Get the results of a previously submitted query"""
    async with response_lock:
        if session_id not in response_store:
            raise HTTPException(status_code=404, detail="Query not found")
        return response_store[session_id]

# Handle CSV data upload and processing
class CSVProcessRequest(BaseModel):
    csv_data: str
    segment: Optional[str] = None
    table_name: Optional[str] = None
    query: Optional[str] = None

@app.post("/process_csv")
async def process_csv_data(request: CSVProcessRequest, background_tasks: BackgroundTasks):
    """Process CSV data through the unified agent"""
    session_id = str(uuid.uuid4())
    async with response_lock:
        response_store[session_id] = {"status": "processing"}
    
    background_tasks.add_task(
        process_csv_task,
        session_id,
        request.csv_data,
        request.segment,
        request.table_name,
        request.query
    )
    
    return {"session_id": session_id, "status": "processing"}

async def process_csv_task(session_id: str, csv_data: str, segment: Optional[str], table_name: Optional[str], query: Optional[str]):
    """Background task to process CSV data"""
    try:
        logger.info(f"Processing CSV data (segment: {segment})")
        
        # Process CSV data through unified agent
        result = await unified_agent.process_csv_data(
            csv_data=csv_data,
            segment=segment,
            table_name=table_name,
            query=query
        )
        
        # Update response store with result
        async with response_lock:
            response_store[session_id] = {
                "status": "completed",
                "result": result,
                "segment": segment
            }
            
    except Exception as e:
        logger.error(f"Error processing CSV data: {str(e)}")
        logger.error(traceback.format_exc())
        async with response_lock:
            response_store[session_id] = {
                "error": str(e),
                "status": "error",
                "segment": segment
            }

# Get available segments
@app.get("/segments")
async def get_segments():
    """Get available segments"""
    segments = [
        {
            "id": "skin_care",
            "name": "Skin Care Segment",
            "description": "Skincare products and treatments"
        },
        {
            "id": "diagnostic",
            "name": "Diagnostic Segment",
            "description": "Medical diagnostic tools and equipment"
        },
        {
            "id": "supplement",
            "name": "Supplement Segment",
            "description": "Dietary supplements and nutritional products"
        },
        {
            "id": "pharmaceutical",
            "name": "Otc Pharmaceutical Segment",
            "description": "Over-the-counter pharmaceutical products"
        },
        {
            "id": "fitness",
            "name": "Fitness Wearable Segment",
            "description": "Wearable fitness and health tracking devices"
        }
    ]
    
    return {"segments": segments}

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

# Debug endpoint
@app.get("/debug")
async def debug_info():
    """Debug endpoint to check configuration"""
    return {
        "api_port": api_port,
        "python_path": sys.path,
        "agent_type": type(unified_agent).__name__
    }

def is_port_available(port: int) -> bool:
    """Check if a port is available"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("0.0.0.0", port))
            return True
        except:
            return False

def find_available_port(start_port: int, max_attempts: int = 10) -> int:
    """Find an available port starting from start_port"""
    for port_offset in range(max_attempts):
        port = start_port + port_offset
        if is_port_available(port):
            return port
    raise RuntimeError(f"Could not find an available port after {max_attempts} attempts")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Run MarketScope API Server")
    parser.add_argument("--port", type=int, default=api_port, help=f"Port to run on (default: {api_port})")
    
    args = parser.parse_args()
    requested_port = args.port
    
    # Check if the requested port is available, if not find one that is
    if not is_port_available(requested_port):
        logger.warning(f"Port {requested_port} is not available, finding another one...")
        try:
            port = find_available_port(requested_port + 1)
            logger.info(f"Found available port: {port}")
        except RuntimeError as e:
            logger.error(f"Error finding available port: {str(e)}")
            sys.exit(1)
    else:
        port = requested_port
        
    # Update API_PORT in environment
    os.environ["API_PORT"] = str(port)
    
    print("\n==================================================")
    print(f"  Starting MarketScope API Server on port {port}")
    print("==================================================\n")
    
    # Import and run uvicorn
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
