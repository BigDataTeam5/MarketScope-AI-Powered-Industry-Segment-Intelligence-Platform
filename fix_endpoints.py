"""
Script to add direct endpoints for Snowflake tools to ensure compatibility
"""
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fix_endpoints")

# Create FastAPI app
app = FastAPI(title="Snowflake API Compatibility Layer")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
SNOWFLAKE_MCP_URL = "http://localhost:8004"

@app.get("/tools")
async def get_tools():
    """Proxy for /mcp/tools endpoint"""
    try:
        # Forward to the real endpoint
        response = requests.get(f"{SNOWFLAKE_MCP_URL}/mcp/tools")
        if response.status_code == 200:
            return response.json()
        return {"detail": f"Error from upstream server: {response.text}"}
    except Exception as e:
        logger.error(f"Error proxying tools request: {str(e)}")
        return {"detail": f"Error: {str(e)}"}

@app.post("/tools/{tool_name}/invoke")
async def invoke_tool(tool_name: str, parameters: dict):
    """Proxy for /mcp/tools/{tool_name}/invoke endpoint"""
    try:
        # Forward to the real endpoint
        response = requests.post(
            f"{SNOWFLAKE_MCP_URL}/mcp/tools/{tool_name}/invoke",
            json=parameters
        )
        if response.status_code == 200:
            return response.json()
        return {"detail": f"Error from upstream server: {response.text}"}
    except Exception as e:
        logger.error(f"Error proxying invoke request: {str(e)}")
        return {"detail": f"Error: {str(e)}"}

@app.get("/health")
async def health():
    """Health check endpoint"""
    try:
        # Forward to the real endpoint
        response = requests.get(f"{SNOWFLAKE_MCP_URL}/health")
        if response.status_code == 200:
            return response.json()
        return {"status": "unhealthy", "detail": response.text}
    except Exception as e:
        logger.error(f"Error checking health: {str(e)}")
        return {"status": "unhealthy", "detail": str(e)}

if __name__ == "__main__":
    # Run the server on port 8005
    port = 8005
    logger.info(f"Starting compatibility server on port {port}")
    logger.info(f"Forwarding requests to {SNOWFLAKE_MCP_URL}")
    uvicorn.run(app, host="0.0.0.0", port=port)
