'''
Enhanced run_all_servers.py: Start all MCP servers in separate processes
with support for multiple segment-specific servers covering different industry segments.
'''
import logging
import sys
import os
import multiprocessing
import time

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("run_all_servers")

def run_market_analysis_server():
    """Start the market analysis MCP server"""
    from mcp_servers.market_analysis_mcp_server import app
    import uvicorn
    logger.info("Starting Market Analysis MCP Server on port 8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)

def run_sales_analytics_server():
    """Start the sales analytics MCP server"""
    from mcp_servers.sales_analytics_mcp_server import app
    import uvicorn
    logger.info("Starting Sales Analytics MCP Server on port 8002")
    uvicorn.run(app, host="0.0.0.0", port=8002)

def run_legacy_segment_server():
    """Start the legacy segment MCP server"""
    # Import server instance directly and run its mount_and_run method
    from mcp_servers.segment_mcp_server import server
    logger.info("Starting Legacy Segment MCP Server on port 8003")
    # Change port to 8003 for consistency
    server.port = 8003
    server.mount_and_run()

def run_snowflake_server():
    """Start the snowflake MCP server"""
    from mcp_servers.snowflake_mcp_server import app
    import uvicorn
    logger.info("Starting Snowflake MCP Server on port 8004")
    uvicorn.run(app, host="0.0.0.0", port=8004)

def run_specific_segment_server(segment_name):
    """Run a specific segment MCP server based on the provided segment name"""
    # Import the run_segment_server function explicitly
    from mcp_servers.segment_mcp_server import run_segment_server
    # Call the function with the segment name
    run_segment_server(segment_name)

def run_unified_server():
    """Start the unified MCP server with registered tools from all other servers"""
    from agents.unified_agent import unified_agent
    from fastapi import FastAPI
    from mcp.server.fastmcp import FastMCP
    import uvicorn
    
    # Create FastAPI app and MCP server
    app = FastAPI(title="MarketScope Unified MCP Server")
    mcp_server = FastMCP("marketscope")
    
    # Register basic tools directly in the unified server
    # [... existing code for tool registration ...]
    
    # Mount the MCP server to the FastAPI app at the /mcp path
    app.mount("/mcp", mcp_server.sse_app())
    
    # Add direct API endpoints
    @app.get("/")
    async def root():
        """Root endpoint"""
        return {
            "status": "healthy",
            "message": "MarketScope Unified MCP Server is running"
        }
        
    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "message": "MarketScope Unified MCP Server is running"
        }
    
    @app.get("/servers")
    async def list_servers():
        """List all connected MCP servers"""
        segment_servers = []
        for segment, config in Config.SEGMENT_CONFIG.items():
            port = config.get("port", 8014)  # Default to 8014 if not specified
            segment_servers.append({
                "name": f"Segment - {segment}", 
                "url": f"http://localhost:{port}"
            })
            
        return {
            "servers": [
                {"name": "Market Analysis", "url": "http://localhost:8001"},
                {"name": "Sales Analytics", "url": "http://localhost:8002"},
                {"name": "Legacy Segment", "url": "http://localhost:8003"},
                {"name": "Snowflake", "url": "http://localhost:8004"},
                {"name": "Unified", "url": f"http://localhost:{Config.MCP_PORT}"}
            ] + segment_servers
        }
        
    @app.get("/segments")
    async def get_segments():
        """Get available segments"""
        return {
            "segments": list(Config.SEGMENT_CONFIG.keys())
        }
    
    logger.info(f"Starting unified MCP Server on port {Config.MCP_PORT}")
    uvicorn.run(app, host="0.0.0.0", port=Config.MCP_PORT)

def main():
    """Start all MCP servers in separate processes"""
    logger.info("Starting all MCP servers...")
    
    # Create and start processes for each server
    processes = []
    
    # Start the individual servers
    p1 = multiprocessing.Process(target=run_market_analysis_server)
    p1.start()
    processes.append(("Market Analysis", p1))
    time.sleep(2)  # Add small delay between server starts
    
    p2 = multiprocessing.Process(target=run_sales_analytics_server)
    p2.start()
    processes.append(("Sales Analytics", p2))
    time.sleep(2)
    
    p3 = multiprocessing.Process(target=run_legacy_segment_server)
    p3.start()
    processes.append(("Legacy Segment", p3))
    time.sleep(2)
    
    p4 = multiprocessing.Process(target=run_snowflake_server)
    p4.start()
    processes.append(("Snowflake", p4))
    time.sleep(2)
    
    # Start all segment-specific servers
    segments = [
        "Skin Care Segment",
        "Healthcare - Diagnostic",
        "Pharmaceutical", 
        "Supplements",
        "Wearables"
    ]
    
    segment_processes = []
    for segment in segments:
        if segment not in Config.SEGMENT_CONFIG:
            logger.warning(f"Segment {segment} not found in Config.SEGMENT_CONFIG, skipping")
            continue
            
        port = Config.SEGMENT_CONFIG[segment].get("port", 8014)  # Default to 8014 if no port specified
        p = multiprocessing.Process(target=run_specific_segment_server, args=(segment,))
        p.start()
        segment_processes.append((f"Segment - {segment}", p))
        processes.append((f"Segment - {segment}", p))
        logger.info(f"Started {segment} MCP Server on port {port}")
        time.sleep(2)  # Add small delay between segment server starts
    
    # Start the unified server last
    p5 = multiprocessing.Process(target=run_unified_server)
    p5.start()
    processes.append(("Unified", p5))
    
    # Log when all servers are started
    logger.info("All server processes started")
    logger.info("- Market Analysis: http://localhost:8001")
    logger.info("- Sales Analytics: http://localhost:8002")
    logger.info("- Legacy Segment: http://localhost:8003")
    logger.info("- Snowflake: http://localhost:8004")
    logger.info("- Unified: http://localhost:8000")
    
    # Log all segment servers
    for segment in segments:
        if segment in Config.SEGMENT_CONFIG:
            port = Config.SEGMENT_CONFIG[segment].get("port", 8014)
            logger.info(f"- {segment}: http://localhost:{port}")
    
    try:
        # Wait for all processes to complete (which won't happen unless they're killed)
        for name, p in processes:
            p.join()
    except KeyboardInterrupt:
        logger.info("Shutting down all servers...")
        for name, p in processes:
            logger.info(f"Terminating {name} server...")
            p.terminate()
            p.join()
        logger.info("All servers shut down")

if __name__ == "__main__":
    main()
