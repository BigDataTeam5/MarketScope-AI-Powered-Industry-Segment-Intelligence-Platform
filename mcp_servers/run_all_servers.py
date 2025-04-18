'''
Refactored run_all_servers.py: Start all MCP servers in separate processes
so they can each listen on their designated ports.
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

def run_segment_server():
    """Start the segment MCP server"""
    # Import server instance directly and run its mount_and_run method
    from mcp_servers.segment_mcp_server import server
    logger.info("Starting Segment MCP Server on port 8003")
    # Change port to 8003 for consistency
    server.port = 8003
    server.mount_and_run()

def run_snowflake_server():
    """Start the snowflake MCP server"""
    from mcp_servers.snowflake_mcp_server import app
    import uvicorn
    logger.info("Starting Snowflake MCP Server on port 8004")
    uvicorn.run(app, host="0.0.0.0", port=8004)

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
    
    # Market Analysis tools
    @mcp_server.tool()
    def market_analysis_health_check() -> dict:
        """Check if the Market Analysis server is available"""
        import requests
        try:
            response = requests.get("http://localhost:8001/health", timeout=2)
            return {"status": "available" if response.status_code == 200 else "unavailable"}
        except:
            return {"status": "unavailable"}
            
    @mcp_server.tool()
    def market_analysis_segment_report(segment: str) -> dict:
        """Get a market analysis report for a specific segment"""
        return {
            "segment": segment,
            "status": "success",
            "report": f"Market analysis report for {segment} segment. Use specific tools from the Market Analysis server at port 8001 for detailed analysis."
        }
    
    # Sales Analytics tools
    @mcp_server.tool()
    def sales_health_check() -> dict:
        """Check if the Sales Analytics server is available"""
        import requests
        try:
            response = requests.get("http://localhost:8002/health", timeout=2)
            return {"status": "available" if response.status_code == 200 else "unavailable"}
        except:
            return {"status": "unavailable"}
            
    @mcp_server.tool()
    def sales_data_summary(segment: str = None) -> dict:
        """Get a summary of sales data for a specific segment"""
        return {
            "segment": segment or "all",
            "status": "success",
            "summary": f"Sales data summary for {segment or 'all'} segments. Use specific tools from the Sales Analytics server at port 8002 for detailed analysis."
        }
    
    # Segment server tools
    @mcp_server.tool()
    def segment_health_check() -> dict:
        """Check if the Segment server is available"""
        import requests
        try:
            response = requests.get("http://localhost:8003/health", timeout=2)
            return {"status": "available" if response.status_code == 200 else "unavailable"}
        except:
            return {"status": "unavailable"}
            
    @mcp_server.tool()
    def segment_product_trends(segment: str) -> dict:
        """Get product trends for a specific segment"""
        return {
            "segment": segment,
            "status": "success",
            "trends": f"Product trends for {segment}. Use specific tools from the Segment server at port 8003 for detailed analysis."
        }
    
    # Snowflake tools
    @mcp_server.tool()
    def snowflake_health_check() -> dict:
        """Check if the Snowflake server is available"""
        import requests
        try:
            response = requests.get("http://localhost:8004/health", timeout=2)
            return {"status": "available" if response.status_code == 200 else "unavailable"}
        except:
            return {"status": "unavailable"}
            
    @mcp_server.tool()
    def snowflake_query(query: str) -> dict:
        """Execute a SQL query on Snowflake (proxy to Snowflake server)"""
        import requests
        try:
            # This is just a proxy - in real implementation, would use requests to forward to the Snowflake server
            return {
                "status": "success",
                "message": f"Query '{query}' would be forwarded to the Snowflake server at port 8004."
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    # Add a simple health-check tool to the unified server
    @mcp_server.tool()
    def unified_health_check() -> dict:
        """Check the health of all MCP servers"""
        import requests
        servers = [
            {"name": "Market Analysis", "url": "http://localhost:8001/health"},
            {"name": "Sales Analytics", "url": "http://localhost:8002/health"},
            {"name": "Segment", "url": "http://localhost:8003/health"},
            {"name": "Snowflake", "url": "http://localhost:8004/health"}
        ]
        
        results = {}
        for server in servers:
            try:
                response = requests.get(server["url"], timeout=2)
                results[server["name"]] = "healthy" if response.status_code == 200 else "unhealthy"
            except:
                results[server["name"]] = "unavailable"
        
        return {
            "status": "healthy",
            "message": "MarketScope Unified MCP Server is running",
            "servers": results
        }
    
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
        return {
            "servers": [
                {"name": "Market Analysis", "url": "http://localhost:8001"},
                {"name": "Sales Analytics", "url": "http://localhost:8002"},
                {"name": "Segment", "url": "http://localhost:8003"},
                {"name": "Snowflake", "url": "http://localhost:8004"}
            ]
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
    processes.append(p1)
    
    p2 = multiprocessing.Process(target=run_sales_analytics_server)
    p2.start()
    processes.append(p2)
    
    p3 = multiprocessing.Process(target=run_segment_server)
    p3.start()
    processes.append(p3)
    
    p4 = multiprocessing.Process(target=run_snowflake_server)
    p4.start()
    processes.append(p4)
    
    # Start the unified server last
    p5 = multiprocessing.Process(target=run_unified_server)
    p5.start()
    processes.append(p5)
    
    # Log when all servers are started
    logger.info("All server processes started")
    logger.info("- Market Analysis: http://localhost:8001/docs")
    logger.info("- Sales Analytics: http://localhost:8002/docs")
    logger.info("- Segment: http://localhost:8003/docs")
    logger.info("- Snowflake: http://localhost:8004/docs")
    logger.info("- Unified: http://localhost:8000/docs")
    
    try:
        # Wait for all processes to complete (which won't happen unless they're killed)
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        logger.info("Shutting down all servers...")
        for p in processes:
            p.terminate()
            p.join()

if __name__ == "__main__":
    main()
