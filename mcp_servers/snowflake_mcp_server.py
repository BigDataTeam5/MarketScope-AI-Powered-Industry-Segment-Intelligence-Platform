"""
Snowflake MCP Server
Provides tools for interacting with Snowflake database
"""
import pandas as pd
import json
import os
import io
import logging
from fastapi import FastAPI
from mcp.server.fastmcp import FastMCP
import uvicorn

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("snowflake_mcp_server")

# Import configuration
from config.config import Config

# Create MCP server first
# Use run() method directly instead of custom mounting
mcp_server = FastMCP("snowflake")

# Register MCP tools
@mcp_server.tool()
def execute_query(query: str) -> str:
    """Execute a SQL query on Snowflake database"""
    try:
        # Mock implementation - in a real scenario, this would connect to Snowflake
        logger.info(f"Executing query: {query}")
        
        # Parse the query to determine what to return
        if query.lower().startswith("select"):
            # For SELECT queries, return mock data based on the query
            if "product_name" in query.lower():
                # Mock product data
                data = [
                    {"PRODUCT_NAME": "HeartGuard Monitor"},
                    {"PRODUCT_NAME": "DiabeCare Sensor"},
                    {"PRODUCT_NAME": "PainEase Gel"},
                    {"PRODUCT_NAME": "Vitamin Complex"},
                    {"PRODUCT_NAME": "PediCare Drops"}
                ]
                return f"5 rows. (Execution time: 0.5s)\n{json.dumps(data)}"
            else:
                # Generic mock data
                data = [
                    {"COLUMN1": "Value1", "COLUMN2": 123},
                    {"COLUMN1": "Value2", "COLUMN2": 456}
                ]
                return f"2 rows. (Execution time: 0.3s)\n{json.dumps(data)}"
        else:
            # For non-SELECT queries, return success message
            return f"Query executed successfully. (Execution time: 0.2s)"
    except Exception as e:
        logger.error(f"Error executing query: {str(e)}")
        return f"Error: {str(e)}"

@mcp_server.tool()
def load_csv_to_table(table_name: str, csv_data: str, create_table: bool = True) -> str:
    """Load CSV data into a Snowflake table"""
    try:
        logger.info(f"Loading data into table: {table_name}")
        
        # Parse CSV data
        df = pd.read_csv(io.StringIO(csv_data))
        row_count = len(df)
        
        # Mock implementation - in a real scenario, this would load data to Snowflake
        return f"Successfully loaded {row_count} rows into table {table_name}."
    except Exception as e:
        logger.error(f"Error loading CSV data: {str(e)}")
        return f"Error: {str(e)}"

@mcp_server.tool()
def get_table_schema(table_name: str) -> str:
    """Get schema information for a Snowflake table"""
    try:
        logger.info(f"Getting schema for table: {table_name}")
        
        # Mock implementation - return fake schema based on table name
        if "sales" in table_name.lower():
            schema = [
                {"COLUMN_NAME": "DATE", "DATA_TYPE": "DATE"},
                {"COLUMN_NAME": "PRODUCT_NAME", "DATA_TYPE": "VARCHAR"},
                {"COLUMN_NAME": "PRICE", "DATA_TYPE": "NUMBER"},
                {"COLUMN_NAME": "UNITS_SOLD", "DATA_TYPE": "NUMBER"},
                {"COLUMN_NAME": "REVENUE", "DATA_TYPE": "NUMBER"}
            ]
        else:
            schema = [
                {"COLUMN_NAME": "ID", "DATA_TYPE": "NUMBER"},
                {"COLUMN_NAME": "NAME", "DATA_TYPE": "VARCHAR"},
                {"COLUMN_NAME": "VALUE", "DATA_TYPE": "NUMBER"}
            ]
        
        return f"Table {table_name} Schema:\n{json.dumps(schema)}"
    except Exception as e:
        logger.error(f"Error getting table schema: {str(e)}")
        return f"Error: {str(e)}"

# Run the server
if __name__ == "__main__":
    port = Config.MCP_PORT  # Use port from config
    logger.info(f"Starting Snowflake MCP Server on port {port}")
    
    # Critical change: Instead of custom FastAPI mounting, use built-in run method
    # This ensures all MCP endpoints are correctly set up
    mcp_server.run(
        transport="sse",
        sse={
            "host": "0.0.0.0",
            "port": port
        }
    )
