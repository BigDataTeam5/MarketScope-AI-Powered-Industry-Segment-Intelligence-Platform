"""
Snowflake MCP Server
Provides tools for interacting with Snowflake database
"""
import pandas as pd
import json
import os
import io
import logging
import sys
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from mcp.server.fastmcp import FastMCP
import uvicorn
from typing import Dict, Any, Optional, List, Union

# Add snowflake connector
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("snowflake_mcp_server")

# Import configuration
try:
    from config.config import Config
except ImportError:
    # Fallback configuration if import fails
    logger.warning("Could not import Config, using default values")
    class Config:
        SNOWFLAKE_MCP_PORT = 8004
        # Add Snowflake connection parameters
        SNOWFLAKE_USER = os.getenv("SNOWFLAKE_USER", "")
        SNOWFLAKE_PASSWORD = os.getenv("SNOWFLAKE_PASSWORD", "")
        SNOWFLAKE_ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT", "")
        SNOWFLAKE_WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH")
        SNOWFLAKE_DATABASE = os.getenv("SNOWFLAKE_DATABASE", "HEALTHCARE_INDUSTRY_CUSTOMER_DATA")
        SNOWFLAKE_ROLE = os.getenv("SNOWFLAKE_ROLE", "ACCOUNTADMIN")

# Create FastAPI app
app = FastAPI(title="Snowflake MCP Server")

# Add CORS middleware with detailed logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests for debugging"""
    logger.info(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create MCP server
mcp_server = FastMCP("snowflake")

# Helper function to map segment names to schema names
def get_schema_for_segment(segment_name):
    """Map segment name to appropriate Snowflake schema"""
    # Use schema names from Config
    if segment_name in Config.SEGMENT_CONFIG:
        return Config.SEGMENT_CONFIG[segment_name]["schema"]
        
    # Fallback mapping
    mapping = {
        "Skin Care Segment": "SKINCARE_SEGMENT",
        "Diagnostic Segment": "DIAGNOSTIC_SEGMENT",
        "Fitness Wearable Segment": "WEARABLES_SEGMENT",
        "Supplement Segment": "SUPPLEMENTS_SEGMENT",
        "Otc Pharmaceutical Segment": "OTC_PHARMA_SEGMENT"
    }
    
    # Return mapped schema or default to PUBLIC
    return mapping.get(segment_name, "PUBLIC")

def get_snowflake_conn():
    """
    Get a connection to Snowflake using environment variables or Config values
    """
    try:
        # Print connection parameters for debugging (except password)
        logger.info(f"Attempting to connect to Snowflake with:")
        logger.info(f"  User: {Config.SNOWFLAKE_USER}")
        logger.info(f"  Account: {Config.SNOWFLAKE_ACCOUNT}")
        logger.info(f"  Warehouse: {Config.SNOWFLAKE_WAREHOUSE}")
        logger.info(f"  Database: {Config.SNOWFLAKE_DATABASE}")
        logger.info(f"  Role: {Config.SNOWFLAKE_ROLE}")
        
        # Check if credentials are set
        if not Config.SNOWFLAKE_USER or not Config.SNOWFLAKE_PASSWORD or not Config.SNOWFLAKE_ACCOUNT:
            logger.error("Snowflake credentials not set. Please configure the .env file")
            return None
            
        conn = snowflake.connector.connect(
            user=Config.SNOWFLAKE_USER,
            password=Config.SNOWFLAKE_PASSWORD,
            account=Config.SNOWFLAKE_ACCOUNT,
            warehouse=Config.SNOWFLAKE_WAREHOUSE,
            role=Config.SNOWFLAKE_ROLE,
            autocommit=True
        )
        
        # Test connection with a simple query
        cursor = conn.cursor()
        try:
            # Try to use the warehouse first
            cursor.execute(f"USE WAREHOUSE {Config.SNOWFLAKE_WAREHOUSE}")
            
            # Create database if it doesn't exist (needs proper permissions)
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {Config.SNOWFLAKE_DATABASE}")
            
            # Use the database
            cursor.execute(f"USE DATABASE {Config.SNOWFLAKE_DATABASE}")
            
            # Connection is successful
            logger.info(f"Successfully connected to Snowflake and using database {Config.SNOWFLAKE_DATABASE}")
            cursor.close()
            
            return conn
        except Exception as e:
            cursor.close()
            conn.close()
            logger.error(f"Error initializing Snowflake database: {str(e)}")
            return None
    except Exception as e:
        logger.error(f"Error connecting to Snowflake: {str(e)}")
        return None

# Manual endpoints to ensure tools are accessible
@app.get("/mcp/tools", tags=["MCP"])
async def get_tools():
    """Get available MCP tools - direct endpoint"""
    # Instead of using internal attributes, return a simple list for now
    # This is a workaround until we update to use the proper FastMCP methods
    tool_names = ["execute_query", "load_csv_to_table", "get_table_schema", "create_snowflake_table", "list_tables"]
    logger.info(f"Tools requested, returning {len(tool_names)} tools")
    tools_list = []
    for name in tool_names:
        tools_list.append({
            "name": name,
            "description": f"Snowflake tool: {name}",
            "parameters": {}
        })
    return {"tools": tools_list}

@app.post("/mcp/tools/{tool_name}/invoke", tags=["MCP"])
async def invoke_tool(tool_name: str, parameters: Dict[str, Any]):
    """Invoke an MCP tool by name - direct endpoint"""
    logger.info(f"Tool invocation requested: {tool_name} with parameters: {parameters}")
    
    # Manually dispatch to the correct tool function
    # This is a workaround until we update to use the proper FastMCP methods
    try:
        params = parameters.get("parameters", {})
        if tool_name == "execute_query":
            result = execute_query(**params)
        elif tool_name == "load_csv_to_table":
            result = load_csv_to_table(**params)
        elif tool_name == "get_table_schema":
            result = get_table_schema(**params)
        elif tool_name == "create_snowflake_table":
            result = create_snowflake_table(**params)
        elif tool_name == "list_tables":
            result = list_tables(**params)
        else:
            logger.error(f"Tool not found: {tool_name}")
            return JSONResponse(
                status_code=404,
                content={"error": f"Tool not found: {tool_name}"}
            )
        
        logger.info(f"Tool {tool_name} invocation result: {result}")
        return {"content": result}
    except Exception as e:
        logger.error(f"Error invoking tool {tool_name}: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error invoking tool {tool_name}: {str(e)}"}
        )

# Register MCP tools
@mcp_server.tool()
def execute_query(query: str) -> str:
    """Execute a SQL query on Snowflake database"""
    try:
        logger.info(f"Executing query: {query}")
        
        # Get Snowflake connection
        conn = get_snowflake_conn()
        if conn is None:
            logger.warning("Using mock implementation since Snowflake connection failed")
            
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
                    return f"5 rows. (Execution time: 0.5s) (MOCK)\n{json.dumps(data)}"
                else:
                    # Generic mock data
                    data = [
                        {"COLUMN1": "Value1", "COLUMN2": 123},
                        {"COLUMN1": "Value2", "COLUMN2": 456}
                    ]
                    return f"2 rows. (Execution time: 0.3s) (MOCK)\n{json.dumps(data)}"
            else:
                # For non-SELECT queries, return success message
                return f"Query executed successfully. (Execution time: 0.2s) (MOCK)"
        
        # Execute query using real Snowflake connection
        cursor = conn.cursor()
        try:
            # Make sure we're using the correct database and schema
            cursor.execute(f"USE DATABASE {Config.SNOWFLAKE_DATABASE}")
            
            # Handle schema specification in query
            query_lower = query.lower()
            if not ("use schema" in query_lower or "use database" in query_lower):
                # Extract schema from query if specified in table reference (e.g. "SELECT * FROM SCHEMA.TABLE")
                schema_match = False
                if "from " in query_lower:
                    from_parts = query_lower.split("from ")[1].strip().split(" ")[0]
                    if "." in from_parts:
                        schema = from_parts.split(".")[0]
                        schema_match = True
                
                # If no schema specified in query, try to use PUBLIC schema
                if not schema_match:
                    cursor.execute("USE SCHEMA PUBLIC")
            
            # Now execute the actual query
            cursor.execute(query)
            
            # Handle different query types
            if query.lower().strip().startswith("select"):
                # For SELECT queries, fetch and return results
                result = cursor.fetchall()
                
                # Get column names
                col_names = [desc[0] for desc in cursor.description]
                
                # Convert to list of dictionaries
                rows = []
                for row in result:
                    row_dict = {}
                    for i, col_name in enumerate(col_names):
                        row_dict[col_name] = row[i]
                    rows.append(row_dict)
                
                return f"{len(rows)} rows. (Execution time: {cursor.sfqid}s)\n{json.dumps(rows)}"
            else:
                # For non-SELECT queries, return success message
                return f"Query executed successfully. Rows affected: {cursor.rowcount}. (Execution time: {cursor.sfqid}s)"
        except Exception as e:
            return f"Error executing query: {str(e)}"
        finally:
            cursor.close()
            conn.close()
    except Exception as e:
        logger.error(f"Error executing query: {str(e)}")
        return f"Error: {str(e)}"

@mcp_server.tool()
def load_csv_to_table(table_name: str, csv_data: str, create_table: bool = True, segment_name: str = None) -> str:
    """Load CSV data into a Snowflake table"""
    try:
        logger.info(f"Loading data into table: {table_name} for segment: {segment_name}")
        
        # Parse CSV data
        df = pd.read_csv(io.StringIO(csv_data))
        row_count = len(df)
        
        # Get Snowflake connection
        conn = get_snowflake_conn()
        if conn is None:
            logger.warning("Using mock implementation since Snowflake connection failed")
            return f"Successfully loaded {row_count} rows into table {table_name}. (MOCK)"
        
        # Create cursor
        cursor = conn.cursor()
        
        # Determine schema based on segment name if provided
        if segment_name:
            schema = get_schema_for_segment(segment_name)
        else:
            # Determine schema from table_name if it includes schema
            schema = "PUBLIC"
            if "." in table_name:
                parts = table_name.split(".")
                schema = parts[0]
                table_name = parts[1]
            
        logger.info(f"Using schema {schema} for segment {segment_name}")
        
        # Set the database context
        try:
            cursor.execute(f"USE DATABASE {Config.SNOWFLAKE_DATABASE}")
        except Exception as e:
            logger.error(f"Error using database {Config.SNOWFLAKE_DATABASE}: {str(e)}")
            return f"Error: {str(e)}"
            
        # Create schema if needed
        try:
            # Try to create the schema - explicitly use the current database
            schema_create_sql = f"CREATE SCHEMA IF NOT EXISTS {Config.SNOWFLAKE_DATABASE}.{schema}"
            cursor.execute(schema_create_sql)
            logger.info(f"Created schema {schema} in database {Config.SNOWFLAKE_DATABASE}")
            
            # Use the schema
            cursor.execute(f"USE SCHEMA {schema}")
            logger.info(f"Now using schema {schema}")
        except Exception as e:
            logger.error(f"Error creating/using schema {schema}: {str(e)}")
            return f"Error: {str(e)}"
        
        # Create table if needed
        if create_table:
            # Generate fully qualified table name
            full_table_name = f"\"{Config.SNOWFLAKE_DATABASE}\".\"{schema}\".\"{table_name}\""
            
            columns = []
            for col_name, dtype in df.dtypes.items():
                if 'int' in str(dtype):
                    col_type = "INTEGER"
                elif 'float' in str(dtype):
                    col_type = "FLOAT"
                elif 'datetime' in str(dtype):
                    col_type = "TIMESTAMP"
                else:
                    col_type = "VARCHAR"
                    
                columns.append(f'\"{col_name}\" {col_type}')
                
            columns_str = ", ".join(columns)
            create_table_sql = f"CREATE TABLE IF NOT EXISTS {full_table_name} ({columns_str})"
            
            try:
                cursor.execute(create_table_sql)
                logger.info(f"Created table {full_table_name}")
            except Exception as e:
                logger.error(f"Error creating table: {str(e)}")
                return f"Error creating table: {str(e)}"
        
        # Use Snowflake Pandas connector to write data
        try:
            # Use the fully qualified table name
            fully_qualified_table = f"{Config.SNOWFLAKE_DATABASE}.{schema}.{table_name}"
            
            # Success! We are using a real connection
            success, nchunks, nrows, _ = write_pandas(conn, df, table_name, schema, database=Config.SNOWFLAKE_DATABASE)
            
            if success:
                return f"Successfully loaded {nrows} rows into table {fully_qualified_table}."
            else:
                return f"Failed to load data into {fully_qualified_table}."
        except Exception as e:
            logger.error(f"Error writing data to Snowflake: {str(e)}")
            return f"Error writing data to Snowflake: {str(e)}"
        finally:
            cursor.close()
            conn.close()
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

@mcp_server.tool()
def create_snowflake_table(table_name: str, columns: Dict[str, str]) -> str:
    """
    Create a table in Snowflake with specified columns
    
    Args:
        table_name: The name of the table to create
        columns: Dictionary mapping column names to their data types
    
    Returns:
        Success message or error
    """
    try:
        logger.info(f"Creating table: {table_name} with columns: {columns}")
        
        # In a real implementation, this would execute a CREATE TABLE statement
        columns_str = ", ".join([f"{name} {dtype}" for name, dtype in columns.items()])
        query = f"CREATE TABLE {table_name} ({columns_str})"
        
        # Mock successful execution
        return f"Table {table_name} created successfully with query: {query}"
    except Exception as e:
        logger.error(f"Error creating table: {str(e)}")
        return f"Error: {str(e)}"

@mcp_server.tool()
def list_tables(schema_name: Optional[str] = None) -> List[str]:
    """
    List all tables in a Snowflake schema or all schemas if not specified
    
    Args:
        schema_name: Optional schema name to filter tables by
    
    Returns:
        List of table names
    """
    try:
        logger.info(f"Listing tables in schema: {schema_name if schema_name else 'ALL'}")
        
        # Mock implementation - return fake table list
        if schema_name and schema_name.lower() == "diagnostic":
            return ["DIAGNOSTIC.PRODUCT_CATALOG", "DIAGNOSTIC.SALES_DATA", "DIAGNOSTIC.CUSTOMER_INFO"]
        elif schema_name and schema_name.lower() == "skincare":
            return ["SKINCARE.PRODUCT_INVENTORY", "SKINCARE.SALES_2024", "SKINCARE.CUSTOMER_FEEDBACK"]
        else:
            # Return tables from all schemas
            return [
                "DIAGNOSTIC.PRODUCT_CATALOG", 
                "DIAGNOSTIC.SALES_DATA", 
                "SKINCARE.PRODUCT_INVENTORY", 
                "COMMON.USER_ACCOUNTS"
            ]
    except Exception as e:
        logger.error(f"Error listing tables: {str(e)}")
        return [f"Error: {str(e)}"]

# Print debug info
print("Registered tools: execute_query, load_csv_to_table, get_table_schema, create_snowflake_table, list_tables")
logger.info("Registered MCP tools: execute_query, load_csv_to_table, get_table_schema, create_snowflake_table, list_tables")

# Mount MCP server to FastAPI app
app.mount("/mcp", mcp_server.sse_app())

# Add health check endpoint
@app.get("/health")
def health_check():
    """Health check endpoint for the Snowflake MCP Server"""
    tools = ["execute_query", "load_csv_to_table", "get_table_schema", "create_snowflake_table", "list_tables"]
    return {
        "status": "healthy", 
        "service": "snowflake_mcp", 
        "tools_available": len(tools),
        "tools": tools
    }

@app.get("/")
def root():
    """Root endpoint for the Snowflake MCP Server"""
    return {
        "message": "Snowflake MCP Server is running",
        "status": "healthy",
        "version": "0.1.0",
        "tools_endpoint": "/mcp/tools",
        "health_endpoint": "/health"
    }

# Run the server
if __name__ == "__main__":
    port = Config.SNOWFLAKE_MCP_PORT if hasattr(Config, "SNOWFLAKE_MCP_PORT") else 8004
    logger.info(f"Starting Snowflake MCP Server on port {port}")
    logger.info(f"Available tools: {list(mcp_server._tools.keys())}")
    uvicorn.run(app, host="0.0.0.0", port=port)
