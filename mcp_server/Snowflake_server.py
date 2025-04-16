#!/usr/bin/env python
import os
import json
import time
import logging
import snowflake.connector
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from typing import Optional, Any, List, Dict

load_dotenv(override=True)

try:
    from config import setup_langsmith
    setup_langsmith()
except ImportError:
    # Define your own setup_langsmith function if config.py doesn't exist
    def setup_langsmith():
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = "snowflake-mcp"

from langsmith import traceable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('snowflake_mcp')

# Load environment variables
load_dotenv()

# Create MCP server instance
mcp = FastMCP("SnowflakeMCP")

# Snowflake state to track connection and query history
snowflake_state = {
    "last_query": None,
    "last_results": None,
    "connection_status": "disconnected",
    "query_history": []
}

class SnowflakeConnection:
    """Snowflake database connection management class"""
    def __init__(self):
        # Initialize configuration from environment variables
        self.config = {
            "user": os.getenv("SNOWFLAKE_USER"),
            "account": os.getenv("SNOWFLAKE_ACCOUNT"),
            "database": os.getenv("SNOWFLAKE_DATABASE"),
            "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
        }
        
        # Determine authentication method
        private_key_file = os.getenv("SNOWFLAKE_PRIVATE_KEY_FILE")
        
        # Priority 1: Key pair authentication if file is provided and exists
        if private_key_file and os.path.exists(private_key_file):
            # Check if using passphrase or not
            passphrase = os.getenv("SNOWFLAKE_PRIVATE_KEY_PASSPHRASE")
            if passphrase:
                logger.info("Using key pair authentication with passphrase")
            else:
                logger.info("Using key pair authentication without passphrase")
                
            # Try to setup key pair authentication
            auth_success = self._setup_key_pair_auth(private_key_file, passphrase)
            
            # If key pair auth failed, fall back to password
            if not auth_success:
                logger.info("Falling back to password authentication")
                password = os.getenv("SNOWFLAKE_PASSWORD")
                if password:
                    self.config["password"] = password
                else:
                    logger.error("No password provided as fallback. Authentication will likely fail.")
        else:
            # Priority 2: Password authentication
            password = os.getenv("SNOWFLAKE_PASSWORD")
            if password:
                self.config["password"] = password
                logger.info("Using password authentication")
            else:
                logger.error("No authentication method configured. Please provide either a private key or password.")
        
        self.conn: Optional[snowflake.connector.SnowflakeConnection] = None
        
        # Log config (excluding sensitive info)
        safe_config = {k: v for k, v in self.config.items() 
                      if k not in ['password', 'private_key', 'private_key_passphrase']}
        logger.info(f"Initialized with config: {json.dumps(safe_config)}")
    
    def _setup_key_pair_auth(self, private_key_file: str, passphrase: str = None) -> bool:
        """Set up key pair authentication"""
        try:
            # Read private key file
            with open(private_key_file, "rb") as key_file:
                private_key = key_file.read()
                
            # Try to load the key using snowflake's recommended approach
            from cryptography.hazmat.backends import default_backend
            from cryptography.hazmat.primitives.serialization import load_pem_private_key
            
            logger.info(f"Loading private key from {private_key_file}")
            
            # Use passphrase only if provided
            p_key = load_pem_private_key(
                private_key,
                password=passphrase.encode() if passphrase else None,
                backend=default_backend()
            )
            
            # Convert key to DER format
            from cryptography.hazmat.primitives.serialization import Encoding, PrivateFormat, NoEncryption
            pkb = p_key.private_bytes(
                encoding=Encoding.DER,
                format=PrivateFormat.PKCS8,
                encryption_algorithm=NoEncryption()
            )
            
            # Add to config (this is what Snowflake expects)
            self.config["private_key"] = pkb
            
            # If we had a passphrase, add it to config
            if passphrase:
                self.config["private_key_passphrase"] = passphrase
                
            logger.info("Private key loaded successfully")
            return True
                
        except Exception as e:
            logger.error(f"Error setting up key pair authentication: {str(e)}")
            logger.error("Details:", exc_info=True)
            return False
    
    def ensure_connection(self) -> snowflake.connector.SnowflakeConnection:
        """Ensure database connection is available"""
        try:
            # Check if connection needs to be re-established
            if self.conn is None:
                logger.info("Creating new Snowflake connection...")
                self.conn = snowflake.connector.connect(
                    **self.config,
                    client_session_keep_alive=True,
                    network_timeout=15,
                    login_timeout=15
                )
                self.conn.cursor().execute("ALTER SESSION SET TIMEZONE = 'UTC'")
                logger.info("New connection established and configured")
                snowflake_state["connection_status"] = "connected"
            
            # Test if connection is valid
            try:
                self.conn.cursor().execute("SELECT 1")
            except:
                logger.info("Connection lost, reconnecting...")
                self.conn = None
                snowflake_state["connection_status"] = "reconnecting"
                return self.ensure_connection()
                
            return self.conn
        except Exception as e:
            logger.error(f"Connection error: {str(e)}")
            snowflake_state["connection_status"] = "error"
            raise

    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        """Execute SQL query and return results"""
        start_time = time.time()
        logger.info(f"Executing query: {query[:200]}...")  # Log only first 200 characters
        snowflake_state["last_query"] = query
        
        try:
            conn = self.ensure_connection()
            with conn.cursor() as cursor:
                # For write operations use transaction
                if any(query.strip().upper().startswith(word) for word in ['INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER']):
                    cursor.execute("BEGIN")
                    try:
                        cursor.execute(query)
                        conn.commit()
                        logger.info(f"Write query executed in {time.time() - start_time:.2f}s")
                        result = [{"affected_rows": cursor.rowcount}]
                        snowflake_state["last_results"] = result
                        
                        # Record in history (limited to last 10 queries)
                        snowflake_state["query_history"].append({
                            "query": query,
                            "type": "write",
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "duration": time.time() - start_time,
                            "affected_rows": cursor.rowcount
                        })
                        if len(snowflake_state["query_history"]) > 10:
                            snowflake_state["query_history"].pop(0)
                            
                        return result
                    except Exception as e:
                        conn.rollback()
                        raise
                else:
                    # Read operations
                    cursor.execute(query)
                    if cursor.description:
                        columns = [col[0] for col in cursor.description]
                        rows = cursor.fetchall()
                        results = [dict(zip(columns, row)) for row in rows]
                        logger.info(f"Read query returned {len(results)} rows in {time.time() - start_time:.2f}s")
                        snowflake_state["last_results"] = results
                        
                        # Record in history (limited to last 10 queries)
                        snowflake_state["query_history"].append({
                            "query": query,
                            "type": "read",
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "duration": time.time() - start_time,
                            "rows_returned": len(results)
                        })
                        if len(snowflake_state["query_history"]) > 10:
                            snowflake_state["query_history"].pop(0)
                            
                        return results
                    
                    # Record in history (limited to last 10 queries)
                    snowflake_state["query_history"].append({
                        "query": query,
                        "type": "read",
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "duration": time.time() - start_time,
                        "rows_returned": 0
                    })
                    if len(snowflake_state["query_history"]) > 10:
                        snowflake_state["query_history"].pop(0)
                        
                    result = []
                    snowflake_state["last_results"] = result
                    return result
                
        except snowflake.connector.errors.ProgrammingError as e:
            logger.error(f"SQL Error: {str(e)}")
            logger.error(f"Error Code: {getattr(e, 'errno', 'unknown')}")
            raise
        except Exception as e:
            logger.error(f"Query error: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            raise

    def close(self):
        """Close database connection"""
        if self.conn:
            try:
                self.conn.close()
                logger.info("Connection closed")
                snowflake_state["connection_status"] = "disconnected"
            except Exception as e:
                logger.error(f"Error closing connection: {str(e)}")
            finally:
                self.conn = None

# Create a singleton instance
snowflake_db = SnowflakeConnection()

# Tools
@mcp.tool()
@traceable(name="execute_query", run_type="chain")
def execute_query(query: str):
    """Execute a SQL query on Snowflake."""
    try:
        start_time = time.time()
        results = snowflake_db.execute_query(query)
        execution_time = time.time() - start_time
        
        # Format the results for better readability
        if isinstance(results, list) and results:
            if len(results) == 1 and "affected_rows" in results[0]:
                return f"Query executed successfully. Affected rows: {results[0]['affected_rows']}. (Execution time: {execution_time:.2f}s)"
            else:
                # For regular result sets, format in a readable way
                # Limit results if too large to avoid response size issues
                max_rows_to_show = 50
                if len(results) > max_rows_to_show:
                    return f"Query returned {len(results)} rows. (Execution time: {execution_time:.2f}s)\nShowing first {max_rows_to_show} rows:\n{json.dumps(results[:max_rows_to_show], indent=2)}"
                else:
                    return f"Query returned {len(results)} rows. (Execution time: {execution_time:.2f}s)\n{json.dumps(results, indent=2)}"
        else:
            return f"Query executed successfully. No results returned. (Execution time: {execution_time:.2f}s)"
    
    except Exception as e:
        error_msg = f"Error executing query: {str(e)}"
        logger.error(error_msg)
        return error_msg

@mcp.tool()
@traceable(name="get_schema_info", run_type="chain")
def get_schema_info(schema_name: str = None):
    """Get information about available tables in a schema."""
    try:
        if schema_name:
            query = f"""
            SELECT 
                table_name, 
                table_type,
                row_count,
                created,
                last_altered
            FROM 
                information_schema.tables
            WHERE 
                table_schema = '{schema_name.upper()}'
            ORDER BY 
                table_name
            """
        else:
            query = """
            SELECT 
                table_schema, 
                COUNT(*) as table_count
            FROM 
                information_schema.tables
            WHERE 
                table_schema NOT IN ('INFORMATION_SCHEMA')
            GROUP BY 
                table_schema
            ORDER BY 
                table_schema
            """
        
        results = snowflake_db.execute_query(query)
        
        if not results:
            return f"No {'tables found in schema ' + schema_name if schema_name else 'schemas found'}"
        
        return json.dumps(results, indent=2)
    
    except Exception as e:
        error_msg = f"Error retrieving schema information: {str(e)}"
        logger.error(error_msg)
        return error_msg

@mcp.tool()
@traceable(name="get_table_columns", run_type="chain")
def get_table_columns(table_name: str, schema_name: str = None):
    """Get column information for a specific table."""
    try:
        schema_filter = f"AND table_schema = '{schema_name.upper()}'" if schema_name else ""
        
        query = f"""
        SELECT 
            column_name, 
            data_type, 
            character_maximum_length,
            is_nullable,
            column_default
        FROM 
            information_schema.columns
        WHERE 
            table_name = '{table_name.upper()}'
            {schema_filter}
        ORDER BY 
            ordinal_position
        """
        
        results = snowflake_db.execute_query(query)
        
        if not results:
            return f"No columns found for table {table_name}"
        
        return json.dumps(results, indent=2)
    
    except Exception as e:
        error_msg = f"Error retrieving column information: {str(e)}"
        logger.error(error_msg)
        return error_msg

@mcp.tool()
@traceable(name="get_query_history", run_type="chain")
def get_query_history():
    """Get the history of recent queries executed on Snowflake."""
    try:
        if not snowflake_state["query_history"]:
            return "No queries have been executed yet."
        
        return json.dumps(snowflake_state["query_history"], indent=2)
    
    except Exception as e:
        error_msg = f"Error retrieving query history: {str(e)}"
        logger.error(error_msg)
        return error_msg

@mcp.tool()
@traceable(name="get_connection_status", run_type="chain")
def get_connection_status():
    """Get the current Snowflake connection status."""
    try:
        # Test the connection
        snowflake_db.ensure_connection()
        
        # Return status information
        return {
            "status": snowflake_state["connection_status"],
            "database": os.getenv("SNOWFLAKE_DATABASE"),
            "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
            "user": os.getenv("SNOWFLAKE_USER"),
            "account": os.getenv("SNOWFLAKE_ACCOUNT")
        }
    
    except Exception as e:
        error_msg = f"Error checking connection status: {str(e)}"
        logger.error(error_msg)
        return {"status": "error", "message": error_msg}

@mcp.tool()
@traceable(name="describe_database", run_type="chain")
def describe_database():
    """Get a comprehensive overview of the connected database."""
    try:
        # Get database info
        db_info_query = """
        SELECT current_database() as database_name, 
               current_schema() as current_schema,
               current_warehouse() as current_warehouse,
               current_role() as current_role
        """
        db_info = snowflake_db.execute_query(db_info_query)
        
        # Get schemas
        schemas_query = """
        SELECT schema_name, 
               created, 
               last_altered
        FROM information_schema.schemata
        WHERE schema_name NOT IN ('INFORMATION_SCHEMA')
        ORDER BY schema_name
        """
        schemas = snowflake_db.execute_query(schemas_query)
        
        # Get tables count by schema
        tables_query = """
        SELECT table_schema, 
               COUNT(*) as table_count,
               SUM(row_count) as total_rows
        FROM information_schema.tables
        WHERE table_schema NOT IN ('INFORMATION_SCHEMA')
        GROUP BY table_schema
        ORDER BY table_schema
        """
        tables = snowflake_db.execute_query(tables_query)
        
        result = {
            "database_info": db_info[0] if db_info else {},
            "schemas": schemas,
            "tables_by_schema": tables
        }
        
        return json.dumps(result, indent=2)
    
    except Exception as e:
        error_msg = f"Error describing database: {str(e)}"
        logger.error(error_msg)
        return error_msg

@mcp.tool()
@traceable(name="create_database", run_type="chain")
def create_database(database_name: str):
    """Create a new database in Snowflake."""
    try:
        # SQL to create a new database
        query = f"CREATE DATABASE IF NOT EXISTS {database_name}"
        result = execute_query(query)
        
        # If successful, update current connection
        if "error" not in result.lower():
            # Update the env variable for future connections
            os.environ["SNOWFLAKE_DATABASE"] = database_name
            # Return success message
            return f"Database '{database_name}' created successfully and set as current database."
        return result
    except Exception as e:
        error_msg = f"Error creating database: {str(e)}"
        logger.error(error_msg)
        return error_msg

@mcp.tool()
@traceable(name="use_database", run_type="chain")
def use_database(database_name: str):
    """Switch to a different database."""
    try:
        # First check if database exists
        check_query = f"""
        SELECT COUNT(*) as DB_EXISTS 
        FROM INFORMATION_SCHEMA.DATABASES 
        WHERE DATABASE_NAME = '{database_name.upper()}'
        """
        results = snowflake_db.execute_query(check_query)
        
        if results[0]["DB_EXISTS"] == 0:
            return f"Database '{database_name}' does not exist. Use create_database tool to create it first."
        
        # Switch to the database
        use_query = f"USE DATABASE {database_name}"
        snowflake_db.execute_query(use_query)
        
        # Update the env variable for future connections
        os.environ["SNOWFLAKE_DATABASE"] = database_name
        
        # Close the current connection to force a new one with the new database
        snowflake_db.close()
        
        return f"Successfully switched to database '{database_name}'."
    except Exception as e:
        error_msg = f"Error switching database: {str(e)}"
        logger.error(error_msg)
        return error_msg

@mcp.tool()
@traceable(name="create_schema", run_type="chain")
def create_schema(schema_name: str, database_name: str = None):
    """Create a new schema in the specified or current database."""
    try:
        # Use specified database or current one
        db_prefix = f"{database_name}." if database_name else ""
        
        # SQL to create a new schema
        query = f"CREATE SCHEMA IF NOT EXISTS {db_prefix}{schema_name}"
        return execute_query(query)
    except Exception as e:
        error_msg = f"Error creating schema: {str(e)}"
        logger.error(error_msg)
        return error_msg

@mcp.tool()
@traceable(name="list_databases", run_type="chain")
def list_databases():
    """List all accessible databases in the Snowflake account."""
    try:
        query = """
        SELECT DATABASE_NAME, CREATED, COMMENT
        FROM INFORMATION_SCHEMA.DATABASES
        ORDER BY DATABASE_NAME
        """
        return execute_query(query)
    except Exception as e:
        error_msg = f"Error listing databases: {str(e)}"
        logger.error(error_msg)
        return error_msg

if __name__ == "__main__":
    try:
        # Import required libraries
        from fastapi import FastAPI, Request, HTTPException
        import uvicorn
        from pydantic import BaseModel
        from typing import Dict, Any

        # Define request model for the /invoke endpoint
        class InvokeRequest(BaseModel):
            name: str
            inputs: Dict[str, Any] = {}

        # Create a FastAPI app
        app = FastAPI(title="Snowflake MCP Server")

        # Tool registry - map tool names to functions
        tool_registry = {
            "execute_query": execute_query,
            "get_schema_info": get_schema_info,
            "get_table_columns": get_table_columns,
            "get_query_history": get_query_history,
            "get_connection_status": get_connection_status,
            "describe_database": describe_database,
            "create_database": create_database,
            "use_database": use_database,
            "create_schema": create_schema,
            "list_databases": list_databases
        }

        # Define the /invoke endpoint
        @app.post("/invoke")
        async def invoke(request: InvokeRequest):
            try:
                if request.name not in tool_registry:
                    raise HTTPException(status_code=404, detail=f"Tool '{request.name}' not found")
                
                # Call the tool function with the inputs
                result = tool_registry[request.name](**request.inputs)
                return result
            except Exception as e:
                logger.error(f"Error invoking tool: {str(e)}")
                return {"error": str(e)}

        # Define the /health endpoint
        @app.get("/health")
        async def health():
            return {"status": "healthy"}

        # Define the /openapi.json endpoint which is automatically generated by FastAPI
        # but we ensure it's up-to-date
        app.openapi_schema = None

        port = 8000  # Use the same port that Streamlit expects
        logger.info(f"Starting custom Snowflake MCP server on port {port}...")
        
        # Run the server
        uvicorn.run(app, host="0.0.0.0", port=port)

    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {str(e)}", exc_info=True)
    finally:
        # Clean up
        snowflake_db.close()