# agents/snowflake_agent.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import warnings
from io import BytesIO
import base64
import time
from typing import Dict, Any, List, Optional
import logging
import json

# Import our custom MCP client instead of the missing one
from custom_mcp_client import CustomMCPClient

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('snowflake_agent')

# Load environment variables
load_dotenv(override=True)

# Initialize MCP client for accessing Snowflake server
def init_mcp_client():
    """Initialize custom MCP client to connect to Snowflake server"""
    try:
        # Default to localhost:8000 but allow override through environment variable
        mcp_server_url = os.getenv("MCP_SERVER_URL", "http://localhost:8000")
        client = CustomMCPClient(base_url=mcp_server_url)
        # Test connection
        health = client.get("/health")
        if health.get("status") != "healthy":
            logger.error(f"MCP server health check failed: {health}")
            raise ConnectionError("Could not connect to MCP server")
        logger.info(f"Successfully connected to MCP server at {mcp_server_url}")
        return client
    except Exception as e:
        logger.error(f"Error initializing MCP client: {str(e)}")
        raise

# Query data from user-uploaded tables
def get_table_data(table_name: str, limit: int = 1000) -> pd.DataFrame:
    """Query data from a specified table using MCP"""
    try:
        client = init_mcp_client()
        # First, check if the table exists
        schema_result = client.invoke("get_schema_info")
        schema_data = json.loads(schema_result) if isinstance(schema_result, str) else schema_result
        
        # Extract schema and table names
        schemas = [item.get("TABLE_SCHEMA") for item in schema_data if "TABLE_SCHEMA" in item]
        
        if not schemas:
            logger.warning(f"No schemas found in database")
            return pd.DataFrame()
        
        # Query the table data
        query = f"SELECT * FROM {table_name} LIMIT {limit}"
        result = client.invoke("execute_query", {"query": query})
        
        # Process the result
        if isinstance(result, str):
            if "rows. (Execution time:" in result:
                # Extract the JSON part from the string result
                json_start = result.find('[')
                json_end = result.rfind(']') + 1
                if json_start >= 0 and json_end > json_start:
                    data_json = result[json_start:json_end]
                    data = json.loads(data_json)
                    return pd.DataFrame(data)
            elif "Error" in result:
                logger.error(f"Query error: {result}")
                return pd.DataFrame()
            
            try:
                # Try to parse the entire result as JSON
                data = json.loads(result)
                if isinstance(data, list):
                    return pd.DataFrame(data)
            except:
                logger.error(f"Could not parse result as JSON: {result[:200]}...")
                return pd.DataFrame()
                
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error getting table data: {str(e)}")
        return pd.DataFrame()

def get_user_tables() -> List[str]:
    """Get a list of user tables available in the database"""
    try:
        client = init_mcp_client()
        schema_result = client.invoke("get_schema_info")
        
        # Process the result
        if isinstance(schema_result, str):
            try:
                schema_data = json.loads(schema_result)
                table_names = []
                
                # Extract schema names from the result
                schemas = []
                for schema_item in schema_data:
                    schema_name = schema_item.get("TABLE_SCHEMA")
                    if schema_name and schema_name not in ["INFORMATION_SCHEMA"]:
                        schemas.append(schema_name)
                
                logger.info(f"Found schemas: {schemas}")
                
                # For each schema, query the tables directly using SQL instead of the API
                for schema_name in schemas:
                    # Use direct SQL query to get table names only (avoids datetime serialization issues)
                    query = f"""
                    SELECT TABLE_NAME 
                    FROM INFORMATION_SCHEMA.TABLES 
                    WHERE TABLE_SCHEMA = '{schema_name}'
                    """
                    query_result = client.invoke("execute_query", {"query": query})
                    
                    # Parse the query result
                    if isinstance(query_result, str):
                        if "rows. (Execution time:" in query_result:
                            # Extract the JSON part from the string result
                            json_start = query_result.find('[')
                            json_end = query_result.rfind(']') + 1
                            if json_start >= 0 and json_end > json_start:
                                try:
                                    data_json = query_result[json_start:json_end]
                                    tables_data = json.loads(data_json)
                                    
                                    for table_item in tables_data:
                                        table_name = table_item.get("TABLE_NAME")
                                        if table_name:
                                            # Include schema name in fully qualified table name
                                            table_names.append(f"{schema_name}.{table_name}")
                                            logger.info(f"Found table: {schema_name}.{table_name}")
                                except Exception as e:
                                    logger.error(f"Error parsing table results for {schema_name}: {str(e)}")
                
                return table_names
            except Exception as e:
                logger.error(f"Failed to parse schema result: {str(e)}")
                logger.error(f"Schema result: {schema_result[:200]}...")
        
        return []
    except Exception as e:
        logger.error(f"Error getting user tables: {str(e)}")
        return []

def get_graph_specs_from_llm(data: pd.DataFrame, model_name:str) -> dict:
    """Get graph specifications from LLM based on the table data."""
    try:
        # Convert DataFrame to a JSON-like string for LLM input
        data_summary = data.to_dict(orient="list")
        prompt = f"""
        Based on the following data:
        {data_summary}
        Generate a graph specification in this format:
        - Title: [Graph title]
        - Type: [line/bar/scatter]
        - X-axis: [label and settings]
        - Y-axis: [label and settings]
        - Colors: [color scheme]
        - Additional elements: [grid, legend position, etc.]

        Focus on making the graph visually informative and easy to interpret.
        """
        llm = initialize_llm(model_name)
        # Send the prompt to the LLM
        response = llm.invoke(prompt)
        if not response or not hasattr(response, "content"):
            raise ValueError("LLM did not return a valid response.")
        
        # Extract the content from the AIMessage object
        if hasattr(response, "content"):
            response_text = response.content
        else:
            response_text = str(response)
        
        # Parse the LLM response into a dictionary
        specs = {}
        for line in response_text.split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                specs[key.strip()] = value.strip()
        
        return specs
    except Exception as e:
        logger.error(f"Error getting graph specs from LLM: {str(e)}")
        return {}

def create_graph_from_llm_specs(data: pd.DataFrame, specs: dict) -> str:
    """Create a graph based on LLM specifications with normalized values."""
    try:
        if 'DATE' not in data.columns:
            # Try to identify a date column with a different name
            date_candidates = [col for col in data.columns if 'DATE' in col.upper() or 'TIME' in col.upper()]
            if date_candidates:
                date_col = date_candidates[0]
                data['DATE'] = data[date_col]
            else:
                # If no date column found, create a placeholder index as DATE
                logger.warning("No date column found. Using row index as date.")
                data['DATE'] = pd.Series(range(len(data))).astype(str)
        
        # Create a copy of the DataFrame for normalization
        df_normalized = data.copy()
        
        # Normalize all numeric columns except DATE
        for col in df_normalized.columns:
            if col != 'DATE' and pd.api.types.is_numeric_dtype(df_normalized[col]):
                min_val = df_normalized[col].min()
                max_val = df_normalized[col].max()
                if max_val - min_val != 0:  # Avoid division by zero
                    df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
        
        # Create figure with larger size to accommodate legend
        plt.figure(figsize=(15, 8))
        
        # Format x-axis data
        if pd.api.types.is_datetime64_any_dtype(df_normalized['DATE']):
            x = pd.to_datetime(df_normalized["DATE"]).dt.strftime('%b %d, %Y')
        else:
            x = df_normalized["DATE"]
        
        # Determine the graph type
        graph_type = specs.get("Type", "line").lower()  # Default to "line"
        y_label = specs.get("Y-axis", "Normalized Value (0-1 scale)")
        title = f"{specs.get('Title', 'Data Visualization')} (Normalized)"
        
        # Define color palette
        colors = plt.cm.Set2(np.linspace(0, 1, len([c for c in df_normalized.columns if c != 'DATE' and pd.api.types.is_numeric_dtype(df_normalized[c])])))
        
        # Get numeric columns only
        numeric_cols = [c for c in df_normalized.columns if c != 'DATE' and pd.api.types.is_numeric_dtype(df_normalized[c])]
        
        # Plot the graph based on the type
        if graph_type == "stacked":
            # Prepare data for stacked area plot
            y_values = df_normalized[numeric_cols].T.values
            plt.stackplot(x, y_values, labels=[col.replace('_', ' ').title() for col in numeric_cols], colors=colors, alpha=0.8)
        elif graph_type == "bar":
            # Bar chart
            for idx, col in enumerate(numeric_cols):
                x_pos = np.arange(len(x))
                width = 0.8 / len(numeric_cols)
                plt.bar(x_pos + idx * width - 0.4 + width/2, 
                        df_normalized[col], 
                        width=width,
                        label=col.replace('_', ' ').title(),
                        color=colors[idx])
            plt.xticks(np.arange(len(x)), x)
        elif graph_type == "scatter":
            # Scatter plot
            for idx, col in enumerate(numeric_cols):
                plt.scatter(x, df_normalized[col],
                            label=col.replace('_', ' ').title(),
                            color=colors[idx],
                            alpha=0.7,
                            s=50)
        else:
            # Default to line plot
            for idx, col in enumerate(numeric_cols):
                plt.plot(x, df_normalized[col], 
                         label=col.replace('_', ' ').title(),
                         marker="o", 
                         color=colors[idx], 
                         linewidth=2)
        
        # Apply formatting
        plt.title(title, fontsize=16, pad=20)
        plt.xlabel(specs.get("X-axis", "Date"), fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # Add legend with better formatting
        plt.legend(
            loc='upper left',
            bbox_to_anchor=(1.05, 1),
            fontsize=12,
            title='Metrics',
            title_fontsize=14,
            frameon=True,
            fancybox=True,
            shadow=True
        )
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha="right", fontsize=12)
        plt.yticks(fontsize=12)
        
        # Adjust layout to prevent legend cutoff
        plt.tight_layout()
        
        # Save the graph
        chart_file_path = "llm_generated_graph.png"
        plt.savefig(
            chart_file_path,
            format="png",
            dpi=150,
            bbox_inches="tight",
            pad_inches=0.5
        )
        plt.close()
        
        return chart_file_path
    except Exception as e:
        logger.error(f"Error creating graph: {str(e)}")
        return None

def generate_graph_for_table(table_name: str, model_name: str = "claude-3-haiku-20240307") -> dict:
    """Generate a graph for the specified table using AI assistance"""
    try:
        # Step 1: Get data from the table
        df = get_table_data(table_name)
        if df.empty:
            return {"status": "failed", "error": f"No data found in table {table_name}"}
        
        # Step 2: Get column information for better understanding of the data
        client = init_mcp_client()
        schema_name = table_name.split('.')[0] if '.' in table_name else None
        base_table_name = table_name.split('.')[-1]
        column_info = client.invoke("get_table_columns", {
            "table_name": base_table_name,
            "schema_name": schema_name
        })
        
        # Step 3: Get graph specifications from LLM
        graph_specs = get_graph_specs_from_llm(df, model_name)
        
        # Step 4: Create and save the graph
        chart_file_path = create_graph_from_llm_specs(df, graph_specs)
        
        # Step 5: Generate analysis with LLM
        analysis_prompt = f"""
        Analyze the following data from table {table_name}:
        {df.head(10).to_string()}
        
        Column Information:
        {column_info}
        
        Graph Specifications:
        {graph_specs}
        
        Provide insights based on the data and the graph. Highlight key trends, patterns, and any significant observations.
        """
        
        llm = initialize_llm(model_name)
        response = llm.invoke(analysis_prompt)
        
        # Handle the AIMessage response correctly
        if hasattr(response, 'content'):
            analysis = response.content
        else:
            analysis = str(response)
        
        return {
            "status": "success",
            "analysis": analysis,
            "chart_path": chart_file_path,
            "summary": df.describe().to_dict(),
            "graph_specs": graph_specs,
            "table_name": table_name,
            "row_count": len(df)
        }
    except Exception as e:
        error_msg = f"Error generating graph for table {table_name}: {str(e)}"
        logger.error(error_msg)
        return {"status": "failed", "error": error_msg}

def analyze_user_table(table_name: str, prompt: str = "", model_name: str = "claude-3-haiku-20240307"):
    """Get AI-generated analysis with graph for a user-uploaded table"""
    try:
        # Generate graph and get initial analysis
        result = generate_graph_for_table(table_name, model_name)
        if result["status"] == "failed":
            return result
        
        # Create a more specific prompt if provided by the user
        if prompt:
            llm_prompt = f"""
            {prompt}
            
            Table: {table_name}
            Data Summary:
            {json.dumps(result['summary'], indent=2)}
            
            Graph Specifications:
            {result['graph_specs']}
            
            Please provide a detailed analysis based on the data and graph.
            """
            
            llm = initialize_llm(model_name)
            response = llm.invoke(llm_prompt)
            
            # Handle the AIMessage response
            if hasattr(response, 'content'):
                analysis = response.content
            else:
                analysis = str(response)
            
            result["analysis"] = analysis
        
        # Always ensure chart_path is included in the response
        chart_path = result.get('chart_path', 'llm_generated_graph.png')
        
        # Make sure the chart file exists
        if not os.path.exists(chart_path):
            logger.warning(f"Chart path not found: {chart_path}")
            # Try to use default location as fallback
            if os.path.exists('llm_generated_graph.png'):
                chart_path = 'llm_generated_graph.png'
        
        return {
            "status": "success",
            "analysis": result["analysis"],
            "chart_path": chart_path,
            "summary": result.get('summary', {}),
            "graph_specs": result.get('graph_specs', ""),
            "table_name": table_name,
            "row_count": result.get('row_count', 0)
        }
    except Exception as e:
        error_msg = f"Error analyzing user table {table_name}: {str(e)}"
        logger.error(error_msg)
        return {"status": "failed", "error": error_msg}

def initialize_llm(model_name="claude-3-haiku-20240307"):
    """Initialize LLM based on model name."""
    # Initialize the appropriate LLM based on the model name
    if "claude" in model_name:
        llm = ChatAnthropic(
            model=model_name,
            temperature=0,
            anthropic_api_key=os.environ.get('ANTHROPIC_API_KEY')
        )
    elif "gemini" in model_name:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0,
            google_api_key=os.environ.get('GEMINI_API_KEY')
        )
    elif "deepseek" in model_name:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=model_name,
            temperature=0,
            api_key=os.environ.get('DEEP_SEEK_API_KEY')
        )
    elif "grok" in model_name:
        from langchain_groq import ChatGroq
        llm = ChatGroq(
            model=model_name,
            temperature=0,
            api_key=os.environ.get('GROK_API_KEY')
        )
    else:
        llm = ChatAnthropic(
            model="claude-3-haiku-20240307",
            temperature=0,
            anthropic_api_key=os.environ.get('ANTHROPIC_API_KEY')
        )
    return llm

def get_ai_analysis_with_graph(prompt: str, model_name: str = "claude-3-haiku-20240307", table_name: str = None):
    """Get AI-generated analysis with graph for user-uploaded data"""
    if not table_name:
        return {"status": "failed", "error": "Table name must be provided"}
        
    return analyze_user_table(table_name, prompt, model_name)

if __name__ == "__main__":
    # Debug: Get direct information about schemas
    client = init_mcp_client()
    result = client.invoke("execute_query", {"query": "SELECT TABLE_SCHEMA FROM INFORMATION_SCHEMA.TABLES GROUP BY TABLE_SCHEMA"})
    print("Available schemas:", result)
    
    # List available user tables
    print("\nAvailable User Tables:")
    tables = get_user_tables()
    for table in tables:
        print(f"- {table}")
    
    # Analyze a user table if available
    if tables:
        user_table = tables[0]  # Take the first table as an example
        print(f"\nAnalyzing user table: {user_table}")
        table_analysis = get_ai_analysis_with_graph(
            prompt=f"Analyze data patterns and trends in the {user_table} table. Identify key insights.",
            model_name="claude-3-haiku-20240307",
            table_name=user_table
        )
        if table_analysis["status"] == "success":
            print(table_analysis["analysis"])
            print(f"Chart saved at: {table_analysis['chart_path']}")
        else:
            print(f"Analysis failed: {table_analysis.get('error', 'Unknown error')}")
    else:
        print("No user tables found. Please upload data to analyze.")