import streamlit as st
import requests
import time
import uuid
import os
import pandas as pd
import io

# Try to import streamlit_shadcn_ui, but provide fallback if components fail
try:
    import streamlit_shadcn_ui as ui
    SHADCN_AVAILABLE = True
except:
    SHADCN_AVAILABLE = False

# API and MCP endpoints
#API_URL = "http://localhost:8001"
SNOWFLAKE_MCP_URL = "http://localhost:8000"  # Updated to match the Snowflake server port

# Page configuration
st.set_page_config(
    page_title="MarketScope AI",
    page_icon="🔍",
    layout="wide",
)

# Add custom CSS for a modern look
st.markdown("""
<style>
    .main {
        padding: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #f0f4ff;
        border-left: 4px solid #3B82F6;
    }
    .assistant-message {
        background-color: #f0fdf4;
        border-left: 4px solid #10B981;
    }
    .modern-button {
        background-color: #3B82F6;
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: 500;
        cursor: pointer;
        transition: background-color 0.2s;
    }
    .modern-button:hover {
        background-color: #2563EB;
    }
    .modern-button:disabled {
        background-color: #94a3b8;
        cursor: not-allowed;
    }
    .modern-input {
        border: 1px solid #d1d5db;
        border-radius: 0.5rem;
        padding: 0.5rem;
    }
    .modern-card {
        border-radius: 0.5rem;
        border: 1px solid #e5e7eb;
        padding: 1rem;
        margin-bottom: 1rem;
        background-color: white;
    }
    .file-uploader {
        border: 2px dashed #e5e7eb;
        border-radius: 0.5rem;
        padding: 1.5rem;
        text-align: center;
        background-color: #f9fafb;
        transition: all 0.2s;
    }
    .file-uploader:hover {
        border-color: #3B82F6;
        background-color: #f0f4ff;
    }
    .tabs-container {
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Session management
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "query_in_progress" not in st.session_state:
    st.session_state.query_in_progress = False
if "upload_status" not in st.session_state:
    st.session_state.upload_status = None
if "snowflake_tools" not in st.session_state:
    st.session_state.snowflake_tools = []

# Function to extract text response
def extract_response_text(response_data):
    """Extract text response from various formats."""
    if isinstance(response_data, str):
        return response_data
        
    if isinstance(response_data, dict):
        for key in ["answer", "content", "output", "result", "text"]:
            if key in response_data:
                return response_data[key]
                
        if "messages" in response_data:
            messages = response_data.get("messages", [])
            if messages and isinstance(messages, list) and len(messages) > 0:
                last_message = messages[-1]
                if isinstance(last_message, dict) and "content" in last_message:
                    return last_message["content"]
    
    return str(response_data)

# Function to load CSV to Snowflake with segment-based schema creation
def upload_csv_to_snowflake(file_data, filename, segment_name=None, table_name=None):
    """Upload CSV to Snowflake using existing database and creates segment-based schema."""
    try:
        # First save the file temporarily
        with open(filename, "wb") as f:
            f.write(file_data)
        
        # Read the CSV to detect headers and infer data types
        df = pd.read_csv(filename)
        
        # Map pandas dtypes to Snowflake data types
        dtype_mapping = {
            'int64': 'INTEGER',
            'float64': 'FLOAT',
            'bool': 'BOOLEAN',
            'datetime64[ns]': 'TIMESTAMP',
            'object': 'STRING'
        }
        
        # Build the column definitions for CREATE TABLE
        column_defs = []
        for col_name, dtype in df.dtypes.items():
            # Convert special characters in column names
            clean_col_name = col_name.replace(' ', '_').replace('-', '_').upper()
            # Map pandas dtype to Snowflake type
            sf_type = dtype_mapping.get(str(dtype), 'STRING')
            column_defs.append(f'"{clean_col_name}" {sf_type}')
        
        # Join column definitions with commas
        columns_sql = ",\n        ".join(column_defs)
        
        # Use provided table name or default
        table_name = table_name or "SALES_DATA"
        
        # Get the current database from Snowflake connection
        response = requests.post(
            f"{SNOWFLAKE_MCP_URL}/invoke",
            json={
                "name": "get_connection_status",
                "inputs": {}
            }
        )
        
        if response.status_code != 200:
            return False, "Error getting current database information"
            
        status = response.json()
        current_db = status.get("database", "HEALTHCARE_PRODUCTS")
        
        # Use segment name as schema name if provided
        schema_name = segment_name if segment_name else "HEALTHCARE_PRODUCTS"
        
        # Create full schema path with current database
        full_schema = f"{current_db}.{schema_name}"
        
        # Call the Snowflake MCP server to create the schema if segment name is provided
        if segment_name:
            create_schema_query = f"""
            CREATE SCHEMA IF NOT EXISTS {full_schema};
            """
            
            response = requests.post(
                f"{SNOWFLAKE_MCP_URL}/invoke",
                json={
                    "name": "execute_query",
                    "inputs": {
                        "query": create_schema_query
                    }
                }
            )
            
            if response.status_code != 200:
                return False, f"Error creating segment schema: {response.text}"
        
        # Call the Snowflake MCP server to create the table dynamically
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {full_schema}.{table_name} (
        {columns_sql}
        );
        """
        
        st.write("Creating table with schema:")
        st.code(create_table_query, language="sql")
        
        response = requests.post(
            f"{SNOWFLAKE_MCP_URL}/invoke",
            json={
                "name": "execute_query",
                "inputs": {
                    "query": create_table_query
                }
            }
        )
        
        if response.status_code != 200:
            return False, f"Error creating table: {response.text}"
        
        # First, get the column names
        columns = ", ".join([f'"{col.replace(" ", "_").replace("-", "_").upper()}"' for col in df.columns])
        
        # Prepare data for bulk insert
        rows_inserted = 0
        batch_size = 100  # Insert in batches to avoid too large requests
        
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i+batch_size]
            values_list = []
            
            # Generate values for each row
            for _, row in batch_df.iterrows():
                values = []
                for val in row:
                    if pd.isna(val):
                        values.append("NULL")
                    elif isinstance(val, (int, float)):
                        values.append(str(val))
                    elif isinstance(val, bool):
                        values.append("TRUE" if val else "FALSE")
                    else:
                        # Escape single quotes in string values
                        values.append(f"'{str(val).replace('\'', '\'\'')}'" if val is not None else "NULL")
                
                values_sql = ", ".join(values)
                values_list.append(f"({values_sql})")
            
            # Join all rows with commas
            all_values = ",\n".join(values_list)
            
            # Create the complete INSERT statement
            insert_query = f"""
            INSERT INTO {full_schema}.{table_name} ({columns})
            VALUES {all_values};
            """
            
            # Execute the insert
            response = requests.post(
                f"{SNOWFLAKE_MCP_URL}/invoke",
                json={
                    "name": "execute_query",
                    "inputs": {
                        "query": insert_query
                    }
                }
            )
            
            if response.status_code != 200:
                return False, f"Error inserting data (batch {i//batch_size + 1}): {response.text}"
            
            # Count inserted rows
            try:
                result = response.json()
                if "affected_rows" in result:
                    rows_inserted += int(result["affected_rows"])
            except:
                # If we can't parse the exact count, just continue
                pass
        
        # Clean up the temporary file
        os.remove(filename)
        
        return True, f"Data uploaded successfully! Inserted {rows_inserted} rows into {full_schema}.{table_name}"
        
    except Exception as e:
        # Ensure temporary file is cleaned up even if there's an error
        if os.path.exists(filename):
            os.remove(filename)
        return False, f"Error uploading to Snowflake: {str(e)}"

# Function to fetch Snowflake tools from the MCP server
def get_snowflake_tools():
    """Fetch available tools from the Snowflake MCP server."""
    try:
        response = requests.get(f"{SNOWFLAKE_MCP_URL}/openapi.json")
        if response.status_code == 200:
            schema = response.json()
            tools = []
            if "paths" in schema:
                for path, methods in schema["paths"].items():
                    if "/invoke" in path:
                        for method, details in methods.items():
                            if "requestBody" in details and "content" in details["requestBody"]:
                                content = details["requestBody"]["content"]
                                if "application/json" in content and "schema" in content["application/json"]:
                                    schema_ref = content["application/json"]["schema"]
                                    if "properties" in schema_ref and "name" in schema_ref["properties"]:
                                        if "enum" in schema_ref["properties"]["name"]:
                                            tools = schema_ref["properties"]["name"]["enum"]
            return tools
        return []
    except Exception as e:
        st.error(f"Error fetching Snowflake tools: {str(e)}")
        return []

# Add this function in streamlit_agent_ui.py
def is_server_running(url):
    """Check if a server is running and accessible."""
    try:
        response = requests.get(f"{url}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

# Sidebar with minimal shadcn components
with st.sidebar:
    st.markdown("## 🔍 MarketScope AI")
    st.markdown("### AI-Powered Industry Segment Intelligence")
    
    st.markdown("### About")
    st.markdown("MarketScope AI helps you understand industry segments and market trends using AI-powered analysis.")
    
    # Sample questions - using standard Streamlit components
    with st.expander("Sample Questions"):
        st.markdown("""
        - What does Kotler suggest about segmentation strategies for pharma?
        - How should pharmaceutical companies approach market segmentation?
        - What are the key segmentation variables for healthcare products?
        - What trends do you see in the uploaded healthcare sales data?
        """)
    
    # LangSmith tracing info
    st.markdown("### LangSmith Tracing")
    st.markdown("All queries are automatically traced in LangSmith.")
    
    project = os.getenv("LANGSMITH_PROJECT", "default")
    if SHADCN_AVAILABLE:
        try:
            ui.link_button(
                text="View Traces", 
                url=f"https://smith.langchain.com/projects/{project}/traces",
                variant="outline",
                key="langsmith_link"
            )
        except:
            st.markdown(f"[View Traces](https://smith.langchain.com/projects/{project}/traces)")
    else:
        st.markdown(f"[View Traces](https://smith.langchain.com/projects/{project}/traces)")
    
    # Direct query option
    if SHADCN_AVAILABLE:
        try:
            direct_query = ui.switch(label="Direct OpenAI query (testing only)", key="direct_query_switch")
        except:
            direct_query = st.checkbox("Direct OpenAI query (testing only)")
    else:
        direct_query = st.checkbox("Direct OpenAI query (testing only)")

# Main content area
st.title("MarketScope AI")
st.subheader("AI-Powered Industry Segment Intelligence Platform")

# Create tabs for different functionality
main_tab1, main_tab2 = st.tabs(["💬 Chat & Query", "📊 Data Upload"])

with main_tab1:
    # Chat container with messages
    st.markdown("### Conversation")
    chat_container = st.container()

    with chat_container:
        if not st.session_state.messages:
            st.info("Start by asking a question about market segmentation")
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    # Query input and submit
    st.markdown("### Ask a question")

    col1, col2 = st.columns([5, 1])

    with col1:
        if SHADCN_AVAILABLE:
            try:
                query = ui.input(
                    placeholder="e.g., What are the key segmentation variables for healthcare products?",
                    disabled=st.session_state.query_in_progress,
                    key="query_input"
                )
            except:
                query = st.text_input(
                    "Enter your question",
                    placeholder="e.g., What are the key segmentation variables for healthcare products?",
                    disabled=st.session_state.query_in_progress,
                    key="query_input"
                )
        else:
            query = st.text_input(
                "Enter your question",
                placeholder="e.g., What are the key segmentation variables for healthcare products?",
                disabled=st.session_state.query_in_progress,
                key="query_input"
            )

    with col2:
        if SHADCN_AVAILABLE:
            try:
                submit_btn = ui.button(
                    text="Ask AI", 
                    disabled=st.session_state.query_in_progress,
                    variant="default",
                    key="submit_btn"
                )
            except:
                submit_btn = st.button(
                    "Ask AI", 
                    disabled=st.session_state.query_in_progress,
                    key="submit_btn"
                )
        else:
            submit_btn = st.button(
                "Ask AI", 
                disabled=st.session_state.query_in_progress,
                key="submit_btn"
            )

    # Handle form submission
    if submit_btn and query:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": query})
        st.session_state.query_in_progress = True
        
        # Submit query to API
        try:
            with st.spinner("Processing your query..."):
                response = requests.post(
                    f"{API_URL}/query",
                    json={
                        "query": query, 
                        "session_id": st.session_state.session_id,
                        "direct": direct_query  # Pass the direct query option
                    }
                )
                
                if response.status_code != 200:
                    st.error(f"Error submitting query: {response.text}")
                    st.session_state.query_in_progress = False
                    st.rerun()
                
                # Poll for results
                max_retries = 60  # 2 minutes at 2-second intervals
                retries = 0
                
                while retries < max_retries:
                    poll_response = requests.get(f"{API_URL}/query/{st.session_state.session_id}")
                    
                    if poll_response.status_code == 200:
                        result = poll_response.json()
                        
                        if result["status"] == "completed":
                            try:
                                # Extract response and add to chat
                                if "result" not in result:
                                    st.error("Response missing 'result' field")
                                    break
                                    
                                ai_response = extract_response_text(result["result"])
                                st.session_state.messages.append({"role": "assistant", "content": ai_response})
                                break
                            except Exception as e:
                                st.error(f"Error processing response: {str(e)}")
                                break
                                
                        elif result["status"] == "error":
                            st.error(f"Error processing query: {result.get('error', 'Unknown error')}")
                            break
                        elif result["status"] == "processing":
                            # Continue polling
                            pass
                    else:
                        st.error(f"Error retrieving results: {poll_response.status_code}")
                        break
                    
                    retries += 1
                    time.sleep(2)
                    
                    if retries >= max_retries:
                        st.error("Query processing timed out. Please try again.")
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        
        # Reset query in progress
        st.session_state.query_in_progress = False
        st.rerun()

with main_tab2:
    st.markdown("## 📊 Healthcare Sales Data Upload")
    st.markdown("""
    Upload your healthcare product sales data in CSV format. The data will be stored in your 
    Snowflake database under a schema based on the market segment.
    """)
    
    # CSV file example
    st.markdown("### Example CSV Format")
    example_data = """
    PRODUCT_ID,PRODUCT_NAME,SEGMENT,COMPANY,SALES,UNITS,DATE
    P001,GlowSerum,Skincare,AesteticMD,12500.00,250,2025-01-15
    P002,HydraBoost Cream,Skincare,AesteticMD,9800.50,196,2025-01-15
    P003,FitBand Pro,Wearables,TechHealth,18750.00,125,2025-01-18
    P004,QuickTest COVID,Diagnostics,MedDiag,8900.25,356,2025-01-20
    """
    st.code(example_data, language="csv")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload your healthcare sales data (CSV)",
        type=["csv"],
        help="Upload a CSV file with healthcare product sales data"
    )
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Preview uploaded file
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.markdown("### Data Preview")
                st.dataframe(df.head(5))
                
                # Reset file pointer
                uploaded_file.seek(0)
                
                # Target segments and tables
                st.markdown("### Market Segment Selection")

                # Show available segments only
                segments_options = ["SKINCARE", "WEARABLES", "DIAGNOSTICS", "OTC-PHARMACEUTICALS"]

                # Add information about segments
                st.info("""
                Choose a market segment for your data. The system will create a schema with this name
                in the configured database. Each segment represents a different healthcare product category.
                """)

                # Direct selectbox for predefined segments
                segment_name = st.selectbox(
                    "Select market segment", 
                    segments_options,
                    help="Select from the predefined healthcare product segments"
                )

                table_name = st.text_input(
                    "Table Name (leave empty for SALES_DATA)", 
                    key="target_table",
                    help="Enter a custom table name or leave empty to use SALES_DATA"
                )
                
            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")
                uploaded_file = None
    
    with col2:
        # Upload button
        if uploaded_file is not None:
            upload_btn = st.button(
                "Upload to Snowflake",
                key="upload_btn",
                help="Click to upload the data to Snowflake"
            )
            
            # Update the upload button handler
            if upload_btn:
                # First check if Snowflake MCP server is running
                if not is_server_running(SNOWFLAKE_MCP_URL):
                    st.error(f"""
                    ⚠️ Cannot connect to Snowflake MCP server at {SNOWFLAKE_MCP_URL}.
                    
                    Please make sure:
                    1. The server is running: `python mcp_server/Snowflake_server.py`
                    2. It's running on port {SNOWFLAKE_MCP_URL.split(':')[-1]}
                    3. Check if there are any firewall issues blocking the connection
                    """)
                else:
                    if not segment_name:
                        st.error("Please select or create a market segment")
                    else:
                        with st.spinner(f"Uploading to Snowflake under {segment_name} segment..."):
                            # Get the file data
                            file_data = uploaded_file.getvalue()
                            
                            # Create a temporary file
                            temp_file = f"temp_{uuid.uuid4()}.csv"
                            
                            # Upload to Snowflake with segment schema
                            success, message = upload_csv_to_snowflake(
                                file_data, 
                                temp_file,
                                segment_name=segment_name,
                                table_name=table_name if table_name else None
                            )
                            
                            if success:
                                st.session_state.upload_status = {
                                    "status": "success",
                                    "message": message
                                }
                            else:
                                st.session_state.upload_status = {
                                    "status": "error",
                                    "message": message
                                }
                        st.rerun()

        # Display upload status
        if st.session_state.upload_status:
            if st.session_state.upload_status["status"] == "success":
                st.success(st.session_state.upload_status["message"])
            else:
                st.error(st.session_state.upload_status["message"])

    # Snowflake connection status
    st.markdown("### Snowflake Connection Status")
    if st.button("Check Snowflake Connection"):
        try:
            # Fetch Snowflake tools if not already available
            if not st.session_state.snowflake_tools:
                st.session_state.snowflake_tools = get_snowflake_tools()
            
            # Check connection status
            response = requests.post(
                f"{SNOWFLAKE_MCP_URL}/invoke",
                json={
                    "name": "get_connection_status",
                    "inputs": {}
                }
            )
            
            if response.status_code == 200:
                status = response.json()
                if isinstance(status, dict) and "status" in status:
                    if status["status"] == "connected":
                        st.success("Connected to Snowflake")
                        st.json(status)
                    else:
                        st.warning(f"Snowflake connection status: {status['status']}")
                        st.json(status)
                else:
                    st.error("Invalid response from Snowflake MCP server")
            else:
                st.error(f"Error checking Snowflake connection: {response.status_code}")
        except Exception as e:
            st.error(f"Error connecting to Snowflake MCP server: {str(e)}")

    # Snowflake schema and table info
    with st.expander("View Healthcare Products Schema"):
        if st.button("Refresh Schema Info"):
            try:
                response = requests.post(
                    f"{SNOWFLAKE_MCP_URL}/invoke",
                    json={
                        "name": "get_schema_info",
                        "inputs": {
                            "schema_name": "HEALTHCARE_PRODUCTS"
                        }
                    }
                )
                
                if response.status_code == 200:
                    schema_info = response.json()
                    st.subheader("Tables in HEALTHCARE_PRODUCTS Schema")
                    st.json(schema_info)
                else:
                    st.error(f"Error fetching schema info: {response.status_code}")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# System information with tabs 
st.markdown("---")
tab1, tab2, tab3 = st.tabs(["System Architecture", "System Health", "Server Configuration"])

with tab1:
    st.markdown("### To run this application:")
    
    st.markdown("**1. Start the Pinecone S3 server:**")
    st.code("python mcp_server/pinecone_s3_server.py  # Runs on port 8000")
    
    st.markdown("**2. Start the Snowflake MCP server:**")
    st.code("python mcp_server/Snowflake_server.py  # Runs on port 8080")
    
    st.markdown("**3. Start the API server:**")
    st.code("python mcp_server/api_server.py  # Runs on port 8001")
    
    st.markdown("**4. Start the Streamlit UI:**")
    st.code("streamlit run frontend/streamlit_agent_ui.py  # Runs on port 8501")

with tab2:
    if st.button("Check API Connection", key="health_check_btn"):
        try:
            response = requests.get(f"{API_URL}/health")
            if response.status_code == 200:
                health_data = response.json()
                st.success("API Connection: Healthy")
                
                # Display LangSmith config without debug info
                project = health_data.get("langsmith_config", {}).get("LANGSMITH_PROJECT", "")
                if project:
                    st.markdown(f"LangSmith project: `{project}`")
                    st.markdown(f"[View LangSmith Traces](https://smith.langchain.com/projects/{project}/traces)")
            else:
                st.error(f"API Connection: {response.status_code}")
        except Exception as e:
            st.error(f"API Connection Error: {str(e)}")
    
    if st.button("Check Snowflake MCP Connection", key="snowflake_check_btn"):
        try:
            response = requests.get(f"{SNOWFLAKE_MCP_URL}/health")
            if response.status_code == 200:
                st.success("Snowflake MCP Connection: Healthy")
            else:
                st.error(f"Snowflake MCP Connection: {response.status_code}")
        except Exception as e:
            st.error(f"Snowflake MCP Connection Error: {str(e)}")

with tab3:
    st.markdown("### Server Configuration")
    
    st.markdown("#### 1. Snowflake MCP Server")
    st.markdown("""
    Create a `.env` file with the following Snowflake credentials:
    ```
    SNOWFLAKE_USER=your_username
    SNOWFLAKE_PASSWORD=your_password
    SNOWFLAKE_ACCOUNT=your_account
    SNOWFLAKE_DATABASE=HEALTHCARE_PRODUCTS
    SNOWFLAKE_WAREHOUSE=COMPUTE_WH
    ```
    
    The system uses the database specified in the SNOWFLAKE_DATABASE environment variable.
    All data will be stored within this database in schemas named after market segments.
    """)
    
    st.markdown("#### 2. API Server")
    st.markdown("""
    Update the `api_server.py` to include the Snowflake MCP server in its configuration.
    ```python
    # Add to server configuration
    servers = {
        "pinecone_s3": {"url": "http://localhost:8080"},
        "snowflake": {"url": "http://localhost:8081"}
    }
    ```
    """)
