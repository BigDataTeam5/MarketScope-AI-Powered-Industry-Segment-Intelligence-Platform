"""
Utility functions for Streamlit frontend
"""
import requests
import streamlit as st
import time
import sys
import os
import logging
from typing import Dict, Any, Optional

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import config and paths properly
from config.paths import setup_paths
setup_paths()
from config.config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("frontend_utils")

# Use the MCP server as the API endpoint
API_URL = "http://localhost:8000"

def get_available_segments() -> list:
    """Get available segments from API"""
    try:
        response = requests.get(f"{API_URL}/segments", timeout=5)
        if response.status_code == 200:
            segments = response.json().get("segments", [])
            if segments:
                return segments
        logger.warning("Couldn't get segments from API, using defaults")
        return list(Config.SEGMENT_CONFIG.keys())
    except Exception as e:
        logger.error(f"Error getting segments: {e}")
        return list(Config.SEGMENT_CONFIG.keys())

def check_server_connection() -> bool:
    """Check if the API server is running"""
    global API_URL
    try:
        logger.info(f"Checking API server connection at {API_URL}/health")
        response = requests.get(f"{API_URL}/health", timeout=2)
        if response.status_code == 200:
            logger.info("API server connection successful")
            return True
        else:
            logger.error(f"API server returned status code {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Server connection error: {str(e)}")
        try:
            # Try alternate ports as a fallback
            alternate_ports = [8000, 8002, 8003, 5000]
            for port in alternate_ports:
                alt_url = f"http://localhost:{port}/health"
                logger.info(f"Trying alternate port: {alt_url}")
                alt_response = requests.get(alt_url, timeout=1)
                if alt_response.status_code == 200:
                    logger.info(f"Found API server on alternate port {port}")
                    API_URL = f"http://localhost:{port}"
                    return True
        except Exception:
            pass
        return False

def extract_response_text(response_data: Any) -> str:
    """Extract the response text from various response formats"""
    if isinstance(response_data, str):
        return response_data
    
    if isinstance(response_data, dict):
        # Check for common response fields
        for key in ["response", "answer", "content", "output", "result", "text"]:
            if key in response_data:
                return response_data[key]
        
        # Check for messages
        if "messages" in response_data:
            messages = response_data.get("messages", [])
            if messages and isinstance(messages, list) and len(messages) > 0:
                last_message = messages[-1]
                if isinstance(last_message, dict) and "content" in last_message:
                    return last_message["content"]
    
    # If all else fails, return the string representation
    return str(response_data)

def process_query(query: str, segment: Optional[str] = None, use_case: Optional[str] = None) -> str:
    """
    Process a query through the API
    
    Args:
        query: Query text
        segment: Optional segment name
        use_case: Optional use case
        
    Returns:
        Response text
    """
    # Check if server is running
    if not check_server_connection():
        return "Error: Backend server is not running. Please start the API server."
    
    try:
        # Create a unique session ID if not already present
        if "session_id" not in st.session_state:
            st.session_state.session_id = str(int(time.time()))
        
        # Send request to API
        response = requests.post(
            f"{API_URL}/query",
            json={
                "query": query,
                "segment": segment,
                "use_case": use_case,
                "agent_type": "unified"
            },
            timeout=10
        )
        
        # Check response
        if response.status_code != 200:
            return f"Error: Server returned status code {response.status_code}. {response.text}"
        
        # Get session ID for polling
        session_id = response.json().get("session_id")
        
        # Poll for results
        max_retries = 40
        retries = 0
        
        with st.spinner("Processing your query..."):
            while retries < max_retries:
                try:
                    poll_response = requests.get(
                        f"{API_URL}/query/{session_id}",
                        timeout=5
                    )
                    
                    if poll_response.status_code == 200:
                        result = poll_response.json()
                        
                        if result["status"] == "completed":
                            return extract_response_text(result["result"])
                        elif result["status"] == "error":
                            return f"Error: {result.get('error', 'Unknown error')}"
                        elif result["status"] == "processing":
                            time.sleep(0.5)
                            retries += 1
                            continue
                    else:
                        return f"Error: Server returned status code {poll_response.status_code}"
                except requests.exceptions.RequestException as e:
                    return f"Error connecting to server: {str(e)}"
        
        return "Query processing timed out. Please try again."
    
    except requests.exceptions.RequestException as e:
        return f"Error connecting to server: {str(e)}"

def process_csv_data(csv_data: str, segment: Optional[str] = None, table_name: Optional[str] = None, query: Optional[str] = None) -> str:
    """
    Process CSV data through the API
    
    Args:
        csv_data: CSV data as string
        segment: Optional segment name
        table_name: Optional table name
        query: Optional query to run on the data
        
    Returns:
        Response text
    """
    # Check if server is running
    if not check_server_connection():
        return "Error: Backend server is not running. Please start the API server."
    
    try:
        # Send request to API
        response = requests.post(
            f"{API_URL}/process_csv",
            json={
                "csv_data": csv_data,
                "segment": segment,
                "table_name": table_name,
                "query": query
            },
            timeout=30
        )
        
        # Check response
        if response.status_code != 200:
            return f"Error: Server returned status code {response.status_code}. {response.text}"
        
        # Get session ID for polling
        session_id = response.json().get("session_id")
        
        # Poll for results
        max_retries = 60  # CSV processing may take longer
        retries = 0
        
        with st.spinner("Processing your CSV data..."):
            while retries < max_retries:
                try:
                    poll_response = requests.get(
                        f"{API_URL}/query/{session_id}",
                        timeout=5
                    )
                    
                    if poll_response.status_code == 200:
                        result = poll_response.json()
                        
                        if result["status"] == "completed":
                            return extract_response_text(result["result"])
                        elif result["status"] == "error":
                            return f"Error: {result.get('error', 'Unknown error')}"
                        elif result["status"] == "processing":
                            time.sleep(1.0)  # Longer delay for CSV processing
                            retries += 1
                            continue
                    else:
                        return f"Error: Server returned status code {poll_response.status_code}"
                except requests.exceptions.RequestException as e:
                    return f"Error connecting to server: {str(e)}"
        
        return "CSV processing timed out. Please try again."
    
    except requests.exceptions.RequestException as e:
        return f"Error connecting to server: {str(e)}"

def sidebar():
    """Create sidebar with configuration options"""
    with st.sidebar:
        st.title("MarketScope AI")
        
        # Initialize session state for segment
        if "selected_segment" not in st.session_state:
            st.session_state.selected_segment = None
        
        # Get available segments (preferably from API)
        segments = get_available_segments()
        
        # Handle segments as dictionaries or strings
        if segments and isinstance(segments[0], dict):
            # Use segment names for display, but keep full info
            segment_names = [seg.get('name', str(seg)) for seg in segments]
            segment_ids = [seg.get('id', str(seg)) for seg in segments]
            
            # Find the current index
            current_index = 0
            if st.session_state.selected_segment:
                try:
                    if isinstance(st.session_state.selected_segment, dict):
                        current_segment_id = st.session_state.selected_segment.get('id')
                        if current_segment_id in segment_ids:
                            current_index = segment_ids.index(current_segment_id)
                    else:
                        current_segment = st.session_state.selected_segment
                        if current_segment in segment_ids:
                            current_index = segment_ids.index(current_segment)
                except (ValueError, AttributeError):
                    current_index = 0
            
            # Display segment names in the dropdown
            selected_name = st.selectbox(
                "Select Healthcare Segment",
                options=segment_names,
                index=current_index
            )
            
            # Get the corresponding segment object
            selected_index = segment_names.index(selected_name)
            st.session_state.selected_segment = segment_ids[selected_index]
        else:
            # Fallback for simple string segments
            st.session_state.selected_segment = st.selectbox(
                "Select Healthcare Segment",
                options=segments,
                index=segments.index(st.session_state.selected_segment) if st.session_state.selected_segment in segments else 0
            )
        
        # Display connection status
        st.markdown("### Connection Status")
        if check_server_connection():
            st.success("Server Connected")
        else:
            st.error("Server Disconnected")
            
            # Add reconnect button
            if st.button("Try Reconnecting"):
                if check_server_connection():
                    st.success("✅ Reconnected successfully!")
                else:
                    st.error("❌ Still disconnected. Please check if the server is running.")
