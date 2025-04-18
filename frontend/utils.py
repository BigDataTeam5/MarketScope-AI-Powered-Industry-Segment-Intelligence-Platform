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
import pandas as pd
import numpy as np

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import config and paths properly
from config.paths import setup_paths
setup_paths()
from config.config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("frontend_utils")

# Use the external IP address for API endpoint - unified server is primary
API_URL = f"http://34.42.74.104:{Config.MCP_PORT}"

# Individual MCP server endpoints for direct communication with external IP
MCP_SERVERS = {
    "unified": {"url": f"http://34.42.74.104:{Config.MCP_PORT}", "health": "/health"},
    "market_analysis": {"url": "http://34.42.74.104:8001", "health": "/health"},
    "sales_analytics": {"url": "http://34.42.74.104:8002", "health": "/health"},
    "segment": {"url": "http://34.42.74.104:8003", "health": "/health"},
    "snowflake": {"url": "http://34.42.74.104:8004", "health": "/health"}
}

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

def check_server_connection(server_key="unified") -> bool:
    """
    Check if a specific MCP server is running
    Args:
        server_key: Key of the server to check ("unified", "market_analysis", etc.)
    """
    global API_URL
    
    # Default to unified server if not specified
    server_info = MCP_SERVERS.get(server_key, MCP_SERVERS["unified"])
    server_url = server_info["url"]
    health_endpoint = server_info["health"]
    
    try:
        logger.info(f"Checking server connection at {server_url}{health_endpoint}")
        response = requests.get(f"{server_url}{health_endpoint}", timeout=2)
        if response.status_code == 200:
            logger.info(f"Server connection to {server_key} successful")
            # If checking unified and successful, ensure API_URL is set correctly
            if server_key == "unified":
                API_URL = server_url
            return True
        else:
            logger.error(f"Server returned status code {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Server connection error for {server_key}: {str(e)}")
        
        # If this was the unified server, try to fall back to individual servers
        if server_key == "unified":
            try:
                # Try alternate servers as a fallback
                alternate_servers = ["market_analysis", "sales_analytics", "segment"]
                for alt_server in alternate_servers:
                    alt_info = MCP_SERVERS.get(alt_server)
                    if not alt_info:
                        continue
                    
                    alt_url = f"{alt_info['url']}{alt_info['health']}"
                    logger.info(f"Trying alternate server: {alt_url}")
                    
                    try:
                        alt_response = requests.get(alt_url, timeout=1)
                        if alt_response.status_code == 200:
                            logger.info(f"Found API server: {alt_server}")
                            API_URL = alt_info["url"]  # Update global API URL
                            return True
                    except:
                        continue
            except Exception:
                pass
        
        return False

def get_mcp_server_url(segment_name):
    """Get the URL for a specific segment MCP server"""
    from config.config import Config
    
    # Default port if segment not found
    default_port = 8014
    
    # Get port from Config
    port = default_port
    if hasattr(Config, 'SEGMENT_CONFIG') and segment_name in Config.SEGMENT_CONFIG:
        port = Config.SEGMENT_CONFIG[segment_name].get('port', default_port)
    
    # Map segment names to server URLs - use external IP
    return f"http://34.42.74.104:{port}"

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

def process_query(query: str, segment: Optional[str] = None, use_case: Optional[str] = None, server_key: Optional[str] = None) -> str:
    """
    Process a query through the API
    
    Args:
        query: Query text
        segment: Optional segment name
        use_case: Optional use case
        server_key: Optional specific server to use
        
    Returns:
        Response text
    """
    # Use specified server or default to unified
    target_url = get_mcp_server_url(server_key) if server_key else API_URL
    
    # Check if server is running
    if not check_server_connection(server_key if server_key else "unified"):
        return "Error: Backend server is not running. Please start the API server."
    
    try:
        # Create a unique session ID if not already present
        if "session_id" not in st.session_state:
            st.session_state.session_id = str(int(time.time()))
        
        # Send request to API
        response = requests.post(
            f"{target_url}/query",
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
                        f"{target_url}/query/{session_id}",
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
    # For CSV processing, try the snowflake server first, then fallback to unified
    target_url = get_mcp_server_url("snowflake") if check_server_connection("snowflake") else API_URL
    
    # Check if server is running
    if not check_server_connection("unified") and not check_server_connection("snowflake"):
        return "Error: Backend server is not running. Please start the API server."
    
    try:
        # Send request to API
        response = requests.post(
            f"{target_url}/process_csv",
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
                        f"{target_url}/query/{session_id}",
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

def create_visualization_from_mcp(segment_name: str, visualization_type: str, table_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Create a visualization from the segment MCP server
    
    Args:
        segment_name: Name of the segment to use
        visualization_type: Type of visualization to create (e.g., "price_comparison", "rating_analysis", "price_distribution")
        table_name: Optional name of the table to use
        
    Returns:
        Dictionary with visualization data or error message
    """
    # Try to use the segment server directly for visualizations
    segment_url = get_mcp_server_url("segment")
    
    # Check if server is running
    if not check_server_connection("segment") and not check_server_connection("unified"):
        return {"status": "error", "message": "Backend server is not running. Please start the API server."}
    
    try:
        # Find the segment config to get the port - use provided segment name or check config
        segment_port = None
        for segment, config in Config.SEGMENT_CONFIG.items():
            if segment.lower() == segment_name.lower():
                segment_port = config.get("port", 8003)  # Default to segment server port if not specified
                break
        
        # Use the segment server port if found, otherwise use default segment port
        if segment_port and segment_port != 8003:
            # Use segment-specific port if available
            vis_server_url = f"http://34.42.74.104:{segment_port}/mcp"
        else:
            # Use main segment server or unified server as fallback
            vis_server_url = f"{segment_url}/mcp"
        
        # Prepare request payload
        payload = {
            "tool": "create_trends_visualization",
            "params": {
                "visualization_type": visualization_type
            }
        }
        
        # Add table_name if provided
        if table_name:
            payload["params"]["table_name"] = table_name
            
        # Send request to segment MCP server
        logger.info(f"Sending visualization request to {vis_server_url}")
        response = requests.post(
            f"{vis_server_url}/tools",
            json=payload,
            timeout=30
        )
        
        # Check response
        if response.status_code != 200:
            return {"status": "error", "message": f"Server returned status code {response.status_code}. {response.text}"}
        
        # Parse response
        result = response.json()
        
        return result
    except Exception as e:
        logger.error(f"Error in create_visualization_from_mcp: {str(e)}")
        return {"status": "error", "message": f"Error creating visualization: {str(e)}"}

def get_server_status() -> Dict[str, str]:
    """Check the status of all MCP servers"""
    results = {}
    
    for server_key, server_info in MCP_SERVERS.items():
        try:
            response = requests.get(f"{server_info['url']}{server_info['health']}", timeout=1)
            results[server_key] = "healthy" if response.status_code == 200 else "unhealthy"
        except:
            results[server_key] = "unavailable"
    
    return results

def render_chart(chart_data: Dict[str, Any]) -> None:
    """
    Render a chart based on the chart_data format from the segment MCP server
    
    Args:
        chart_data: Chart data in the format returned by the segment MCP server
    """
    try:
        chart_type = chart_data.get("type", "bar")
        title = chart_data.get("title", "Chart")
        x_label = chart_data.get("x_label", "")
        y_label = chart_data.get("y_label", "")
        
        st.subheader(title)
        
        if chart_type == "bar":
            labels = chart_data.get("labels", [])
            values = chart_data.get("values", [])
            color = chart_data.get("color", "blue")
            
            # Create DataFrame for chart
            df = pd.DataFrame({
                x_label: labels,
                y_label: values
            })
            
            # Create Streamlit bar chart
            st.bar_chart(df.set_index(x_label))
            
        elif chart_type == "line":
            labels = chart_data.get("labels", [])
            values = chart_data.get("values", [])
            
            # Create DataFrame for chart
            df = pd.DataFrame({
                x_label: labels,
                y_label: values
            })
            
            # Create Streamlit line chart
            st.line_chart(df.set_index(x_label))
            
        elif chart_type == "histogram":
            hist_values = chart_data.get("hist_values", [])
            bin_edges = chart_data.get("bin_edges", [])
            median = chart_data.get("median")
            mean = chart_data.get("mean")
            
            # Create bins for histogram (use middle of each bin)
            bin_centers = [(bin_edges[i] + bin_edges[i+1])/2 for i in range(len(bin_edges)-1)]
            
            # Create DataFrame for chart
            df = pd.DataFrame({
                "Bin": bin_centers,
                "Frequency": hist_values
            })
            
            # Create Streamlit bar chart to represent histogram
            st.bar_chart(df.set_index("Bin"))
            
            if median is not None and mean is not None:
                st.write(f"**Median:** ${median:.2f}")
                st.write(f"**Mean:** ${mean:.2f}")
        
        else:
            st.warning(f"Unsupported chart type: {chart_type}")
            st.json(chart_data)  # Fallback to showing the raw data
    except Exception as e:
        logger.error(f"Error rendering chart: {str(e)}")
        st.error(f"Error rendering chart: {str(e)}")
        st.json(chart_data)  # Show the raw data on error

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
