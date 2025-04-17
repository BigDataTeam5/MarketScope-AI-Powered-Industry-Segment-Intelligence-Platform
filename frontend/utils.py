import requests
import streamlit as st
import time
import sys
import os
from typing import Dict, Any, Optional

# Add the root directory to the Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import config from the config package
from config import Config

# Use Config for API URL
API_URL = f"http://localhost:{Config.API_PORT}"

def get_available_agents():
    try:
        response = requests.get(f"{API_URL}/agents")
        if response.status_code == 200:
            return response.json()["agents"]
        return ["marketing_management"]
    except:
        return ["marketing_management"]

def check_server_connection():
    try:
        requests.get(f"{API_URL}/health")
        return True
    except:
        return False

def extract_response_text(response_data: Any) -> str:
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

def process_query(query: str, agent_type: str, config: Optional[Dict[str, Any]] = None) -> str:
    if not check_server_connection():
        return "Error: Backend server is not running. Please start the FastAPI server on port 8002."
    try:
        response = requests.post(
            f"{API_URL}/query",
            json={
                "query": query,
                "agent_type": agent_type,
                "config": config,
                "session_id": st.session_state.session_id
            },
            timeout=10
        )
        if response.status_code != 200:
            return f"Error: Server returned status code {response.status_code}. {response.text}"
        max_retries = 40
        retries = 0
        while retries < max_retries:
            try:
                poll_response = requests.get(
                    f"{API_URL}/query/{st.session_state.session_id}",
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

def sidebar():
    """Create sidebar with configuration options"""
    with st.sidebar:
        st.title("MarketScope AI")
        
        # Initialize session state for segment
        if "selected_segment" not in st.session_state:
            st.session_state.selected_segment = None
            
        # Segment selection
        segments = [
            "Diagnostic Segment",
            "Supplement Segment",
            "Otc Pharmaceutical Segment",
            "Fitness Wearable Segment",
            "Skin Care Segment"
        ]
        
        st.session_state.selected_segment = st.selectbox(
            "Select Healthcare Segment",
            options=segments,
            index=segments.index(st.session_state.selected_segment) if st.session_state.selected_segment else 0
        )
        
        # Display connection status
        st.markdown("### Connection Status")
        if check_server_connection():
            st.success("✅ Server Connected")
        else:
            st.error("❌ Server Disconnected")