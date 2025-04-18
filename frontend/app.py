import streamlit as st

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add root directory to path for imports
from frontend.utils import sidebar, get_server_status
from agents.custom_mcp_client import MCPClient

# Set page config to full width
st.set_page_config(layout="wide")

# Initialize ALL session state variables
if "selected_segment" not in st.session_state:
    st.session_state.selected_segment = None
if "sales_data" not in st.session_state:
    st.session_state.sales_data = None
if "snowflake_uploaded" not in st.session_state:
    st.session_state.snowflake_uploaded = False
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "trends_result" not in st.session_state:
    st.session_state.trends_result = None
if "market_size_result" not in st.session_state:
    st.session_state.market_size_result = None
# Add widget-specific session state initializations
if "csv_uploader" not in st.session_state:
    st.session_state.csv_uploader = None
if "segment_selectbox" not in st.session_state:
    st.session_state.segment_selectbox = "Skin Care Segment"  # Set default segment

def show():
    """Show main app interface with market size analysis"""
    try:
        sidebar()
        
        st.title("MarketScope AI - Market Size Analysis")
        
        # Check if a segment is selected
        if st.session_state.get("selected_segment"):
            segment = st.session_state.selected_segment
            st.info(f"Currently analyzing: **{segment}**")
        
        # Add Server Status component
        with st.expander("MCP Server Status", expanded=False):
            st.markdown("### MarketScope MCP Server Status")
            st.markdown("Check the status of all backend MCP servers that power MarketScope AI:")
            
            # Get status of all MCP servers
            server_status = get_server_status()
            
            # Display server status in a clean format
            for server_name, status in server_status.items():
                col1, col2 = st.columns([1, 3])
                with col1:
                    if status == "healthy":
                        st.markdown("✅")
                    elif status == "unhealthy":
                        st.markdown("⚠️")
                    else:
                        st.markdown("❌")
                
                with col2:
                    display_name = server_name.replace("_", " ").title()
                    if server_name == "unified":
                        display_name += " (Main)"
                    
                    if status == "healthy":
                        st.markdown(f"**{display_name}**: Connected")
                    elif status == "unhealthy":
                        st.markdown(f"**{display_name}**: Responding but has errors")
                    else:
                        st.markdown(f"**{display_name}**: Not available")
            
            if st.button("Refresh Status"):
                st.rerun()

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    show()