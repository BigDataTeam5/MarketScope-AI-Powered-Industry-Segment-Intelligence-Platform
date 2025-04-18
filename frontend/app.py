import streamlit as st

import sys
import os

# Add root directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from frontend.utils import sidebar, get_server_status
# Set page config to full width
st.set_page_config(layout="wide")

# Initialize session state variables that need to be shared across pages
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

def show():
    """Show main app interface"""
    sidebar()
    
    st.title("Welcome to MarketScope AI")
    
    st.markdown("""
    ## AI-Powered Healthcare Market Intelligence Platform
    
    MarketScope AI helps you understand and analyze different healthcare market segments 
    using advanced AI and natural language processing. Select a segment from the sidebar 
    to get started.
    
    ### Features:
    - Market Segmentation Analysis
    - Strategic Query Optimization
    - Product Comparison
    
    ### Getting Started:
    1. Select a healthcare segment from the sidebar
    2. Navigate to one of our analysis tools
    3. Enter your query or requirements
    """)
    
    # Display current segment if selected
    if st.session_state.get("selected_segment"):
        st.info(f"Currently analyzing: **{st.session_state.selected_segment}**")
    else:
        st.warning("Please select a segment from the sidebar to begin analysis.")
    
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
                # Display icon based on status
                if status == "healthy":
                    st.markdown("✅")
                elif status == "unhealthy":
                    st.markdown("⚠️")
                else:  # unavailable
                    st.markdown("❌")
            
            with col2:
                # Format server name for display
                display_name = server_name.replace("_", " ").title()
                if server_name == "unified":
                    display_name += " (Main)"
                
                if status == "healthy":
                    st.markdown(f"**{display_name}**: Connected")
                elif status == "unhealthy":
                    st.markdown(f"**{display_name}**: Responding but has errors")
                else:  # unavailable
                    st.markdown(f"**{display_name}**: Not available")
        
        # Add a button to refresh server status
        if st.button("Refresh Status"):
            st.experimental_rerun()

show()