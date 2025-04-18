import streamlit as st

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add root directory to path for imports
from frontend.utils import sidebar, get_server_status
from agents.custom_mcp_client import SimpleMCPClient as MCPClient

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
if "market_size_result" not in st.session_state:
    st.session_state.market_size_result = None


def show():
    """Show main app interface with market size analysis"""
    sidebar()
    
    st.title("MarketScope AI - Market Size Analysis")
    
    # Check if a segment is selected
    if st.session_state.get("selected_segment"):
        segment = st.session_state.selected_segment
        st.info(f"Currently analyzing: **{segment}**")
        
        # Display market size analysis
        st.subheader("Market Size Analysis")
        
        with st.spinner(f"Analyzing market size for {segment}..."):
            try:
                # Initialize MCP client
                client = MCPClient("http://localhost:8003/mcp")  # Assuming segment server is on port 8003
                
                # Check if we already have results or need to fetch them
                if "market_size_result" not in st.session_state or not st.session_state.market_size_result:
                    # Call the market size analysis tool
                    result = client.invoke("analyze_market_size", {"segment": segment})
                    st.session_state.market_size_result = result
                else:
                    result = st.session_state.market_size_result
                
                # Check if the call was successful
                if result.get("status") == "success":
                    # Display market summary
                    st.write(result["market_summary"])
                    
                    # Display metrics in three columns
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Addressable Market (TAM)", 
                                  result["market_size"]["TAM"] if result["market_size"]["TAM"] else "Not available")
                    
                    with col2:
                        st.metric("Serviceable Available Market (SAM)", 
                                  result["market_size"]["SAM"] if result["market_size"]["SAM"] else "Not available")
                    
                    with col3:
                        st.metric("Serviceable Obtainable Market (SOM)", 
                                  result["market_size"]["SOM"] if result["market_size"]["SOM"] else "Not available")
                    
                    # Display additional information
                    if "companies_analyzed" in result and result["companies_analyzed"]:
                        st.subheader("Companies Analyzed")
                        st.write(", ".join(result["companies_analyzed"]))
                    
                    if "sources" in result and result["sources"]:
                        st.subheader("Data Sources")
                        for source in result["sources"]:
                            st.write(f"- {source}")
                    
                    # Add refresh button
                    if st.button("Refresh Analysis"):
                        st.session_state.market_size_result = None
                        st.experimental_rerun()
                        
                else:
                    st.error(f"Error analyzing market size: {result.get('message', 'Unknown error')}")
                    st.button("Retry Analysis", on_click=lambda: setattr(st.session_state, "market_size_result", None))
                    
            except Exception as e:
                st.error(f"Failed to connect to the market analysis service: {str(e)}")
                st.write("Please check that all services are running correctly and try again.")
                if st.button("Retry Connection"):
                    st.session_state.market_size_result = None
                    st.experimental_rerun()
    else:
        # No segment selected
        st.warning("Please select a segment from the sidebar to begin market size analysis.")
    
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