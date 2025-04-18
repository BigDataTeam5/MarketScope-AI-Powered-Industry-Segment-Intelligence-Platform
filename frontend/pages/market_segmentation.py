import streamlit as st
import sys
import os
from typing import Dict, Any
import requests
import json

# Add root directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import from config package
from config import Config
from frontend.utils import process_query, sidebar, create_visualization_from_mcp, render_chart, get_mcp_server_url

# Replace:
# from mcp.client import MCPClient

# With:
from agents.custom_mcp_client import MCPClient

# Create a simple MCPClient alternative that uses requests
class SimpleMCPClient:
    def __init__(self, base_url):
        self.base_url = base_url
        print(f"Initialized SimpleMCPClient with base_url: {base_url}")
        
    def invoke(self, tool_name, params=None):
        """Invoke an MCP tool"""
        try:
            # The correct endpoint structure for FastMCP
            url = f"{self.base_url}/invoke/{tool_name}"
            print(f"Invoking tool: {tool_name} at URL: {url}")
            
            response = requests.post(
                url,
                json=params or {},
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                error_msg = f"Request failed with status {response.status_code}: {response.text}"
                print(f"Error: {error_msg}")
                return {
                    "status": "error",
                    "message": error_msg
                }
        except Exception as e:
            print(f"Exception calling MCP: {str(e)}")
            return {
                "status": "error",
                "message": f"Error invoking MCP tool: {str(e)}"
            }
    
    # Add the missing invoke_sync method as an alias to invoke
    invoke_sync = invoke

# Use SimpleMCPClient instead of MCPClient
MCPClient = SimpleMCPClient

# Check your get_mcp_server_url function to ensure it's looking up the right port
def get_mcp_server_url(segment_name):
    """Get the URL for a specific segment MCP server"""
    from config.config import Config
    
    # Default port if segment not found
    default_port = 8014
    
    # Get port from Config
    port = default_port
    if hasattr(Config, 'SEGMENT_CONFIG') and segment_name in Config.SEGMENT_CONFIG:
        port = Config.SEGMENT_CONFIG[segment_name].get('port', default_port)
    else:
        print(f"Warning: Segment {segment_name} not found in Config.SEGMENT_CONFIG")
        print(f"Available segments: {Config.SEGMENT_CONFIG.keys() if hasattr(Config, 'SEGMENT_CONFIG') else 'None'}")
    
    # Map segment names to server URLs
    return f"http://localhost:{port}"

st.set_page_config(layout="wide")


def show():
    """
    Display the Market Segmentation page content
    """
    sidebar()

    # Get model information from session state with fallback to config
    selected_model = "gpt-4o"
    temperature = 0.3
    
    # Get model config for display
    model_name = "gpt-4o"
    
    st.title("Market Segmentation Analysis")
    st.write(f"Using model: {model_name} with temperature: {temperature}")
    
    # Add a description
    st.markdown("""
    This tool helps analyze market segments based on various criteria:
    - Demographics (age, gender, income)
    - Geographic location
    - Customer needs and preferences
    - Industry-specific factors
    """)
    
    # Create tabs for different functionalities - Added new "Market Size Analysis" tab
    tab1, tab2, tab3, tab4 = st.tabs(["Segment Analysis", "Market Size Analysis", "Product Trends Visualization", "Data Upload"])
    
    with tab1:
        # Original segment analysis form - No changes here
        with st.form("market_segment_form"):
            st.subheader("Define Market Segment")
            
            col1, col2 = st.columns(2)
            
            with col1:
                product_type = st.selectbox("Product Category", 
                                          ["Technology", "Healthcare", "Financial Services", "Consumer Goods", "Manufacturing", "Other"])
                
                target_audience = st.multiselect("Target Audience", 
                                              ["B2B - Large Enterprise", "B2B - SMB", "B2C - General", "B2C - Premium", "Government"])
            
            with col2:
                geographic_focus = st.selectbox("Geographic Focus", 
                                              ["North America", "Europe", "Asia-Pacific", "Latin America", "Global"])
                
                price_segment = st.select_slider("Price Segment", 
                                              options=["Budget", "Mid-range", "Premium", "Ultra-premium"])
            
            analysis_query = st.text_area("Specific Analysis Question", 
                                        "What are the key growth opportunities in this segment?")
            
            submitted = st.form_submit_button("Analyze Segment")
            
            if submitted:
                # Build the query for the API
                prompt = f"""Analyze the market segment: {product_type} products for {', '.join(target_audience)} in {geographic_focus} ({price_segment}).
                
                Focus on the following question: "{analysis_query}"
                
                Provide a detailed market analysis including:
                1. Market size and growth potential
                2. Key competitors and their market share
                3. Regulatory considerations if applicable
                4. Customer pain points and needs
                5. Specific recommendations for success in this market segment
                """
                
                # Show analysis in progress
                with st.spinner("Analyzing market segment..."):
                    # Process the query using the market_analysis MCP server
                    response = process_query(prompt, "marketing_management", None, "market_analysis")
                    
                    if response and not response.startswith("Error"):
                        st.success("Analysis complete!")
                        st.subheader("Market Segment Analysis")
                        st.markdown(f"### {product_type} for {', '.join(target_audience)} in {geographic_focus} ({price_segment})")
                        st.markdown(response)
                    else:
                        st.error(f"Failed to generate analysis: {response}")
        
        # Chat interface for follow-up questions
        st.subheader("Follow-up Questions")
        user_question = st.text_input("Ask a question about this market segment:")
        
        if user_question:
            st.chat_message("user").markdown(user_question)
            
            with st.spinner("Generating response..."):
                # Prepare API config
                api_config = {
                    "model": selected_model,
                    "temperature": temperature
                }
                
                # Process the follow-up question
                response = process_query(user_question, "marketing_management", None, "market_analysis")
                
                if response and not response.startswith("Error"):
                    st.chat_message("assistant").markdown(response)
                else:
                    st.chat_message("assistant").error(f"Failed to generate response: {response}")

    with tab2:
        st.subheader("Market Size Analysis (TAM, SAM, SOM)")
        st.markdown("""
        Analyze the Total Addressable Market (TAM), Serviceable Available Market (SAM), 
        and Serviceable Obtainable Market (SOM) for selected industry segments based on Form 10Q reports.
        """)
        
        # Get the selected segment from session state or let user select one
        selected_segment = st.session_state.get("selected_segment")
        
        # Allow user to select a different segment for analysis
        segments = ["Skin Care Segment", "Healthcare - Diagnostic", "Pharmaceutical"]
        try:
            # Try to get segments from the API
            segment_url = get_mcp_server_url("segment")
            response = requests.get(f"{segment_url}/segments", timeout=2)
            if response.status_code == 200:
                segments = response.json().get("segments", segments)
        except Exception as e:
            st.warning(f"Could not retrieve segments from API: {str(e)}")
        
        selected_segment = st.selectbox(
            "Select industry segment for analysis:",
            options=segments,
            index=segments.index(selected_segment) if selected_segment in segments else 0
        )
        
        # Button to perform analysis
        col1, col2 = st.columns([1, 3])
        with col1:
            perform_analysis = st.button("Analyze Market Size")
        with col2:
            refresh_analysis = st.button("Clear & Refresh")
            
        if refresh_analysis:
            # Clear any cached results
            if "market_size_result" in st.session_state:
                del st.session_state.market_size_result
                
        if perform_analysis or "market_size_result" in st.session_state:
            with st.spinner(f"Analyzing market size for {selected_segment}..."):
                try:
                    # Use cached result if available, otherwise fetch from server
                    if "market_size_result" in st.session_state and not refresh_analysis:
                        result = st.session_state.market_size_result
                        st.info("Using cached analysis results. Click 'Clear & Refresh' for fresh data.")
                    else:
                        # Direct API call to get market size data
                        segment_url = get_mcp_server_url(selected_segment)
                        
                        response = requests.post(
                            f"{segment_url}/direct/analyze_market_size",
                            params={"segment": selected_segment},
                            timeout=60
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            # Store in session state
                            st.session_state.market_size_result = result
                        else:
                            st.error(f"Error: Server returned status {response.status_code}")
                            st.code(response.text)
                            return
                            
                    # Replace the display section with this improved version:
                    if result.get("status") == "success":
                        st.success("Analysis complete!")
                        
                        # Display market summary with better formatting
                        if result.get("market_summary"):
                            st.markdown(result["market_summary"])
                        
                        # Display metrics in three columns with better error handling and formatting
                        st.subheader("Market Size Metrics")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            tam_value = result["market_size"]["TAM"] if result["market_size"]["TAM"] else "Not available"
                            st.metric("Total Addressable Market (TAM)", tam_value)
                            st.caption("The total market demand for a product or service")
                        
                        with col2:
                            sam_value = result["market_size"]["SAM"] if result["market_size"]["SAM"] else "Not available"
                            st.metric("Serviceable Available Market (SAM)", sam_value)
                            st.caption("The segment of TAM targeted by your products/services")
                        
                        with col3:
                            som_value = result["market_size"]["SOM"] if result["market_size"]["SOM"] else "Not available"
                            st.metric("Serviceable Obtainable Market (SOM)", som_value)
                            st.caption("The portion of SAM that can be captured")
                        
                        # Display Data Sources section
                        st.subheader("Data Sources")
                        with st.expander("View Source Information", expanded=True):
                            st.markdown(f"**Documents analyzed:** {result.get('documents_analyzed', 15)}")
                            
                            # Show companies analyzed
                            if result.get("companies_analyzed") and len(result["companies_analyzed"]) > 0:
                                companies = ", ".join(result["companies_analyzed"])
                                st.markdown(f"**Companies analyzed:** {companies}")
                            
                            # Show sources in a list format
                            if result.get("sources") and len(result["sources"]) > 0:
                                st.markdown("**Source documents:**")
                                for source in result["sources"]:
                                    st.markdown(f"- {source}")
                    else:
                        st.error(f"Error: {result.get('message', 'Unknown error')}")
                        
                        # Show fallback data if available
                        if result.get("market_size"):
                            st.warning("Showing available data despite errors:")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                tam_value = result["market_size"].get("TAM") or "Not available"
                                st.metric("Total Addressable Market (TAM)", tam_value)
                            
                            with col2:
                                sam_value = result["market_size"].get("SAM") or "Not available"
                                st.metric("Serviceable Available Market (SAM)", sam_value)
                            
                            with col3:
                                som_value = result["market_size"].get("SOM") or "Not available"
                                st.metric("Serviceable Obtainable Market (SOM)", som_value)
                                
                except Exception as e:
                    st.error(f"Error analyzing market size: {str(e)}")
                    st.code(f"Exception details: {type(e).__name__}: {str(e)}")
                    st.info("Try refreshing the page or selecting a different segment.")
        
        # Add vector search functionality
        st.subheader("Segment Data Search")
        st.markdown("Search for specific information within segment Form 10Q reports")
        
        search_query = st.text_input("Enter search query:", placeholder="market growth rate in diagnostic segment")
        search_button = st.button("Search")
        
        if search_button and search_query:
            with st.spinner("Searching and summarizing..."):
                try:
                    # Call the new endpoint
                    segment_url = get_mcp_server_url(selected_segment)
                    
                    # Add this right before your search request for debugging
                    st.info(f"Using server URL: {segment_url} for segment: {selected_segment}")
                    
                    response = requests.post(
                        f"{segment_url}/direct/vector_search_and_summarize",
                        params={"query": search_query, "top_k": 5},
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        if result.get("status") == "success":
                            st.success("Query answered successfully!")
                            st.write(f"**Answer:** {result['answer']}")
                            
                            # Optionally display the chunks
                            with st.expander("View relevant chunks"):
                                for i, chunk in enumerate(result["chunks"]):
                                    st.markdown(f"**Chunk {i+1}:** {chunk}")
                        else:
                            st.warning(f"No results: {result.get('message', 'Unknown error')}")
                    else:
                        st.error(f"Search failed with status code: {response.status_code}")
                except Exception as e:
                    st.error(f"Search failed: {str(e)}")

    
show()