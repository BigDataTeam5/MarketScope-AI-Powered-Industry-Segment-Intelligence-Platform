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
    
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Segment Analysis", "Product Trends Visualization", "Data Upload"])
    
    with tab1:
        # Example segmentation form
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
        st.subheader("Product Trends Visualization")
        st.markdown("""
        Visualize product trends and analyze market data from various segments.
        Select a segment and visualization type to generate insights from market data.
        """)
        
        # Get the selected segment from session state
        selected_segment = st.session_state.get("selected_segment", "Diagnostic Segment")
        
        # Allow user to select visualization type
        viz_type = st.selectbox(
            "Select visualization type",
            options=["price_comparison", "rating_analysis", "price_distribution"],
            format_func=lambda x: {
                "price_comparison": "Price Comparison",
                "rating_analysis": "Rating Analysis",
                "price_distribution": "Price Distribution"
            }.get(x, x)
        )
        
        # Optional table name input
        table_name = st.text_input("Snowflake table name (optional):", "")
        
        # Button to generate visualization
        if st.button("Generate Visualization"):
            with st.spinner(f"Generating {viz_type} visualization..."):
                # Call segment MCP server for visualization - direct connection
                result = create_visualization_from_mcp(selected_segment, viz_type, table_name)
                
                if result.get("status") == "success" and "chart_data" in result:
                    # Render chart
                    render_chart(result["chart_data"])
                    
                    # Show additional information
                    with st.expander("View Data Details"):
                        st.json(result.get("data", {}))
                        
                        # Show statistics if available
                        if "statistics" in result:
                            st.subheader("Statistics")
                            st.write(result["statistics"])
                        
                        # Show sample size
                        if "sample_size" in result:
                            st.write(f"**Sample size:** {result['sample_size']} items")
                        
                        if "products_analyzed" in result:
                            if isinstance(result["products_analyzed"], int):
                                st.write(f"**Products analyzed:** {result['products_analyzed']}")
                            else:
                                st.write(f"**Products analyzed:** {len(result['products_analyzed'])}")
                else:
                    st.error(f"Failed to generate visualization: {result.get('message', 'Unknown error')}")
                    
                    # Check server status directly
                    segment_url = get_mcp_server_url("segment")
                    try:
                        status_res = requests.get(f"{segment_url}/health", timeout=1)
                        if status_res.status_code != 200:
                            st.warning("⚠️ Segment MCP Server may not be running properly.")
                    except:
                        st.warning("⚠️ Segment MCP Server connection failed.")
                    
                    # Provide guidance on next steps
                    st.warning("""
                    To generate visualizations, you need to:
                    1. First upload sales data using the Data Upload tab
                    2. Make sure the MCP server for your selected segment is running
                    3. The segment must have product trends data available
                    """)
    
    

show()