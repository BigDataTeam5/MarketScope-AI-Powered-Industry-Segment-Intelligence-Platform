"""
Query Optimization Page for MarketScope AI
Allows users to query Philip Kotler's Marketing Management book for relevant content
and get optimized marketing strategies
"""
import streamlit as st
import sys
import os
import json
import asyncio
import pandas as pd
import re
from typing import Dict, Any, List

# Add root directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.config import Config
from frontend.utils import sidebar
from agents.unified_agent import unified_agent
# from mcp_servers.snowflake_mcp_server import SnowflakeMCPServer

# Set page config to full width
st.set_page_config(
    page_title="Marketing Knowledge Query",
    page_icon="🔍",
    layout="wide"
)

async def process_marketing_query(query: str, segment: str = None, use_context: bool = False, context_data: Dict = None) -> Dict:
    """Process a query about marketing knowledge using the unified agent"""
    try:
        # First verify that the unified agent is ready
        if unified_agent.llm is None:
            st.error("LLM not initialized. Please check your configuration.")
            return None
            
        # Add context for better response quality
        context = {
            "source": "marketing_book",
            "segment": segment,
            "query_type": "knowledge_search"
        }
        
        if use_context and context_data:
            context.update(context_data)
        
        # Process query with retries
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                result = await unified_agent.process_query(
                    query=query,
                    use_case="marketing_strategy",
                    segment=segment,
                    context=context
                )
                
                if result and result.get("status") == "success":
                    return result
                elif attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    st.error(f"Failed to process query after {max_retries} attempts")
                    return None
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    st.error(f"Error processing query: {str(e)}")
                    return None
                    
    except Exception as e:
        st.error(f"Error initializing query processing: {str(e)}")
        return None

async def get_sales_data(segment: str) -> pd.DataFrame:
    """Get sales data from Snowflake for a specific segment"""
    try:
        # Query Snowflake through MCP server
        query = f"""
        SELECT * FROM SALES_DATA 
        WHERE segment = '{segment}'
        ORDER BY date DESC
        LIMIT 1000
        """
        
        result = await unified_agent.mcp_client.invoke(
            "execute_query", 
            {"query": query}
        )
        
        if isinstance(result, dict) and "data" in result:
            return pd.DataFrame(result["data"])
        return pd.DataFrame()
        
    except Exception as e:
        st.error(f"Error fetching sales data: {str(e)}")
        return pd.DataFrame()

def show():
    """Show the query optimization page"""
    sidebar()
    
    st.title("📚 Marketing Knowledge Query")
    
    # Get segment selection from session state
    segment = st.session_state.get("selected_segment", None)
    
    if not segment:
        st.warning("Please select a segment on the Home page first.")
        return
        
    # Initialize session state for results
    if "search_result" not in st.session_state:
        st.session_state["search_result"] = None
    if "strategy_result" not in st.session_state:
        st.session_state["strategy_result"] = None
    
    st.markdown(f"""
    ## Query Marketing Management Knowledge for {segment}
    
    This tool allows you to access relevant marketing knowledge and strategies from 
    Philip Kotler's Marketing Management book, tailored to your specific segment and needs.
    
    The system will search for the most relevant content and provide strategic recommendations.
    """)
    
    # Tab selector for different query types
    query_tab, strategy_tab = st.tabs(["📖 Search Knowledge", "🎯 Generate Strategy"])
    
    with query_tab:
        st.subheader("Search Marketing Knowledge")
        
        # Search options
        col1, col2 = st.columns([3, 1])
        with col1:
            user_query = st.text_area(
                "Enter your marketing question",
                height=100,
                placeholder="e.g., What are the key segmentation strategies for healthcare markets?",
                key="search_query"
            )
        with col2:
            use_context = st.checkbox("Use Sales Context", value=True)
            search_button = st.button("Search Knowledge", type="primary", key="search_button")
        
        # Process search when button is clicked
        if search_button and user_query:
            with st.spinner("Searching for relevant marketing knowledge..."):
                search_query = f"Query Philip Kotler's Marketing Management book for: {user_query}"
                
                # Fetch sales context if needed
                context_data = None
                if use_context:
                    df = asyncio.run(get_sales_data(segment))
                    if not df.empty:
                        context_data = {
                            "sales_data": df.to_dict(orient="records"),
                            "metrics": {
                                "total_revenue": df["revenue"].sum(),
                                "avg_margin": df["estimated_margin_pct"].mean(),
                                "top_products": df.groupby("product_name")["revenue"].sum().nlargest(3).to_dict()
                            }
                        }
                
                # Execute search asynchronously
                try:
                    result = asyncio.run(process_marketing_query(
                        search_query, 
                        segment,
                        use_context=use_context,
                        context_data=context_data
                    ))
                    
                    if result and result.get("status") == "success":
                        st.session_state['search_result'] = result
                        st.success("Found relevant marketing knowledge!")
                    else:
                        st.error("Search failed. Please try again or rephrase your question.")
                except Exception as e:
                    st.error(f"Error during search: {str(e)}")
        
        # Display search results
        if 'search_result' in st.session_state and st.session_state['search_result']:
            result = st.session_state['search_result']
            response_text = result.get("response", "")
            
            # Extract chunks and sources
            chunks = []
            sources = []
            
            # Extract chunk citations and source links if available
            chunk_pattern = r"Chunk ID: ([A-Za-z0-9-]+)"
            source_pattern = r"Source: \[(.*?)\]\((.*?)\)"
            
            chunk_ids = re.findall(chunk_pattern, response_text)
            sources = re.findall(source_pattern, response_text)
            
            # Clean up the response text if needed
            clean_response = response_text
            if chunk_ids:
                clean_response = re.sub(r"\nChunk ID: [A-Za-z0-9-]+", "", response_text)
            
            st.markdown("### Search Results")
            st.markdown(clean_response)
            
            # Show source sections in expandable areas
            if chunk_ids:
                with st.expander("View Source Chunks"):
                    for chunk_id in chunk_ids:
                        st.markdown(f"**Chunk ID:** {chunk_id}")
            
            if sources:
                with st.expander("View Related Sources"):
                    for title, url in sources:
                        st.markdown(f"- [{title}]({url})")
    
    with strategy_tab:
        st.subheader(f"Generate Strategy for {segment}")
        
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            product_type = st.text_input(
                "Product Type", 
                placeholder="e.g., Wearable glucose monitor",
                key="product_type"
            )
        with col2:
            competitive_position = st.selectbox(
                "Competitive Position",
                options=["leader", "challenger", "follower", "nicher"],
                index=1,
                key="competitive_position"
            )
        with col3:
            generate_button = st.button("Generate Strategy", type="primary", key="generate_button")
        
        # Additional context section
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Market Context")
            use_sales = st.checkbox("Include Sales Analysis", value=True)
            use_trends = st.checkbox("Include Market Trends", value=True)
        
        with col2:
            st.markdown("### Business Context")
            additional_context = st.text_area(
                "Additional Business Context",
                height=100,
                placeholder="Add specific details about your product, target market, or business goals",
                key="strategy_context"
            )
        
        # Process strategy generation when button is clicked
        if generate_button and product_type:
            with st.spinner(f"Generating marketing strategy for {segment}..."):
                # Fetch sales and market data
                context_data = {}
                
                if use_sales:
                    df = asyncio.run(get_sales_data(segment))
                    if not df.empty:
                        context_data["sales_analysis"] = {
                            "total_revenue": df["revenue"].sum(),
                            "avg_margin": df["estimated_margin_pct"].mean(),
                            "top_products": df.groupby("product_name")["revenue"].sum().nlargest(3).to_dict(),
                            "sales_trend": df.groupby("date")["revenue"].sum().to_dict()
                        }
                
                if use_trends:
                    try:
                        # Convert async call to sync using asyncio.run
                        trends = asyncio.run(unified_agent.mcp_client.invoke(
                            "get_market_trends",
                            {"segment": segment}
                        ))
                        context_data["market_trends"] = trends
                    except Exception as e:
                        st.warning(f"Could not fetch market trends: {str(e)}")
                
                # Add business context
                if additional_context:
                    context_data["business_context"] = additional_context
                
                strategy_query = (
                    f"Generate a marketing strategy for {product_type} in the {segment} "
                    f"as a {competitive_position}. Use the generate_segment_strategy tool "
                    f"with segment_name='{segment}', product_type='{product_type}', "
                    f"and competitive_position='{competitive_position}'."
                )
                
                try:
                    # Execute strategy generation asynchronously
                    result = asyncio.run(process_marketing_query(
                        strategy_query,
                        segment,
                        use_context=True,
                        context_data=context_data
                    ))
                    
                    if result and result.get("status") == "success":
                        st.session_state['strategy_result'] = result
                        st.success("Strategy generated successfully!")
                    else:
                        st.error("Strategy generation failed. Please try again with different parameters.")
                except Exception as e:
                    st.error(f"Error generating strategy: {str(e)}")
        
        # Display strategy results
        if 'strategy_result' in st.session_state and st.session_state['strategy_result']:
            result = st.session_state['strategy_result']
            response_text = result.get("response", "")
            
            st.markdown("### Marketing Strategy")
            
            # Check if there are sections to display
            sections = response_text.split("##")
            if len(sections) > 1:
                # Display executive summary first
                st.markdown(sections[0])
                
                # Create tabs for other sections
                section_titles = [s.split("\n")[0].strip() for s in sections[1:]]
                section_tabs = st.tabs(section_titles)
                
                for tab, content in zip(section_tabs, sections[1:]):
                    with tab:
                        st.markdown(f"##{content}")
            else:
                st.markdown(response_text)
            
            # Download strategy as text or PDF
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="Download Strategy (Text)",
                    data=response_text,
                    file_name=f"{segment}_marketing_strategy.txt",
                    mime="text/plain",
                    key="download_strategy_text"
                )
            with col2:
                st.download_button(
                    label="Download Strategy (PDF)",
                    data=response_text,
                    file_name=f"{segment}_marketing_strategy.pdf",
                    mime="application/pdf",
                    key="download_strategy_pdf"
                )

# Call the main function
show()
