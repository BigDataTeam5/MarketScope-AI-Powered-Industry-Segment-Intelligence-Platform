"""
Query Optimization Page for MarketScope AI
Allows users to query Philip Kotler's Marketing Management book for relevant content
and get optimized marketing strategies
"""
import streamlit as st
import sys
import os
import requests
import json
import asyncio
import time

# Add root directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.config import Config
from frontend.utils import sidebar
from agents.unified_agent import unified_agent

# Set page config to full width
st.set_page_config(
    page_title="Marketing Knowledge Query",
    page_icon="🔍",
    layout="wide"
)

# Function to process queries through the unified agent
async def process_marketing_query(query, segment=None):
    """Process a query about marketing knowledge using the unified agent"""
    try:
        result = await unified_agent.process_query(
            query=query,
            use_case="marketing_strategy",
            segment=segment,
            context={"source": "marketing_book"}
        )
        return result
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
        return None

def show():
    """Show the query optimization page"""
    sidebar()
    
    st.title("📚 Marketing Knowledge Query")
    
    # Get segment selection from session state
    segment = st.session_state.get("selected_segment", None)
    
    if not segment:
        st.warning("Please select a segment on the Home page first.")
        return
    
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
            top_k = st.number_input("Number of results", min_value=1, max_value=5, value=3)
            search_button = st.button("Search Knowledge", type="primary")
        
        # Process search when button is clicked
        if search_button and user_query:
            with st.spinner("Searching for relevant marketing knowledge..."):
                search_query = f"Query Philip Kotler's Marketing Management book using the query_marketing_book tool for: {user_query}"
                
                # Execute search asynchronously
                result = asyncio.run(process_marketing_query(search_query, segment))
                
                if result and result.get("status") == "success":
                    st.session_state['search_result'] = result
                    st.success("Found relevant marketing knowledge!")
                else:
                    st.error(f"Search failed: {result.get('message', 'Unknown error')}")
        
        # Display search results
        if 'search_result' in st.session_state:
            result = st.session_state['search_result']
            response_text = result.get("response", "")
            
            # Extract chunk citations if available
            chunks = []
            import re
            chunk_pattern = r"Chunk ID: ([A-Za-z0-9-]+)"
            chunk_ids = re.findall(chunk_pattern, response_text)
            
            st.markdown("### Search Results")
            st.markdown(response_text)
            
            if chunk_ids:
                with st.expander("View Source Chunks"):
                    for chunk_id in chunk_ids:
                        st.markdown(f"**Chunk ID:** {chunk_id}")
                        # Note: In a real implementation, you might want to fetch the actual chunks
    
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
            generate_button = st.button("Generate Strategy", type="primary")
        
        additional_context = st.text_area(
            "Additional Context (optional)",
            height=100,
            placeholder="Add any specific details about your product, target market, or business goals",
            key="strategy_context"
        )
        
        # Process strategy generation when button is clicked
        if generate_button and product_type:
            with st.spinner(f"Generating marketing strategy for {segment}..."):
                context_text = f" with context: {additional_context}" if additional_context else ""
                
                strategy_query = f"Generate a marketing strategy for {product_type} in the {segment} as a {competitive_position}{context_text}. Use the generate_segment_strategy tool with segment_name='{segment}', product_type='{product_type}', and competitive_position='{competitive_position}'."
                
                # Execute strategy generation asynchronously
                result = asyncio.run(process_marketing_query(strategy_query, segment))
                
                if result and result.get("status") == "success":
                    st.session_state['strategy_result'] = result
                    st.success("Strategy generated successfully!")
                else:
                    st.error(f"Strategy generation failed: {result.get('message', 'Unknown error')}")
        
        # Display strategy results
        if 'strategy_result' in st.session_state:
            result = st.session_state['strategy_result']
            response_text = result.get("response", "")
            
            st.markdown("### Marketing Strategy")
            
            # Check if there are sources to reference
            sources_section = ""
            if "Sources:" in response_text:
                parts = response_text.split("Sources:")
                main_content = parts[0]
                sources_section = "Sources:" + parts[1] if len(parts) > 1 else ""
                st.markdown(main_content)
                
                if sources_section:
                    with st.expander("View Sources"):
                        st.markdown(sources_section)
            else:
                st.markdown(response_text)
            
            # Download strategy as text
            strategy_text = response_text
            st.download_button(
                label="Download Strategy as Text",
                data=strategy_text,
                file_name=f"{segment}_marketing_strategy.txt",
                mime="text/plain",
                key="download_strategy"
            )

# Call the main function
show()
