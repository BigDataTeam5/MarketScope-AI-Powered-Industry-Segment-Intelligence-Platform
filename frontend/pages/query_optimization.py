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

async def query_marketing_knowledge(question: str, segment: str = None, top_k: int = 3) -> Dict:
    """
    Query Philip Kotler's Marketing Management book for relevant content chunks and 
    generate a comprehensive report using the LiteLLM model.
    """
    try:
        st.info("Retrieving relevant marketing content...")

        # Import required dependencies
        from agents.custom_mcp_client import MCPClient

        # Create a direct connection to the market_analysis service
        market_analysis_client = MCPClient("market_analysis")

        # Query the marketing book using the client
        query_result = market_analysis_client.invoke_sync(
            "query_marketing_book",
            {"query": question, "top_k": top_k}
        )
        
        # Store the query_result in session_state for later use when displaying sources
        st.session_state['last_query_result'] = query_result

        # Process the query result
        chunks = []
        chunk_ids = []

        if query_result and isinstance(query_result, dict) and "chunks" in query_result:
            chunks = query_result.get("chunks", [])
            for chunk in chunks:
                if isinstance(chunk, dict) and "content" in chunk:
                    chunk_ids.append(chunk.get("chunk_id", "unknown"))

        # If no chunks were found, inform the user
        if not chunks:
            st.error("No relevant marketing content found. Please try a different query.")
            return {"status": "error", "response": "No relevant content found."}

        # Format chunk content for the LLM
        chunk_content = "\n\n".join([chunk.get("content", "") for chunk in chunks])

        # Generate report using the LiteLLM model
        st.info("Generating comprehensive marketing report...")

        # Prepare the prompt for the LLM
        segment_context = f"for the {segment} segment" if segment else ""
        prompt = f"""
        Based on the following excerpts from Philip Kotler's Marketing Management book, 
        provide a comprehensive answer to this question: "{question}" {segment_context}.
        
        EXCERPTS FROM MARKETING MANAGEMENT:
        {chunk_content}
        
        Please provide a well-structured professional report with:
        1. A concise executive summary
        2. Key marketing concepts explained clearly
        3. Practical applications for {segment if segment else "businesses"}
        4. Strategic recommendations based on Kotler's insights
        5. Clear section headings for readability
        
        Format your response as a professional business report without mentioning "chunks" or raw data sources in the main text.
        """

        # Use direct litellm.completion instead of the wrapper to avoid compatibility issues
        try:
            import litellm
            
            # Direct litellm call using the older API style that works with your version
            response = litellm.completion(
                model=Config.DEFAULT_MODEL if hasattr(Config, 'DEFAULT_MODEL') else "gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a marketing expert specializing in professional report generation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            # Extract content from the response
            if response and hasattr(response, 'choices') and len(response.choices) > 0:
                if hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'content'):
                    final_response = response.choices[0].message.content.strip()
                else:
                    # Fallback in case the response structure is different
                    final_response = str(response.choices[0]).strip()
            else:
                final_response = "Error: No response generated from the language model."
                
        except Exception as llm_error:
            st.error(f"Error generating marketing report: {str(llm_error)}")
            # Provide fallback content instead of completely failing
            final_response = f"""
            # Marketing Analysis for {question}
            
            ## Error Generating Full Report
            
            There was an error generating the complete marketing analysis using the language model.
            However, the system successfully found relevant content from Kotler's Marketing Management book.
            
            ### Retrieved Marketing Content:
            
            {chunk_content[:500]}... [Content truncated]
            
            Please try again later or contact support for assistance.
            """

        # Add source references at the end if not already present
        if not any(f"Source {i+1}" in final_response for i in range(len(chunk_ids))):
            final_response += "\n\n## References\n"
            for i, chunk_id in enumerate(chunk_ids):
                final_response += f"Source {i+1}: {chunk_id}\n"

        return {
            "status": "success",
            "response": final_response,
            "chunk_ids": chunk_ids
        }

    except Exception as e:
        st.error(f"Error processing marketing query: {str(e)}")
        return {"status": "error", "response": f"Failed to process marketing query: {str(e)}"}
    
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
            # Use our new specialized function that directly connects to the query_marketing_book tool
            try:
                # Execute search asynchronously using the specialized marketing knowledge function
                result = asyncio.run(query_marketing_knowledge(
                    question=user_query,
                    segment=segment,
                    top_k=5  # Get top 5 chunks for comprehensive coverage
                ))
                
                if result and result.get("status") == "success":
                    st.session_state['search_result'] = result
                    st.success("✅ Found relevant marketing knowledge!")
                else:
                    st.error(f"❌ Search failed: {result.get('response', 'Unknown error')}")
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
        
        # Get the referenced chunks directly from the result
        if "chunk_ids" in result:
            chunk_ids = result["chunk_ids"]
        
        # Clean up the response text - remove the chunk IDs from the main report
        clean_response = response_text
        if chunk_ids:
            # Remove the chunk ID lines and any "References" section at the end
            clean_response = re.sub(r"\n+## References\s*\n+Chunk ID: [A-Za-z0-9-]+(\n+Chunk ID: [A-Za-z0-9-]+)*", "", response_text)
            clean_response = re.sub(r"\nSource \d+: [A-Za-z0-9_-]+", "", clean_response)
            clean_response = re.sub(r"\nChunk ID: [A-Za-z0-9-]+", "", clean_response)
        
        # Format the report with professional styling
        st.markdown("## Marketing Knowledge Report")
        st.markdown("---")
        st.markdown(clean_response)
        
        # Try to get the actual chunks content
        chunk_contents = []
        try:
            # Get chunks from the query result if available
            if 'last_query_result' in st.session_state and isinstance(st.session_state['last_query_result'], dict):
                query_result = st.session_state['last_query_result']
                if "chunks" in query_result:
                    chunks = query_result.get("chunks", [])
                    for chunk in chunks:
                        if isinstance(chunk, dict) and "content" in chunk:
                            chunk_contents.append({
                                "id": chunk.get("chunk_id", "unknown"),
                                "content": chunk.get("content", "No content available")
                            })
        except NameError:
            # If query_result is not defined, try to fetch chunks again
            try:
                # Create a direct connection to get the chunks
                from agents.custom_mcp_client import MCPClient
                market_analysis_client = MCPClient("market_analysis")
                for chunk_id in chunk_ids:
                    try:
                        chunk_content = market_analysis_client.invoke_sync(
                            "fetch_s3_chunk",
                            {"chunk_id": chunk_id}
                        )
                        if chunk_content and not isinstance(chunk_content, str):
                            chunk_contents.append({"id": chunk_id, "content": str(chunk_content)})
                        elif chunk_content:
                            chunk_contents.append({"id": chunk_id, "content": chunk_content})
                    except Exception as chunk_error:
                        st.warning(f"Could not fetch content for chunk {chunk_id}: {str(chunk_error)}")
            except Exception as e:
                st.warning(f"Could not fetch chunk contents: {str(e)}")
        
        # Move source information to an expandable section at the bottom
        with st.expander("View Source Information"):
            st.markdown("### Referenced Sources")
            
            # Display chunks with their content if available
            if chunk_contents:
                for i, chunk_data in enumerate(chunk_contents):
                    st.markdown(f"**Source {i+1}:** {chunk_data['id']}")
                    st.text_area(
                        f"Content from {chunk_data['id']}", 
                        value=chunk_data['content'],
                        height=200,
                        disabled=True,
                        key=f"source_content_{i}"
                    )
                    st.markdown("---")  # Add a separator between sources
            elif chunk_ids:
                for i, chunk_id in enumerate(chunk_ids):
                    st.markdown(f"**Source {i+1}:** {chunk_id}")
            else:
                st.markdown("No specific sources referenced.")
                
            if sources:
                st.markdown("### Related Materials")
                for title, url in sources:
                    st.markdown(f"- [{title}]({url})")



# Call the main function
show()
