import streamlit as st
import os
import pandas as pd
import json

# Set page config to full width
st.set_page_config(layout="wide", page_title="MarketScope AI", page_icon="📊")

# Add custom CSS
with open('styles.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

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

# Mock data for segments
MOCK_SEGMENTS = [
    "Healthcare IT",
    "Medical Devices",
    "Pharmaceutical",
    "Biotechnology",
    "Health Insurance"
]

def sidebar():
    """Create sidebar with configuration options"""
    with st.sidebar:
        st.title("MarketScope AI")
        
        # Initialize session state for segment
        if "selected_segment" not in st.session_state:
            st.session_state.selected_segment = None
        
        # Use mock segments for cloud deployment
        segments = MOCK_SEGMENTS
        
        st.session_state.selected_segment = st.selectbox(
            "Select Healthcare Segment",
            options=segments,
            index=segments.index(st.session_state.selected_segment) if st.session_state.selected_segment in segments else 0
        )
        
        # Display connection status
        st.markdown("### Connection Status")
        st.info("Cloud Deployment Mode - Backend services not connected")
        st.caption("This is a demo version running on Streamlit Cloud")

def show():
    """Show main app interface with market size analysis"""
    sidebar()
    
    st.title("MarketScope AI - Market Size Analysis")
    
    # Check if a segment is selected
    if st.session_state.get("selected_segment"):
        segment = st.session_state.selected_segment
        st.info(f"Currently analyzing: **{segment}**")
    
    # Demo content
    st.write("## Welcome to MarketScope AI")
    st.write("""
    This is a cloud deployment of the MarketScope AI platform. In this demo version, 
    backend services are not connected. In the full version, you would be able to:
    
    * Analyze market segments with AI
    * Process sales data
    * Generate market intelligence reports
    * Optimize queries for segment-specific insights
    """)
    
    # Sample visualization
    chart_data = pd.DataFrame({
        'Quarter': ['Q1', 'Q2', 'Q3', 'Q4'],
        'Revenue': [120, 135, 128, 150]
    })
    
    st.subheader(f"Sample {st.session_state.selected_segment} Revenue Trend")
    st.line_chart(chart_data.set_index('Quarter'))
    
    # Demo form
    st.subheader("Try a Demo Query")
    query = st.text_input("Enter a market analysis question:")
    
    if st.button("Analyze"):
        if query:
            with st.spinner("Processing query..."):
                st.write("### Analysis Result")
                st.write(f"Query: {query}")
                st.info(f"This is a simulated analysis result for the {st.session_state.selected_segment} segment. In the full version, this would return real-time AI analysis based on your query.")
        else:
            st.warning("Please enter a query to analyze.")

show()
