import streamlit as st
from frontend.utils import sidebar
import sys
import os

# Add root directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set page config to full width
st.set_page_config(layout="wide")

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
    
    # --- Upload sales product data ---
    st.markdown("### Upload Your Sales/Product Data")
    uploaded_file = st.file_uploader(
        "Upload a CSV or Excel file containing your sales or product data for analysis.",
        type=["csv", "xlsx"]
    )
    if uploaded_file is not None:
        st.success(f"File '{uploaded_file.name}' uploaded successfully!")
        # You can add further processing here if needed

    # Display current segment if selected
    if st.session_state.get("selected_segment"):
        st.info(f"Currently analyzing: **{st.session_state.selected_segment}**")
    else:
        st.warning("Please select a segment from the sidebar to begin analysis.")

show()