import streamlit as st
import sys
import os

# Add the root directory to the Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# Import from config package
from config import Config
from frontend.utils import process_query, sidebar

def show():
    sidebar()
    st.title("🏠 Home - MarketScope AI")

    # Access selected model and use Config to get the model name
    model_id = st.session_state.get("selected_model", Config.DEFAULT_MODEL)
    model_name = Config.get_model_config(model_id).get("name", model_id)
    st.write(f"Using model: **{model_name}**")

    st.markdown("""
    Welcome to the **MarketScope AI Platform**!  
    This app uses AI agents to help businesses:
    - Understand market segmentation
    - Compare their products with competitors
    - Optimize marketing strategies
    - Analyze industry segments

    Navigate using the sidebar to get started.
    """)

    st.subheader("Select Healthcare Segment")
    segment = st.selectbox(
        "Choose a market segment:",
        [
            "Skincare Segment",
            "Diagnostic Segment",
            "Supplement Segment",
            "OTC Pharmaceutical Segment",
            "Wearable Segment"
        ]
    )

    st.subheader("Upload Your Sales Data")
    uploaded_file = st.file_uploader(
        "Upload your sales data (CSV or Excel format):",
        type=["csv", "xlsx", "xls"]
    )

    if uploaded_file is not None:
        st.success(f"File '{uploaded_file.name}' uploaded successfully!")
        # You can add code here to process the uploaded file if needed

show()