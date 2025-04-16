import streamlit as st
import sys
import os
import pandas as pd

# Add root directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import from config package
from config import Config
from frontend.utils import process_query,sidebar

def show():
    """
    Display the Product Comparison page content
    """
    # Get model information from session state
    selected_model = st.session_state.get("selected_model", Config.DEFAULT_MODEL)
    temperature = st.session_state.get("temperature", Config.DEFAULT_TEMPERATURE)
    
    # Get model name for display
    model_name = Config.get_model_config(selected_model).get("name", selected_model)
    
    st.title("Product Comparison Analysis")
    st.write(f"Using model: {model_name} with temperature: {temperature}")
    
    st.markdown("""
    Compare your product against competitors to understand strengths, weaknesses, and market positioning.
    Enter details about your product and up to three competitors.
    """)
    
    # Create tabs for different comparison sections
    tab1, tab2 = st.tabs(["Basic Comparison", "Detailed Analysis"])
    
    with tab1:
        with st.form("basic_comparison_form"):
            st.subheader("Basic Product Comparison")
            
            # Your product details
            st.markdown("### Your Product")
            your_product_name = st.text_input("Product Name", "Your Product")
            your_product_desc = st.text_area("Brief Description", 
                                           "Enter a short description of your product's key features and benefits")
            
            # Competitor details
            st.markdown("### Competitor Products")
            
            # Create input fields for up to 3 competitors
            competitor_names = []
            competitor_descs = []
            
            col1, col2 = st.columns(2)
            
            with col1:
                competitor_names.append(st.text_input("Competitor 1 Name", "Competitor A"))
                competitor_descs.append(st.text_area("Competitor 1 Description", "Key features of competitor 1"))
            
            with col2:
                competitor_names.append(st.text_input("Competitor 2 Name", "Competitor B"))
                competitor_descs.append(st.text_area("Competitor 2 Description", "Key features of competitor 2"))
            
            # Specific comparison focus
            comparison_focus = st.text_input("Specific Comparison Focus", 
                                           "e.g., Price vs. Performance, Feature Completeness, Ease of Use")
            
            submitted = st.form_submit_button("Compare Products")
            
            if submitted and your_product_name and your_product_desc and any(competitor_names) and any(competitor_descs):
                # Build the query for the API
                competitors_text = ""
                for i, (name, desc) in enumerate(zip(competitor_names, competitor_descs)):
                    if name and desc:
                        competitors_text += f"Competitor {i+1}: {name}\n{desc}\n\n"
                
                prompt = f"""Perform a competitive analysis between the following products:

                Your Product: {your_product_name}
                {your_product_desc}

                {competitors_text}

                Focus specifically on: {comparison_focus}

                Provide:
                1. A side-by-side comparison of key features
                2. Strengths and weaknesses of each product
                3. Market positioning analysis
                4. Strategic recommendations to improve competitive position
                """
                
                # Show analysis in progress
                with st.spinner("Comparing products..."):
                    # Prepare config for API
                    api_config = {
                        "model": selected_model,
                        "temperature": temperature
                    }
                    
                    # Process the query using the backend API
                    response = process_query(prompt, "marketing_management", api_config)
                    
                    if response and not response.startswith("Error"):
                        st.success("Comparison complete!")
                        st.subheader("Competitive Analysis")
                        st.markdown(response)
                    else:
                        st.error(f"Failed to generate comparison: {response}")
    
    with tab2:
        st.info("The detailed analysis feature provides in-depth comparison with market data integration.")
        st.markdown("### Coming soon: Advanced Feature & Market Analysis")
        st.markdown("""
        Future capabilities will include:
        - Integration with market research data
        - Price point optimization analysis
        - Feature priority recommendations
        - Customer sentiment analysis from reviews
        """)
        
        st.image("https://placehold.co/600x400?text=Feature+Coming+Soon", use_column_width=True)

show()