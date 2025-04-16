import streamlit as st
import sys
import os
from typing import Dict, Any

# Add root directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import from config package
from config import Config
from frontend.utils import process_query,sidebar


def show():
    """
    Display the Market Segmentation page content
    """
    # Get model information from session state
    selected_model = st.session_state.get("selected_model", Config.DEFAULT_MODEL)
    temperature = st.session_state.get("temperature", Config.DEFAULT_TEMPERATURE)
    
    # Get model name for display
    model_name = Config.get_model_config(selected_model).get("name", selected_model)
    
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
                # Prepare config for API
                api_config = {
                    "model": selected_model,
                    "temperature": temperature
                }
                
                # Process the query using the backend API
                response = process_query(prompt, "marketing_management", api_config)
                
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
            response = process_query(user_question, "marketing_management", api_config)
            
            if response and not response.startswith("Error"):
                st.chat_message("assistant").markdown(response)
            else:
                st.chat_message("assistant").error(f"Failed to generate response: {response}")

show()