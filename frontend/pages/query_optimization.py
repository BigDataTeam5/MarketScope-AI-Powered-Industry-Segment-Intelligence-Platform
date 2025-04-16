import streamlit as st
import sys
import os

# Add root directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import from config package
from config import Config
from frontend.utils import process_query,sidebar

def show():
    """
    Display the Query Optimization page
    """
    # Get model information from session state
    selected_model = st.session_state.get("selected_model", Config.DEFAULT_MODEL)
    temperature = st.session_state.get("temperature", Config.DEFAULT_TEMPERATURE)
    
    # Get model name for display
    model_name = Config.get_model_config(selected_model).get("name", selected_model)
    
    st.title("Marketing Query Optimization")
    st.write(f"Using model: {model_name} with temperature: {temperature}")
    
    st.markdown("""
    This tool helps you optimize marketing queries and messages for different platforms and target audiences.
    Enter your original message or query, and get optimized versions for your specific needs.
    """)
    
    with st.form("query_optimization_form"):
        original_query = st.text_area("Original Marketing Message/Query", 
                                    "Enter your marketing message or query here.", 
                                    height=150)
        
        col1, col2 = st.columns(2)
        
        with col1:
            platform = st.selectbox("Target Platform", 
                                  ["Website", "Email", "Social Media", "Search Ads", "Print", "Television/Radio"])
            
            audience = st.multiselect("Target Audience", 
                                    ["B2B Decision Makers", "Small Business Owners", "Enterprise Clients", 
                                     "Young Adults (18-25)", "Adults (25-40)", "Seniors (60+)", "Technical Users"])
        
        with col2:
            tone = st.select_slider("Message Tone", 
                                  options=["Formal", "Professional", "Neutral", "Conversational", "Casual"])
            
            style_focus = st.radio("Style Focus", 
                                 ["Direct/Informative", "Persuasive", "Educational", "Emotional"])
            
        character_limit = st.slider("Character Limit (if applicable)", 0, 2000, 500)
        
        col3, col4 = st.columns(2)
        with col3:
            submitted = st.form_submit_button("Optimize Message")
        
        if submitted and original_query:
            # Build the query for the API
            audiences_text = ", ".join(audience) if audience else "general audience"
            
            prompt = f"""Optimize the following marketing message:

            Original message: "{original_query}"
            
            Please optimize this message for:
            - Platform: {platform}
            - Target audience: {audiences_text}
            - Tone: {tone}
            - Style: {style_focus}
            - Maximum character count: {character_limit}
            
            Provide:
            1. An optimized version of the message within the character limit
            2. Explanation of key changes made
            3. 2-3 alternate versions with slight variations
            4. SEO keywords if applicable
            """
            
            # Show optimization in progress
            with st.spinner("Optimizing your message..."):
                # Prepare config for API
                api_config = {
                    "model": selected_model,
                    "temperature": temperature
                }
                
                # Process the query using the backend API
                response = process_query(prompt, "marketing_management", api_config)
                
                if response and not response.startswith("Error"):
                    st.success("Optimization complete!")
                    st.subheader("Optimized Marketing Message")
                    st.markdown(response)
                else:
                    st.error(f"Failed to optimize message: {response}")
    
    # Add a section for A/B testing suggestions
    st.markdown("---")
    st.subheader("A/B Testing Suggestions")
    st.markdown("""
    After optimizing your message, consider testing different versions to see what resonates best with your audience.
    This can help refine your marketing approach and improve conversion rates.
    """)
    
    if st.button("Generate A/B Testing Plan"):
        if not original_query:
            st.warning("Please enter a message and optimize it first.")
        else:
            # Build A/B testing prompt
            prompt = f"""Create an A/B testing plan for marketing messages targeting {platform}.
            
            Original message: "{original_query}"
            
            Provide:
            1. What elements to test (headline, CTA, imagery suggestions, etc.)
            2. How to measure success (metrics to track)
            3. Recommended sample size and duration
            4. How to interpret potential results
            """
            
            with st.spinner("Generating A/B testing plan..."):
                # Prepare config for API
                api_config = {
                    "model": selected_model,
                    "temperature": temperature
                }
                
                # Process the query using the backend API
                response = process_query(prompt, "marketing_management", api_config)
                
                if response and not response.startswith("Error"):
                    st.markdown(response)
                else:
                    st.error(f"Failed to generate A/B testing plan: {response}")

show()