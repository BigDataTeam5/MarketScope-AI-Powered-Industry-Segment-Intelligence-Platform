import streamlit as st
import sys
import os
import requests

# Add root directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import Config
from frontend.utils import sidebar

# Set page config to full width
st.set_page_config(layout="wide")

API_URL = f"http://localhost:{Config.API_PORT}"

def show():
    sidebar()
    st.title("Marketing Query Optimization")

    # Get segment selection from session state (set in app page)
    segment = st.session_state.get("selected_segment", None)

    if not segment:
        st.warning("Please select a segment on the Home page first.")
        return

    st.write(f"**Selected Segment:** {segment}")

    st.markdown("""
    Enter your strategic question for this segment to receive an AI-powered optimization report.
    """)

    user_query = st.text_area(
        "Enter your strategic question for this segment (e.g., 'What marketing strategy should I use for this segment?')",
        height=100,
        key="query_opt_text"
    )
    submit_btn = st.button("Generate Optimization Report")

    if submit_btn:
        if not user_query.strip():
            st.error("Please enter a strategic question.")
            return

        with st.spinner("Generating optimization report..."):
            try:
                # Call the FastAPI /bookquery endpoint with hardcoded GPT-4 model
                payload = {
                    "query": user_query,
                    "segment": segment,
                    "model": "gpt-4o", 
                    "agent_type": "marketing_management"
                }
                response = requests.post(
                    f"{API_URL}/bookquery",
                    json=payload,
                    timeout=60
                )
                if response.status_code != 200:
                    st.error(f"API Error: {response.status_code} - {response.text}")
                    return

                data = response.json()
                session_id = data.get("session_id")
                status = data.get("status")
                # Poll for result if status is processing
                poll_url = f"{API_URL}/bookquery/{session_id}"
                max_retries = 30
                import time
                for i in range(max_retries):
                    poll_response = requests.get(poll_url)
                    poll_data = poll_response.json()
                    if poll_data.get("status") == "completed":
                        result = poll_data.get("result", {})
                        output = result.get("output")
                        st.markdown(f"### Optimization Report for {segment}")
                        st.markdown("#### Strategic Question:")
                        st.write(user_query)
                        st.markdown("#### AI-powered Optimization Report:")
                        st.write(output)
                        return
                    elif poll_data.get("status") == "error":
                        st.error(f"API Error: {poll_data.get('error')}")
                        return
                    time.sleep(2)
                st.error("Timed out waiting for optimization report.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

show()