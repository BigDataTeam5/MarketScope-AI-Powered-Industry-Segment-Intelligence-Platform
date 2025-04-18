import streamlit as st
import requests
import json
import sys
import os

# Add root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(layout="wide")

st.title("MarketScope Connection Test")

# Test specific endpoints
st.header("Test Direct Connections")

server_url = st.text_input("Server URL", "http://localhost:8014")
endpoint = st.text_input("Endpoint", "/health")

if st.button("Test Connection"):
    try:
        response = requests.get(f"{server_url}{endpoint}", timeout=5)
        st.write(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            try:
                st.json(response.json())
            except:
                st.text(response.text)
        else:
            st.error(f"Error: {response.text}")
    except Exception as e:
        st.error(f"Connection Error: {str(e)}")

# Test market size endpoint
st.header("Test Market Size Analysis")

selected_segment = st.selectbox(
    "Select segment:", 
    ["Skin Care Segment", "Healthcare - Diagnostic", "Pharmaceutical"]
)

if st.button("Test Market Size Analysis"):
    with st.spinner("Testing..."):
        try:
            # Try direct endpoint
            url = f"{server_url}/direct/analyze_market_size"
            st.write(f"Trying: {url}")
            
            response = requests.post(
                url, 
                params={"segment": selected_segment},
                timeout=10
            )
            
            if response.status_code == 200:
                st.success("Success!")
                st.json(response.json())
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Error: {str(e)}")