import streamlit as st
import requests
import time
import uuid
import os

# Try to import streamlit_shadcn_ui, but provide fallback if components fail
try:
    import streamlit_shadcn_ui as ui
    SHADCN_AVAILABLE = True
except:
    SHADCN_AVAILABLE = False

# API endpoint
API_URL = "http://localhost:8001"

# Page configuration
st.set_page_config(
    page_title="MarketScope AI",
    page_icon="🔍",
    layout="wide",
)

# Add custom CSS for a modern look
st.markdown("""
<style>
    .main {
        padding: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #f0f4ff;
        border-left: 4px solid #3B82F6;
    }
    .assistant-message {
        background-color: #f0fdf4;
        border-left: 4px solid #10B981;
    }
    .modern-button {
        background-color: #3B82F6;
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: 500;
        cursor: pointer;
        transition: background-color 0.2s;
    }
    .modern-button:hover {
        background-color: #2563EB;
    }
    .modern-button:disabled {
        background-color: #94a3b8;
        cursor: not-allowed;
    }
    .modern-input {
        border: 1px solid #d1d5db;
        border-radius: 0.5rem;
        padding: 0.5rem;
    }
    .modern-card {
        border-radius: 0.5rem;
        border: 1px solid #e5e7eb;
        padding: 1rem;
        margin-bottom: 1rem;
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)

# Session management
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "query_in_progress" not in st.session_state:
    st.session_state.query_in_progress = False

# Function to extract text response
def extract_response_text(response_data):
    """Extract text response from various formats."""
    if isinstance(response_data, str):
        return response_data
        
    if isinstance(response_data, dict):
        for key in ["answer", "content", "output", "result", "text"]:
            if key in response_data:
                return response_data[key]
                
        if "messages" in response_data:
            messages = response_data.get("messages", [])
            if messages and isinstance(messages, list) and len(messages) > 0:
                last_message = messages[-1]
                if isinstance(last_message, dict) and "content" in last_message:
                    return last_message["content"]
    
    return str(response_data)

# Sidebar with minimal shadcn components
with st.sidebar:
    st.markdown("## 🔍 MarketScope AI")
    st.markdown("### AI-Powered Industry Segment Intelligence")
    
    st.markdown("### About")
    st.markdown("MarketScope AI helps you understand industry segments and market trends using AI-powered analysis.")
    
    # Sample questions - using standard Streamlit components
    with st.expander("Sample Questions"):
        st.markdown("""
        - What does Kotler suggest about segmentation strategies for pharma?
        - How should pharmaceutical companies approach market segmentation?
        - What are the key segmentation variables for healthcare products?
        """)
    
    # LangSmith tracing info
    st.markdown("### LangSmith Tracing")
    st.markdown("All queries are automatically traced in LangSmith.")
    
    project = os.getenv("LANGSMITH_PROJECT", "default")
    if SHADCN_AVAILABLE:
        try:
            ui.link_button(
                text="View Traces", 
                url=f"https://smith.langchain.com/projects/{project}/traces",
                variant="outline",
                key="langsmith_link"
            )
        except:
            st.markdown(f"[View Traces](https://smith.langchain.com/projects/{project}/traces)")
    else:
        st.markdown(f"[View Traces](https://smith.langchain.com/projects/{project}/traces)")
    
    # Direct query option
    if SHADCN_AVAILABLE:
        try:
            direct_query = ui.switch(label="Direct OpenAI query (testing only)", key="direct_query_switch")
        except:
            direct_query = st.checkbox("Direct OpenAI query (testing only)")
    else:
        direct_query = st.checkbox("Direct OpenAI query (testing only)")

# Main content area
st.title("MarketScope AI")
st.subheader("AI-Powered Industry Segment Intelligence Platform")

# Chat container with messages
st.markdown("### Conversation")
chat_container = st.container()

with chat_container:
    if not st.session_state.messages:
        st.info("Start by asking a question about market segmentation")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

# Query input and submit
st.markdown("### Ask a question")

col1, col2 = st.columns([5, 1])

with col1:
    if SHADCN_AVAILABLE:
        try:
            query = ui.input(
                placeholder="e.g., What are the key segmentation variables for healthcare products?",
                disabled=st.session_state.query_in_progress,
                key="query_input"
            )
        except:
            query = st.text_input(
                "Enter your question",
                placeholder="e.g., What are the key segmentation variables for healthcare products?",
                disabled=st.session_state.query_in_progress,
                key="query_input"
            )
    else:
        query = st.text_input(
            "Enter your question",
            placeholder="e.g., What are the key segmentation variables for healthcare products?",
            disabled=st.session_state.query_in_progress,
            key="query_input"
        )

with col2:
    if SHADCN_AVAILABLE:
        try:
            submit_btn = ui.button(
                text="Ask AI", 
                disabled=st.session_state.query_in_progress,
                variant="default",
                key="submit_btn"
            )
        except:
            submit_btn = st.button(
                "Ask AI", 
                disabled=st.session_state.query_in_progress,
                key="submit_btn"
            )
    else:
        submit_btn = st.button(
            "Ask AI", 
            disabled=st.session_state.query_in_progress,
            key="submit_btn"
        )

# Handle form submission
if submit_btn and query:
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.query_in_progress = True
    
    # Submit query to API
    try:
        with st.spinner("Processing your query..."):
            response = requests.post(
                f"{API_URL}/query",
                json={
                    "query": query, 
                    "session_id": st.session_state.session_id,
                    "direct": direct_query  # Pass the direct query option
                }
            )
            
            if response.status_code != 200:
                st.error(f"Error submitting query: {response.text}")
                st.session_state.query_in_progress = False
                st.rerun()
            
            # Poll for results
            max_retries = 60  # 2 minutes at 2-second intervals
            retries = 0
            
            while retries < max_retries:
                poll_response = requests.get(f"{API_URL}/query/{st.session_state.session_id}")
                
                if poll_response.status_code == 200:
                    result = poll_response.json()
                    
                    if result["status"] == "completed":
                        try:
                            # Extract response and add to chat
                            if "result" not in result:
                                st.error("Response missing 'result' field")
                                break
                                
                            ai_response = extract_response_text(result["result"])
                            st.session_state.messages.append({"role": "assistant", "content": ai_response})
                            break
                        except Exception as e:
                            st.error(f"Error processing response: {str(e)}")
                            break
                            
                    elif result["status"] == "error":
                        st.error(f"Error processing query: {result.get('error', 'Unknown error')}")
                        break
                    elif result["status"] == "processing":
                        # Continue polling
                        pass
                else:
                    st.error(f"Error retrieving results: {poll_response.status_code}")
                    break
                
                retries += 1
                time.sleep(2)
                
                if retries >= max_retries:
                    st.error("Query processing timed out. Please try again.")
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
    
    # Reset query in progress
    st.session_state.query_in_progress = False
    st.rerun()


def shadcn_chart_demo():
    df = pd.DataFrame({
        "category": ["A", "B", "C", "D"],
        "value": [10, 30, 20, 40]
    })
    chart = alt.Chart(df).mark_bar().encode(
        x="category",
        y="value",
        tooltip=["category", "value"]
    )

    # Convert to raw HTML
    chart_html = chart.to_html()

    # Wrap it in a ShadCN card, then add the raw HTML inside
    with ui.card(title="Altair Chart via HTML"):
        ui.html(chart_html)  # or ui.component(html=chart_html)

# System information with tabs 
st.markdown("---")
tab1, tab2, tab3 = st.tabs(["System Architecture", "System Health", "ShadCN Charts Demo"])

with tab1:
    st.markdown("### To run this application:")
    
    st.markdown("**1. Start the Pinecone S3 server:**")
    st.code("python mcp_server/pinecone_s3_server.py")
    
    st.markdown("**2. Start the API server:**")
    st.code("python mcp_server/api_server.py")
    
    st.markdown("**3. Start the Streamlit UI:**")
    st.code("streamlit run frontend/streamlit_agent_ui.py")

with tab2:
    if st.button("Check API Connection", key="health_check_btn"):
        try:
            response = requests.get(f"{API_URL}/health")
            if response.status_code == 200:
                health_data = response.json()
                st.success("API Connection: Healthy")
                
                # Display LangSmith config without debug info
                project = health_data.get("langsmith_config", {}).get("LANGSMITH_PROJECT", "")
                if project:
                    st.markdown(f"LangSmith project: `{project}`")
                    st.markdown(f"[View LangSmith Traces](https://smith.langchain.com/projects/{project}/traces)")
            else:
                st.error(f"API Connection: {response.status_code}")
        except Exception as e:
            st.error(f"API Connection Error: {str(e)}")

with tab3:
    st.markdown("## ShadCN Charts & UI Demo")

    import pandas as pd
    import altair as alt

    ############################################################################
    # 1) Line Chart Example
    ############################################################################
    # Create a small time-series dataset (Month vs Sales)
    df_line = pd.DataFrame({
        "month": pd.date_range(start="2025-01-01", periods=6, freq="M"),
        "sales": [20, 35, 30, 50, 45, 60]
    })

    line_chart = alt.Chart(df_line).mark_line(point=True).encode(
        x=alt.X("month:T", title="Month"),
        y=alt.Y("sales:Q", title="Sales"),
        tooltip=["month:T", "sales:Q"]
    ).properties(
        width=500,
        height=300,
        title="Monthly Sales (Line Chart)"
    )

    st.markdown("### Line Chart Demo")

    if SHADCN_AVAILABLE:
        try:
            with ui.card(title="Line Chart in ShadCN Card"):
                st.altair_chart(line_chart, use_container_width=True)
        except Exception as e:
            st.error(f"ShadCN line chart failed: {str(e)}")
            st.altair_chart(line_chart, use_container_width=True)
    else:
        st.markdown("ShadCN not available, falling back to standard Streamlit chart:")
        st.altair_chart(line_chart, use_container_width=True)

    st.markdown("---")

    ############################################################################
    # 2) Scatter Plot Example
    ############################################################################
    # Create a sample scatter dataset
    df_scatter = pd.DataFrame({
        "product": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
        "marketing_spend": [10, 15, 20, 12, 18, 25, 14, 20, 30],
        "sales": [40, 55, 65, 50, 65, 80, 60, 75, 95]
    })

    scatter_chart = alt.Chart(df_scatter).mark_circle(size=60).encode(
        x=alt.X("marketing_spend:Q", title="Marketing Spend (k$)"),
        y=alt.Y("sales:Q", title="Sales (k$)"),
        color="product:N",
        tooltip=["product:N", "marketing_spend:Q", "sales:Q"]
    ).properties(
        width=500,
        height=300,
        title="Marketing Spend vs Sales (Scatter Plot)"
    )

    st.markdown("### Scatter Plot Demo")

    if SHADCN_AVAILABLE:
        try:
            with ui.card(title="Scatter Plot in ShadCN Card"):
                st.altair_chart(scatter_chart, use_container_width=True)
        except Exception as e:
            st.error(f"ShadCN scatter plot failed: {str(e)}")
            st.altair_chart(scatter_chart, use_container_width=True)
    else:
        st.markdown("ShadCN not available, falling back to standard Streamlit chart:")
        st.altair_chart(scatter_chart, use_container_width=True)
