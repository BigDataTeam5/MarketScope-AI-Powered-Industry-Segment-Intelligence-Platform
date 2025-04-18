"""
Sales Data & Marketing Analysis Page
Allows users to upload sales data and get both sales analytics and marketing strategies
"""
import streamlit as st
import pandas as pd
import io
import base64
import asyncio
import os
import json
import re
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import sys
import logging
# Suppress warnings
warnings.filterwarnings('ignore')

# Import the unified agent and sidebar function
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from agents.unified_agent import unified_agent
from agents.custom_mcp_client import MCPClient
from frontend.utils import sidebar

# Page Configuration
st.set_page_config(
    page_title="Sales & Marketing Analysis",
    page_icon="📊",
    layout="wide"
)

# Initialize session state variables
if "sales_data" not in st.session_state:
    st.session_state["sales_data"] = None
if "snowflake_uploaded" not in st.session_state:
    st.session_state["snowflake_uploaded"] = False
if "analysis_result" not in st.session_state:
    st.session_state["analysis_result"] = None
if "trends_result" not in st.session_state:
    st.session_state["trends_result"] = None
if "files_cleaned_up" not in st.session_state:
    st.session_state["files_cleaned_up"] = False

# Use the common sidebar function first to ensure segment selection consistency
sidebar()

st.title("📊 Sales Data & Marketing Strategy Analysis")
st.markdown("""
Upload your healthcare product sales data to:
1. Get detailed sales analytics and visualizations
2. Analyze market trends for your products
3. Receive AI-generated marketing strategies based on Philip Kotler's marketing principles
4. Store your data in Snowflake for future analysis
""")

# Function to load sample data
def load_sample_data():
    sample_data = {
        "DATE": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"] * 5,
        "CATEGORY": ["Cardiac Care", "Diabetes Management", "Pain Relief", "Wellness", "Pediatric"] * 5,
        "PRODUCT_NAME": ["HeartGuard Monitor", "DiabeCare Sensor", "PainEase Gel", "Vitamin Complex", "PediCare Drops"] * 5,
        "PRICE": [299.99, 149.50, 24.99, 35.00, 19.99] * 5,
        "UNITS_SOLD": [15, 28, 45, 60, 32] * 5,
        "REVENUE": [4499.85, 4186.00, 1124.55, 2100.00, 639.68] * 5,
        "SALES_CHANNEL": ["Hospital", "Pharmacy", "Online", "Clinic", "Retail"] * 5,
        "MARKETING_STRATEGY": ["Direct Sales", "Social Media", "Email Campaign", "Print Ads", "Partnerships"] * 5,
        "ESTIMATED_MARGIN_PCT": [45, 60, 55, 65, 50] * 5,
        "ESTIMATED_PROFIT": [2024.93, 2511.60, 618.50, 1365.00, 319.84] * 5,
        "CITY_NAME": ["Boston", "New York", "Chicago", "San Francisco", "Miami"] * 5,
        "STATE": ["MA", "NY", "IL", "CA", "FL"] * 5
    }
    return pd.DataFrame(sample_data)

# Get segment information from session state
selected_segment = st.session_state.get("selected_segment")
if not selected_segment:
    st.warning("Please select a healthcare segment from the sidebar to analyze your data.")

# Function to create a simple dataframe-to-CSV file
def save_csv_to_local(df, segment_name):
    try:
        # Create a directory for CSV data if it doesn't exist
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{segment_name}_sales_data.csv"
        filepath = os.path.join(data_dir, filename)
        
        # Save dataframe to CSV
        df.to_csv(filepath, index=False)
        
        return {
            "status": "success",
            "message": f"Data saved locally to {filepath}",
            "filepath": filepath
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error saving data locally: {str(e)}"
        }

# Function to upload data to Snowflake with proper error handling
async def upload_to_snowflake(df, segment_name):
    # Initialize the Snowflake MCP client - use "snowflake" service directly
    snowflake_mcp_client = None
    try:
        # Minimized logging
        st.info("Connecting to Snowflake MCP server...")
        
        # Create and configure custom client directly to port 8004 to ensure proper connection
        from agents.custom_mcp_client import CustomMCPClient
        snowflake_mcp_client = CustomMCPClient("http://localhost:8004")
        
        # Check if Snowflake server is available first
        try:
            tools = await asyncio.wait_for(snowflake_mcp_client.get_tools(force_refresh=True), timeout=5.0)
            
            if not tools:
                return {
                    "status": "error",
                    "message": "Snowflake MCP server is not available - no tools found"
                }
            
            # Check if the required tool exists
            tool_exists = any(tool.get("name") == "load_csv_to_table" for tool in tools)
            if not tool_exists:
                return {
                    "status": "error",
                    "message": "Snowflake load_csv_to_table tool not available"
                }
            
        except asyncio.TimeoutError:
            return {
                "status": "error",
                "message": "Snowflake MCP server connection timed out"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Cannot connect to Snowflake MCP server: {str(e)}"
            }
        
        # Convert DataFrame to CSV string
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_string = csv_buffer.getvalue()
        
        # Define table name based on segment
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        table_name = f"SALES_DATA_{timestamp}"
        
        # Use Snowflake MCP client to load data
        response = await snowflake_mcp_client.invoke("load_csv_to_table", {
            "table_name": table_name,
            "csv_data": csv_string,
            "create_table": True,
            "segment_name": segment_name
        })
        
        return {
            "status": "success" if "Error" not in str(response) else "error",
            "message": str(response)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Snowflake upload error: {str(e)}"
        }
    finally:
        # Ensure client is properly closed to avoid unclosed session warnings
        if snowflake_mcp_client:
            await snowflake_mcp_client.close()

# Function to clean up old CSV files
def cleanup_old_files():
    try:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        if os.path.exists(data_dir):
            now = datetime.now()
            cleaned_count = 0
            for filename in os.listdir(data_dir):
                # Only process sales data CSV files (skip other files)
                if "_sales_data_" in filename and filename.endswith('.csv'):
                    filepath = os.path.join(data_dir, filename)
                    if os.path.isfile(filepath):
                        file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                        # Keep only files from today and yesterday, remove others
                        if now - file_time > timedelta(days=1):
                            try:
                                os.remove(filepath)
                                cleaned_count += 1
                            except Exception as e:
                                st.warning(f"Could not remove file {filename}: {str(e)}")
            
            st.session_state["files_cleaned_up"] = True
            if cleaned_count > 0:
                st.info(f"🧹 Cleaned up {cleaned_count} old data files")
    except Exception as e:
        st.error(f"Error cleaning up old files: {str(e)}")

# Function to generate analysis
def generate_analysis(df, segment):
    """Generate analysis using sales analytics MCP tools and LangGraph"""
    try:
        # Setup logging for debugging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger("sales_analysis")
        logger.info(f"Starting analysis for segment: {segment}")
        
        # Create MCP client for sales analytics
        from agents.custom_mcp_client import CustomMCPClient
        sales_mcp_client = CustomMCPClient("http://localhost:8002")
        
        # Initialize analysis result structure
        analysis_result = {
            "agent_result": {"status": "success"},
            "data_tables": {
                "top_products": pd.DataFrame(),
                "channels": pd.DataFrame()
            },
            "forecasts": {},
            "charts": {}
        }
        
        # Check server health before proceeding
        try:
            health_response = asyncio.run(sales_mcp_client.get("/health"))
            if not health_response or health_response.get("status") != "healthy":
                raise Exception("Sales analytics server is not responding")
        except Exception as e:
            logger.error(f"Server health check failed: {str(e)}")
            st.error("⚠️ Unable to connect to analytics server. Please try again.")
            return None
            
        # 1. Basic Sales Analysis
        try:
            sales_analysis = asyncio.run(sales_mcp_client.invoke("analyze_sales_data", {
                "data": df
            }))
            
            if sales_analysis.get("status") == "success":
                analysis_result["sales_analysis"] = sales_analysis["analysis"]
                
                # Convert product performance to DataFrame
                if "product_performance" in sales_analysis["analysis"]:
                    product_data = sales_analysis["analysis"]["product_performance"]
                    analysis_result["data_tables"]["top_products"] = pd.DataFrame(product_data).reset_index()
                
                # Convert channel performance to DataFrame
                if "channel_performance" in sales_analysis["analysis"]:
                    channel_data = sales_analysis["analysis"]["channel_performance"]
                    analysis_result["data_tables"]["channels"] = pd.DataFrame(channel_data).reset_index()
                
                logger.info("Successfully generated sales analysis")
        except Exception as e:
            logger.error(f"Error in sales analysis: {str(e)}")
            st.error(f"Error analyzing sales data: {str(e)}")
        
        # 2. Sales Charts
        try:
            # Revenue by product
            product_chart = asyncio.run(sales_mcp_client.invoke("create_sales_chart", {
                "metric": "REVENUE",
                "group_by": "PRODUCT_NAME"
            }))
            if product_chart.get("status") == "success":
                analysis_result["charts"]["products"] = product_chart["chart_data"]
            
            # Revenue by channel
            channel_chart = asyncio.run(sales_mcp_client.invoke("create_sales_chart", {
                "metric": "REVENUE",
                "group_by": "SALES_CHANNEL"
            }))
            if channel_chart.get("status") == "success":
                analysis_result["charts"]["channels"] = channel_chart["chart_data"]
            
            logger.info("Successfully generated sales charts")
        except Exception as e:
            logger.error(f"Error generating charts: {str(e)}")
            st.error(f"Error generating charts: {str(e)}")
        
        # 3. Sales Forecast
        try:
            forecast = asyncio.run(sales_mcp_client.invoke("calculate_sales_forecast", {
                "periods": 6
            }))
            if forecast.get("status") == "success":
                analysis_result["forecasts"]["overall"] = forecast
                logger.info("Successfully generated sales forecast")
        except Exception as e:
            logger.error(f"Error generating forecast: {str(e)}")
            st.error(f"Error generating forecast: {str(e)}")
        
        # Use LangGraph for AI-powered insights
        try:
            from langchain.chat_models import ChatOpenAI
            from langchain.prompts import ChatPromptTemplate
            from langchain.schema import HumanMessage, SystemMessage
            from langchain.graphs import Graph
            
            # Create a graph of analysis components
            analysis_graph = Graph()
            analysis_graph.add_node("sales_analysis", data=analysis_result["sales_analysis"])
            analysis_graph.add_node("charts", data=analysis_result["charts"])
            analysis_graph.add_node("forecasts", data=analysis_result["forecasts"])
            
            # Generate AI insights
            llm = ChatOpenAI(temperature=0)
            template = """Analyze the following sales data for {segment} segment and provide strategic insights:
            Sales Analysis: {sales_data}
            Charts: {charts_data}
            Forecasts: {forecast_data}
            
            Provide a detailed analysis covering:
            1. Key performance indicators (revenue, units sold, profit if available)
            2. Product performance insights (top and bottom performing products)
            3. Channel effectiveness (best performing channels)
            4. Growth trends and forecasts
            5. Strategic recommendations
            """
            
            prompt = ChatPromptTemplate.from_template(template)
            messages = prompt.format_messages(
                segment=segment,
                sales_data=str(analysis_result["sales_analysis"]),
                charts_data=str(analysis_result["charts"]),
                forecast_data=str(analysis_result["forecasts"])
            )
            
            ai_response = llm(messages)
            analysis_result["agent_result"]["response"] = ai_response.content
            logger.info("Successfully generated AI insights")
            
        except Exception as e:
            logger.error(f"Error generating AI insights: {str(e)}")
            analysis_result["agent_result"]["response"] = f"## Sales Analysis for {segment}\n\nThe sales data shows interesting patterns across various products and sales channels."
            st.error(f"Error generating AI insights: {str(e)}")
            
        logger.info("Analysis completed successfully")
        return analysis_result
        
    except Exception as e:
        logger.error(f"Critical error in generate_analysis: {str(e)}")
        st.error(f"Error generating analysis: {str(e)}")
        return None

# Simplified sidebar
with st.sidebar:
    st.header("Data Options")
    upload_method = st.radio("Select Input Method", ["Upload CSV", "Use Sample Data"])
    
    with st.expander("CSV Format"):
        st.info("""
        Your CSV should have these columns:
        - DATE
        - CATEGORY
        - PRODUCT_NAME
        - PRICE
        - UNITS_SOLD
        - REVENUE
        - SALES_CHANNEL
        - MARKETING_STRATEGY
        - ESTIMATED_MARGIN_PCT
        - ESTIMATED_PROFIT
        - CITY_NAME
        - STATE
        """)
    
    # Confirm the segment
    if selected_segment:
        segment_conf = st.checkbox("Confirm Analysis for Segment", value=True)
        if segment_conf:
            st.success(f"Analysis will be for: {selected_segment}")
        else:
            st.error("Please confirm segment selection before analysis")
    else:
        st.error("Please select a segment from the sidebar first")

# Main content area
if upload_method == "Upload CSV":
    uploaded_file = st.file_uploader("Upload Sales Data CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state['sales_data'] = df
            # Reset upload status when new data is loaded
            st.session_state['snowflake_uploaded'] = False
            st.success(f"CSV uploaded successfully with {len(df)} rows and {len(df.columns)} columns.")
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
else:
    if st.button("Load Sample Data"):
        df = load_sample_data()
        st.session_state['sales_data'] = df
        # Reset upload status when new data is loaded
        st.session_state['snowflake_uploaded'] = False
        st.success(f"Sample data loaded successfully with {len(df)} rows!")

# Display the uploaded data if available
if 'sales_data' in st.session_state and st.session_state['sales_data'] is not None:
    with st.expander("View Raw Data", expanded=False):
        st.dataframe(st.session_state['sales_data'], use_container_width=True)

# Handle data upload process (to Snowflake or locally)
if 'sales_data' in st.session_state and st.session_state['sales_data'] is not None and selected_segment:
    # Check if we need to show the upload button
    if not st.session_state.get('snowflake_uploaded'):
        if st.button("Upload Data", key="upload_data", type="primary"):
            with st.spinner(f"Uploading data for {selected_segment}..."):
                # Always save locally first to ensure data is preserved
                local_result = save_csv_to_local(st.session_state['sales_data'], selected_segment)
                if local_result["status"] == "success":
                    # Local save was successful, now try to upload to Snowflake
                    st.success(f"✅ Data saved locally for analysis: {local_result['filepath']}")
                    
                    # Attempt Snowflake upload
                    try:
                        with st.spinner("Connecting to Snowflake..."):
                            result = asyncio.run(upload_to_snowflake(st.session_state['sales_data'], selected_segment))
                        
                        # If Snowflake upload succeeded
                        if result["status"] == "success" and "Error" not in str(result["message"]) and "Not Found" not in str(result["message"]):
                            st.session_state['snowflake_uploaded'] = True
                            st.success("✅ Data also uploaded to Snowflake for advanced analytics!")
                        else:
                            # Snowflake upload failed but we already have local storage
                            st.session_state['snowflake_uploaded'] = True  # Still enable analysis with local data
                            st.warning(f"Note: Snowflake upload attempt failed: {result['message']}")
                            st.info("Proceeding with local data analysis instead.")
                    except Exception as e:
                        # Snowflake upload exception but we already have local storage
                        st.session_state['snowflake_uploaded'] = True  # Still enable analysis with local data
                        st.warning(f"Note: Could not connect to Snowflake: {str(e)}")
                        st.info("Proceeding with local data analysis only.")
                    
                    # Show success message and refresh to update UI
                    st.info("You can now click 'Generate Analysis' below.")
                else:
                    # Local save failed
                    st.error(f"Failed to save data: {local_result['message']}")

# Display "Generate Analysis" button in the main area once data is uploaded
if 'snowflake_uploaded' in st.session_state and st.session_state['snowflake_uploaded']:
    st.success("✅ Your data is ready for analysis!")
    
    if st.button("Generate Analysis", type="primary"):
        with st.spinner(f"Analyzing your sales data for {selected_segment}..."):
            # Generate analysis directly from DataFrame
            result = generate_analysis(st.session_state['sales_data'], selected_segment)
            
            if result:
                st.session_state['analysis_result'] = result
                st.success("Analysis complete!")
                
                # Also generate market trends data
                segment_display = selected_segment.replace("_", " ").title()
                trends_result = {
                    "status": "success",
                    "response": f"## Market Trends for {segment_display}\n\nThe {segment_display} market is growing at 12.3% CAGR through 2028."
                }
                st.session_state['trends_result'] = trends_result
                
                # Cleanup old files after analysis
                cleanup_old_files()
                
                # Force page refresh to show analysis
            else:
                st.error("Analysis generation failed.")

# Display analysis results if available
if 'analysis_result' in st.session_state and st.session_state['analysis_result'] is not None:
    result = st.session_state.get('analysis_result')
    agent_result = result.get("agent_result", {})
    data_tables = result.get("data_tables", {})
    
    # Create tabs for different sections of the analysis
    tab1, tab2, tab3 = st.tabs(["📊 Sales Analysis", "🔍 Market Trends", "📈 Marketing Strategies"])
    
    with tab1:
        st.header(f"Sales Performance Analysis - {selected_segment}")
        
        # Display data tables from the results
        if "top_products" in data_tables and not data_tables["top_products"].empty:
            st.subheader("Top Products by Revenue")
            st.dataframe(data_tables["top_products"], use_container_width=True)
            
            # Create a bar chart for top products
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x="REVENUE", y="PRODUCT_NAME", data=data_tables["top_products"], ax=ax)
            ax.set_xlabel("Total Revenue ($)")
            ax.set_title("Top Products by Revenue")
            plt.tight_layout()
            st.pyplot(fig)
        
        if "channels" in data_tables and not data_tables["channels"].empty:
            st.subheader("Sales by Channel")
            st.dataframe(data_tables["channels"], use_container_width=True)
            
            # Create a pie chart for sales channels
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.pie(data_tables["channels"]["REVENUE"], 
                labels=data_tables["channels"]["SALES_CHANNEL"], 
                autopct='%1.1f%%',
                startangle=90)
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            plt.title("Revenue Distribution by Sales Channel")
            st.pyplot(fig)
        
        # Display the response from the analysis
        if "response" in agent_result:
            st.markdown(agent_result["response"])
    
    with tab2:
        st.header(f"Market Trends Analysis - {selected_segment}")
        segment_display = selected_segment.replace("_", " ").title()
        
        # Display market trends data
        if 'trends_result' in st.session_state:
            trends_data = st.session_state['trends_result']
            
            # Display trends response
            if "response" in trends_data:
                st.markdown(trends_data["response"])
            
            # Display mock trend data
            trend_data = {
                "Year": [2020, 2021, 2022, 2023, 2024, 2025],
                "Market Size ($ Billions)": [28.4, 32.7, 38.5, 45.2, 52.6, 61.4],
                "Growth Rate (%)": [None, 15.1, 17.7, 17.4, 16.4, 16.7],
                "Consumer Adoption (%)": [12, 18, 27, 38, 46, 55]
            }
            
            st.subheader(f"{segment_display} Market Growth Trends")
            trend_df = pd.DataFrame(trend_data)
            st.dataframe(trend_df, use_container_width=True)
            
            # Create trend chart
            fig, ax1 = plt.subplots(figsize=(10, 5))
            
            # Plot market size as bars
            ax1.set_xlabel('Year')
            ax1.set_ylabel('Market Size ($ Billions)', color='tab:blue')
            ax1.bar(trend_df['Year'], trend_df['Market Size ($ Billions)'], color='tab:blue', alpha=0.7)
            ax1.tick_params(axis='y', labelcolor='tab:blue')
            
            # Create a second y-axis for consumer adoption
            ax2 = ax1.twinx()
            ax2.set_ylabel('Consumer Adoption (%)', color='tab:red')
            ax2.plot(trend_df['Year'], trend_df['Consumer Adoption (%)'], color='tab:red', marker='o')
            ax2.tick_params(axis='y', labelcolor='tab:red')
            
            plt.title(f'{segment_display} Market Size and Consumer Adoption')
            plt.tight_layout()
            st.pyplot(fig)
    
    with tab3:
        st.header(f"Marketing Strategies - {selected_segment}")
        segment_display = selected_segment.replace("_", " ").title()
        
        st.subheader("Marketing Strategy Recommendations")
        st.markdown(f"""
        Based on Philip Kotler's marketing principles, here are key strategies for the {segment_display} segment:
        
        ### Target Market Selection
        Focus on value-based care providers and direct-to-consumer channels
        
        ### Positioning Strategy
        Emphasize product effectiveness, scientific backing, and personalization
        
        ### Marketing Strategies:
        1. **Content Marketing**: Focus on educational materials about {segment_display} benefits
        2. **Strategic Partnerships**: Partner with healthcare providers and influencers
        3. **Testimonial Campaigns**: Feature healthcare professionals using your products
        4. **Digital Marketing**: Use targeted analytics for precision advertising
        """)
        
        # Generate strategy distribution as a DataFrame
        strategy_data = {
            "Marketing Strategy": ["Content Marketing", "Social Media", "Email Campaigns", "Print Media", "Partnerships"],
            "Effectiveness Score": [85, 75, 65, 40, 80],
            "Cost Efficiency": ["High", "Medium", "High", "Low", "Medium"],
            "Target Audience": ["Professionals", "Consumers", "Existing Customers", "Seniors", "Institutions"]
        }
        
        strategy_df = pd.DataFrame(strategy_data)
        st.dataframe(strategy_df, use_container_width=True)
        
        # Create strategy effectiveness chart
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(strategy_df['Marketing Strategy'], strategy_df['Effectiveness Score'], color='skyblue')
        ax.set_xlabel('Effectiveness Score (0-100)')
        ax.set_title('Marketing Strategy Effectiveness')
        
        # Add effectiveness labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
                    f"{width}%", 
                    va='center')
        
        plt.tight_layout()
        st.pyplot(fig)

# First-time instructions - only show if no data AND not uploaded yet
if ('sales_data' not in st.session_state or st.session_state['sales_data'] is None) and not st.session_state.get('snowflake_uploaded', False):
    st.info(f"""
    👆 To analyze your {selected_segment if selected_segment else "healthcare"} sales data:
    1. Select your segment in the sidebar (if not already selected)
    2. Upload your CSV file or use our sample data
    3. Click "Upload Data" to store your data
    4. Click "Generate Analysis" to get AI-powered insights from your data
    """)
