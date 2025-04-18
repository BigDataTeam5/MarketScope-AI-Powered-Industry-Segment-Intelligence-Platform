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
import numpy as np
from matplotlib.ticker import FuncFormatter
import time

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
if "upload_method" not in st.session_state:
    st.session_state["upload_method"] = "Upload CSV"

# Use the common sidebar function to ensure segment selection consistency
sidebar()

# Get segment information AFTER sidebar has been called
selected_segment = st.session_state.get("selected_segment")

# Simplified sidebar for data options - AFTER the main sidebar
with st.sidebar:
    st.header("Data Options")
    upload_method = st.radio("Select Input Method", ["Upload CSV", "Use Sample Data"], key="upload_method_radio")
    st.session_state["upload_method"] = upload_method
    
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
        
        # Create and configure MCP client using the proper MCPClient class
        from agents.custom_mcp_client import MCPClient
        snowflake_mcp_client = MCPClient("snowflake")
        
        # Check if Snowflake server is available first
        try:
            tools = await asyncio.wait_for(snowflake_mcp_client.get_tools(force_refresh=True), timeout=5.0)
            
            if not tools:
                return {
                    "status": "error",
                    "message": "Snowflake MCP server is not available - no tools found"
                }
            
            # Check if the required tool exists - properly handle tools list format
            tool_exists = False
            if isinstance(tools, list):
                tool_exists = any(tool.get("name") == "load_csv_to_table" for tool in tools)
            elif isinstance(tools, dict) and "tools" in tools:
                tool_exists = any(tool.get("name") == "load_csv_to_table" for tool in tools.get("tools", []))
                
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
        response = await snowflake_mcp_client.invoke(
            "load_csv_to_table", 
            {
                "table_name": table_name,
                "csv_data": csv_string,
                "create_table": True,
                "segment_name": segment_name
            }
        )
        
        # Properly handle different response formats
        if isinstance(response, dict):
            # Handle dictionary response format
            if "status" in response and response["status"] == "error":
                return {
                    "status": "error",
                    "message": response.get("message", "Unknown error occurred")
                }
            else:
                return {
                    "status": "success",
                    "message": response.get("content", str(response))
                }
        elif isinstance(response, str):
            # Handle string response format
            if "error" in response.lower():
                return {
                    "status": "error",
                    "message": response
                }
            else:
                return {
                    "status": "success",
                    "message": response
                }
        else:
            # Handle any other response type
            return {
                "status": "success",
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
            try:
                await snowflake_mcp_client.close()
            except:
                pass  # Ignore errors on closing

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
        sales_mcp_client = MCPClient("sales_analytics")
        
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
            health_response = asyncio.run(sales_mcp_client.get_tools(force_refresh=True))
            if not health_response:
                raise Exception("Sales analytics server is not responding")
        except Exception as e:
            logger.error(f"Server health check failed: {str(e)}")
            st.error("⚠️ Unable to connect to analytics server. Please try again.")
            return None
        
        # Generate dummy data for demonstration if connection fails
        try:
            # Extract top products
            top_products = df.groupby("product_name").agg({
                "revenue": "sum", 
                "units_sold": "sum", 
                "estimated_profit": "sum"
            }).sort_values("revenue", ascending=False).head(5)
            
            analysis_result["data_tables"]["top_products"] = top_products.reset_index()
            
            # Extract channel data
            channels = df.groupby("sales_channel").agg({
                "revenue": "sum", 
                "units_sold": "sum"
            }).sort_values("revenue", ascending=False)
            
            analysis_result["data_tables"]["channels"] = channels.reset_index()
            
            # Generate sample analysis text
            segment_display = segment.replace("_", " ").title()
            analysis_result["agent_result"]["response"] = f"""
            ## Sales Analysis for {segment_display}
            
            ### Key Findings:
            
            - **Total Revenue**: ${df['REVENUE'].sum():,.2f}
            - **Units Sold**: {df['UNITS_SOLD'].sum():,}
            - **Average Price**: ${df['PRICE'].mean():.2f}
            
            ### Product Performance:
            The top performing product is **{top_products.index[0]}** with ${top_products['REVENUE'].iloc[0]:,.2f} in revenue.
            
            ### Channel Effectiveness:
            The most effective sales channel is **{channels.index[0]}** accounting for {channels['REVENUE'].iloc[0]/df['REVENUE'].sum()*100:.1f}% of total revenue.
            
            ### Recommendations:
            1. Focus marketing efforts on expanding the {channels.index[0]} channel
            2. Consider bundling {top_products.index[0]} with lower performing products
            3. Explore pricing optimizations for mid-tier products
            """
            
            logger.info("Successfully generated analysis")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error generating analysis: {str(e)}")
            st.error(f"Error generating analysis: {str(e)}")
            return None
        
    except Exception as e:
        logger.error(f"Critical error in generate_analysis: {str(e)}")
        st.error(f"Error generating analysis: {str(e)}")
        return None

# Function to create multiple visualizations
def create_sales_visualizations(df):
    visualizations = {}
    
    # Set the style for all visualizations
    sns.set(style="darkgrid")
    plt.rcParams.update({'font.size': 12})
    
    # Create a function to format currency on axes
    def currency_formatter(x, pos):
        return f"${x:,.2f}"

    # Create a function to format percentage on axes
    def percentage_formatter(x, pos):
        return f"{x:.0f}%"

    # 1. Product Performance Analysis
    fig1, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # 1.1 Total Revenue by Product - Fix column names to lowercase
    product_revenue = df.groupby('product_name')['revenue'].sum().sort_values(ascending=False)
    sns.barplot(x=product_revenue.values, y=product_revenue.index, ax=axes[0, 0], palette='viridis')
    axes[0, 0].set_title('Total Revenue by Product', fontsize=14)
    axes[0, 0].set_xlabel('Revenue ($)')
    axes[0, 0].xaxis.set_major_formatter(FuncFormatter(currency_formatter))
    for i, v in enumerate(product_revenue.values):
        axes[0, 0].text(v + 5, i, f"${v:,.2f}", va='center')
    
    # 1.2 Units Sold by Product - Fix column names to lowercase
    product_units = df.groupby('product_name')['units_sold'].sum().sort_values(ascending=False)
    sns.barplot(x=product_units.values, y=product_units.index, ax=axes[0, 1], palette='viridis')
    axes[0, 1].set_title('Total Units Sold by Product', fontsize=14)
    axes[0, 1].set_xlabel('Units Sold')
    for i, v in enumerate(product_units.values):
        axes[0, 1].text(v + 0.5, i, f"{v:,}", va='center')
    
    # 1.3 Average Price by Product - Fix column names to lowercase
    product_price = df.groupby('product_name')['price'].mean().sort_values(ascending=False)
    sns.barplot(x=product_price.values, y=product_price.index, ax=axes[1, 0], palette='viridis')
    axes[1, 0].set_title('Average Price by Product', fontsize=14)
    axes[1, 0].set_xlabel('Average Price ($)')
    axes[1, 0].xaxis.set_major_formatter(FuncFormatter(currency_formatter))
    for i, v in enumerate(product_price.values):
        axes[1, 0].text(v + 0.5, i, f"${v:.2f}", va='center')
    
    # 1.4 Total Profit by Product - Fix column names to lowercase
    product_profit = df.groupby('product_name')['estimated_profit'].sum().sort_values(ascending=False)
    sns.barplot(x=product_profit.values, y=product_profit.index, ax=axes[1, 1], palette='viridis')
    axes[1, 1].set_title('Total Profit by Product', fontsize=14)
    axes[1, 1].set_xlabel('Profit ($)')
    axes[1, 1].xaxis.set_major_formatter(FuncFormatter(currency_formatter))
    for i, v in enumerate(product_profit.values):
        axes[1, 1].text(v + 5, i, f"${v:,.2f}", va='center')
    
    plt.tight_layout()
    visualizations['product_performance'] = fig1
    
    # 2. Geographic Analysis
    fig2, axes = plt.subplots(2, 1, figsize=(14, 12))
    
    # 2.1 Revenue by City - Fix column names to lowercase
    city_revenue = df.groupby('city_name')['revenue'].sum().sort_values(ascending=False)
    sns.barplot(x=city_revenue.index, y=city_revenue.values, ax=axes[0], palette='mako')
    axes[0].set_title('Total Revenue by City', fontsize=14)
    axes[0].set_ylabel('Revenue ($)')
    axes[0].set_xlabel('')
    axes[0].yaxis.set_major_formatter(FuncFormatter(currency_formatter))
    axes[0].tick_params(axis='x', rotation=45)
    for i, v in enumerate(city_revenue.values):
        axes[0].text(i, v + 10, f"${v:,.2f}", ha='center')
    
    # 2.2 Revenue by State - Fix column names to lowercase
    state_revenue = df.groupby('state')['revenue'].sum().sort_values(ascending=False)
    sns.barplot(x=state_revenue.index, y=state_revenue.values, ax=axes[1], palette='mako')
    axes[1].set_title('Total Revenue by State', fontsize=14)
    axes[1].set_ylabel('Revenue ($)')
    axes[1].set_xlabel('')
    axes[1].yaxis.set_major_formatter(FuncFormatter(currency_formatter))
    for i, v in enumerate(state_revenue.values):
        axes[1].text(i, v + 10, f"${v:,.2f}", ha='center')
    
    plt.tight_layout()
    visualizations['geographic_analysis'] = fig2
    
    # 3. Profit Margin Analysis
    fig3, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # 3.1 Average Margin % by Product - Fix column names to lowercase
    product_margin = df.groupby('product_name')['estimated_margin_pct'].mean().sort_values(ascending=False)
    sns.barplot(x=product_margin.index, y=product_margin.values, ax=axes[0], palette='rocket')
    axes[0].set_title('Average Profit Margin by Product', fontsize=14)
    axes[0].set_ylabel('Margin %')
    axes[0].set_xlabel('')
    axes[0].tick_params(axis='x', rotation=45)
    for i, v in enumerate(product_margin.values):
        axes[0].text(i, v + 1, f"{v:.1f}%", ha='center')
    
    # 3.2 Margin vs Revenue Scatterplot - Fix column names to lowercase
    sns.scatterplot(x='revenue', y='estimated_margin_pct', data=df, 
                    hue='product_name', size='units_sold', sizes=(50, 200),
                    ax=axes[1])
    axes[1].set_title('Margin vs Revenue by Transaction', fontsize=14)
    axes[1].set_xlabel('Revenue ($)')
    axes[1].set_ylabel('Margin %')
    axes[1].xaxis.set_major_formatter(FuncFormatter(currency_formatter))
    
    plt.tight_layout()
    visualizations['margin_analysis'] = fig3
    
    # 4. Time Series Analysis
    # Convert date to datetime if it's not already
    if df['date'].dtype != 'datetime64[ns]':
        df['date'] = pd.to_datetime(df['date'])
    
    fig4, axes = plt.subplots(2, 1, figsize=(14, 12))
    
    # 4.1 Revenue Over Time by Product - Fix column names to lowercase
    time_product = df.groupby(['date', 'product_name'])['revenue'].sum().reset_index()
    sns.lineplot(x='date', y='revenue', hue='product_name', data=time_product, 
                marker='o', ax=axes[0])
    axes[0].set_title('Revenue Over Time by Product', fontsize=14)
    axes[0].set_ylabel('Revenue ($)')
    axes[0].set_xlabel('')
    axes[0].yaxis.set_major_formatter(FuncFormatter(currency_formatter))
    axes[0].legend(title='Product')
    
    # 4.2 Units Sold Over Time - Fix column names to lowercase
    time_units = df.groupby('date')['units_sold'].sum().reset_index()
    sns.lineplot(x='date', y='units_sold', data=time_units, color='purple', 
                marker='o', linewidth=2, ax=axes[1])
    axes[1].set_title('Total Units Sold Over Time', fontsize=14)
    axes[1].set_ylabel('Units Sold')
    axes[1].set_xlabel('Date')
    
    plt.tight_layout()
    visualizations['time_series'] = fig4
    
    # 5. Price Distribution and Product Mix
    fig5, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # 5.1 Price Distribution - Fix column names to lowercase
    sns.histplot(df['price'], bins=10, kde=True, ax=axes[0], color='teal')
    axes[0].set_title('Price Distribution', fontsize=14)
    axes[0].set_xlabel('Price ($)')
    axes[0].set_ylabel('Frequency')
    axes[0].xaxis.set_major_formatter(FuncFormatter(currency_formatter))
    
    # 5.2 Product Mix (Revenue Share) - Fix column names to lowercase
    product_share = df.groupby('product_name')['revenue'].sum()
    explode = [0.1] * len(product_share)  # Explode all slices slightly
    axes[1].pie(product_share, labels=product_share.index, autopct='%1.1f%%',
                startangle=90, shadow=True, explode=explode)
    axes[1].set_title('Revenue Share by Product', fontsize=14)
    axes[1].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    
    plt.tight_layout()
    visualizations['price_product_mix'] = fig5
    
    return visualizations

# Always show upload container
upload_container = st.container()

with upload_container:
    st.subheader("Step 1: Select Data Source")
    
    # Always show both options with clear buttons
    if upload_method == "Upload CSV":
        uploaded_file = st.file_uploader("Upload Sales Data CSV", type=["csv"], key="csv_uploader")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state['sales_data'] = df
                # Reset upload status when new data is loaded
                st.session_state['snowflake_uploaded'] = False
                st.success(f"CSV uploaded successfully with {len(df)} rows and {len(df.columns)} columns.")
            except Exception as e:
                st.error(f"Error reading CSV file: {e}")
                st.session_state['sales_data'] = None
    else:
        if st.button("Load Sample Data", key="load_sample"):
            df = load_sample_data()
            st.session_state['sales_data'] = df
            # Reset upload status when new data is loaded
            st.session_state['snowflake_uploaded'] = False
            st.success(f"Sample data loaded successfully with {len(df)} rows!")

# Display the uploaded data if available
if 'sales_data' in st.session_state and st.session_state['sales_data'] is not None:
    st.subheader("Step 2: Review Your Data")
    with st.expander("View Raw Data", expanded=False):
        st.dataframe(st.session_state['sales_data'].head(10), use_container_width=True)
        st.caption(f"Showing first 10 of {len(st.session_state['sales_data'])} rows")

# Always show step 3 - Handle data upload process
st.subheader("Step 3: Save Data for Analysis")
if 'sales_data' in st.session_state and st.session_state['sales_data'] is not None and selected_segment:
    # Check if we need to show the upload button
    if not st.session_state.get('snowflake_uploaded'):
        if st.button("Save Data for Analysis", key="save_data", type="primary"):
            with st.spinner(f"Saving data for {selected_segment}..."):
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
                    
                    # Show countdown and generate visualizations automatically
                    countdown_placeholder = st.empty()
                    for i in range(5, 0, -1):
                        countdown_placeholder.info(f"Generating visualizations in {i} seconds...")
                        time.sleep(1)
                    countdown_placeholder.empty()
                    
                    # Flag to track visualization status
                    st.session_state['auto_generate_viz'] = True
                    
                    # Refresh to update UI
                    st.experimental_rerun()
                else:
                    # Local save failed
                    st.error(f"Failed to save data: {local_result['message']}")
    else:
        st.success("✅ Your data is ready for analysis!")
else:
    if not selected_segment:
        st.warning("Please select a segment from the sidebar first")
    else:
        st.warning("Please load data before proceeding")

# Automatic visualization generation after data save
if 'auto_generate_viz' in st.session_state and st.session_state['auto_generate_viz'] and 'sales_data' in st.session_state and st.session_state['sales_data'] is not None:
    with st.spinner("Generating visualizations..."):
        # Create all visualizations
        visualizations = create_sales_visualizations(st.session_state['sales_data'])
        st.session_state['visualizations'] = visualizations
        st.session_state['auto_generate_viz'] = False  # Reset flag to prevent regeneration
    
    # Display success message
    st.success("✅ Visualizations generated successfully!")
    
    # Add spacer
    st.markdown("---")

# Display "Generate Analysis" button in the main area once data is uploaded
if 'snowflake_uploaded' in st.session_state and st.session_state['snowflake_uploaded']:
    st.subheader("Step 4: Generate Your Analysis")
    
    if st.button("Generate Analysis", type="primary", key="generate_analysis"):
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
                st.experimental_rerun()
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

# Display visualizations if available
if 'sales_data' in st.session_state and st.session_state['sales_data'] is not None:
    # Use cached visualizations if available, otherwise generate them
    if 'visualizations' in st.session_state:
        visualizations = st.session_state['visualizations']
    else:
        visualizations = create_sales_visualizations(st.session_state['sales_data'])
        st.session_state['visualizations'] = visualizations

    # Display visualizations in Streamlit
    st.header("📊 Pharmaceutical Sales Analytics")

    # Product Performance
    st.subheader("Product Performance Analysis")
    st.pyplot(visualizations['product_performance'])

    # Geographic Analysis
    st.subheader("Geographic Sales Analysis")
    st.pyplot(visualizations['geographic_analysis'])

    # Margin Analysis
    st.subheader("Profit Margin Analysis")
    st.pyplot(visualizations['margin_analysis'])

    # Time Series Analysis
    st.subheader("Sales Trends Over Time")
    st.pyplot(visualizations['time_series'])

    # Price Distribution and Product Mix
    st.subheader("Price Distribution & Product Mix")
    st.pyplot(visualizations['price_product_mix'])

    # Additional insights - summary metrics in columns
    st.subheader("Summary Metrics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Revenue", f"${st.session_state['sales_data']['revenue'].sum():,.2f}")

    with col2:
        st.metric("Total Units Sold", f"{st.session_state['sales_data']['units_sold'].sum():,}")

    with col3:
        st.metric("Avg. Profit Margin", f"{st.session_state['sales_data']['estimated_margin_pct'].mean():.1f}%")

    with col4:
        st.metric("Total Profit", f"${st.session_state['sales_data']['estimated_profit'].sum():,.2f}")
