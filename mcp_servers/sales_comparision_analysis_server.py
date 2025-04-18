"""
Sales Analytics MCP Server
Provides tools for analyzing sales data and trends with competitor comparison
"""
import pandas as pd
import json
import os
import io
import base64
import logging
import matplotlib.pyplot as plt
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from mcp.server.fastmcp import FastMCP
import uvicorn
from typing import Dict, Any, List, Optional
from sentence_transformers import SentenceTransformer
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sales_comparision_mcp_server")

# Import configuration
from config.config import Config

# Import Pinecone for vector search
import pinecone
#from pinecone import ServerlessSpec

# Import for Snowflake connection
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas

# Import OpenAI for embeddings
from openai import OpenAI

# Initialize the OpenAI client
openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)

# Initialize the SentenceTransformer model for embeddings
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Create FastAPI app
app = FastAPI(title="Sales Comparision MCP Server")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Create MCP server
mcp_server = FastMCP("sales_comparisions")

# Initialize Pinecone connection
def init_pinecone():
    try:
        # Initialize Pinecone
        pinecone.init(api_key=Config.PINECONE_API_KEY, environment="us-east-1-gcp")  # Replace with your Pinecone environment
        
        # Use the healthcare-industry-reports index
        index_name = "healthcare-industry-reports"
        if index_name not in pinecone.list_indexes():
            raise ValueError(f"Index '{index_name}' does not exist in Pinecone.")
        
        index = pinecone.Index(index_name)
        logger.info("Pinecone connection initialized successfully")
        return index
    except Exception as e:
        logger.error(f"Error initializing Pinecone: {str(e)}")
        return None

# Initialize Snowflake connection
def get_snowflake_connection():
    try:
        conn = snowflake.connector.connect(
            user=Config.SNOWFLAKE_USER,
            password=Config.SNOWFLAKE_PASSWORD,
            account=Config.SNOWFLAKE_ACCOUNT,
            warehouse=Config.SNOWFLAKE_WAREHOUSE,
            database=Config.SNOWFLAKE_DATABASE,
            schema=Config.SNOWFLAKE_SCHEMA
        )
        logger.info("Snowflake connection established successfully")
        return conn
    except Exception as e:
        logger.error(f"Error connecting to Snowflake: {str(e)}")
        return None

# In-memory state for the server
state = {
    "uploaded_data": None,
    "analysis_results": {},
    "competitor_data": {},
    "comparative_analysis": {}
}

# Register MCP tools
@mcp_server.tool()
def analyze_sales_data(csv_data: Optional[str] = None) -> Dict[str, Any]:
    """Analyze sales data to extract key insights"""
    try:
        logger.info("Analyzing sales data")
        
        # Use provided CSV data or previously uploaded data
        df = None
        if csv_data:
            df = pd.read_csv(io.StringIO(csv_data))
            state["uploaded_data"] = df
        elif state["uploaded_data"] is not None:
            df = state["uploaded_data"]
        else:
            return {
                "status": "error",
                "message": "No data available for analysis. Please upload data first."
            }
        
        # Ensure required columns exist
        required_columns = ["PRODUCT_NAME", "REVENUE", "UNITS_SOLD"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return {
                "status": "error",
                "message": f"Missing required columns: {', '.join(missing_columns)}"
            }
        
        # Perform basic sales analysis
        total_revenue = df["REVENUE"].sum()
        total_units = df["UNITS_SOLD"].sum()
        total_profit = df["ESTIMATED_PROFIT"].sum() if "ESTIMATED_PROFIT" in df.columns else None
        
        # Product performance
        product_performance = df.groupby("PRODUCT_NAME").agg({
            "REVENUE": "sum",
            "UNITS_SOLD": "sum"
        }).sort_values("REVENUE", ascending=False).to_dict()
        
        # Channel performance if available
        channel_performance = None
        if "SALES_CHANNEL" in df.columns:
            channel_performance = df.groupby("SALES_CHANNEL").agg({
                "REVENUE": "sum",
                "UNITS_SOLD": "sum"
            }).sort_values("REVENUE", ascending=False).to_dict()
        
        # Store analysis results
        result = {
            "total_revenue": float(total_revenue),
            "total_units": int(total_units),
            "product_performance": product_performance,
            "channel_performance": channel_performance
        }
        
        if total_profit is not None:
            result["total_profit"] = float(total_profit)
        
        state["analysis_results"] = result
        
        return {
            "status": "success",
            "analysis": result
        }
    except Exception as e:
        logger.error(f"Error analyzing sales data: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

@mcp_server.tool()
def create_sales_chart(metric: str = "REVENUE", group_by: str = "PRODUCT_NAME") -> Dict[str, Any]:
    """Create a chart visualizing sales data"""
    try:
        logger.info(f"Creating sales chart: {metric} by {group_by}")
        
        # Ensure data is available
        if state["uploaded_data"] is None:
            return {
                "status": "error",
                "message": "No data available for visualization. Please upload data first."
            }
        
        df = state["uploaded_data"]
        
        # Ensure required columns exist
        if metric not in df.columns:
            return {
                "status": "error",
                "message": f"Column '{metric}' not found in data"
            }
        
        if group_by not in df.columns:
            return {
                "status": "error",
                "message": f"Column '{group_by}' not found in data"
            }
        
        # Group the data
        grouped_data = df.groupby(group_by)[metric].sum().sort_values(ascending=False)
        
        # Convert to dict for JSON serialization
        chart_data = {
            "labels": grouped_data.index.tolist(),
            "values": grouped_data.values.tolist(),
            "type": "bar"
        }
        
        return {
            "status": "success",
            "title": f"{metric} by {group_by}",
            "chart_data": chart_data,
            "data": grouped_data.reset_index().to_dict(orient="records")
        }
    except Exception as e:
        logger.error(f"Error creating sales chart: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

@mcp_server.tool()
def calculate_sales_forecast(periods: int = 6, product_name: Optional[str] = None) -> Dict[str, Any]:
    """Calculate a sales forecast based on historical data"""
    try:
        logger.info(f"Calculating sales forecast for {periods} periods")
        
        # Ensure data is available
        if state["uploaded_data"] is None:
            return {
                "status": "error",
                "message": "No data available for forecasting. Please upload data first."
            }
        
        df = state["uploaded_data"]
        
        # Ensure required columns exist
        if "DATE" not in df.columns or "REVENUE" not in df.columns:
            return {
                "status": "error",
                "message": "Required columns 'DATE' and 'REVENUE' not found in data"
            }
        
        # Convert DATE to datetime
        df["DATE"] = pd.to_datetime(df["DATE"])
        
        # Filter by product if specified
        if product_name:
            if "PRODUCT_NAME" not in df.columns:
                return {
                    "status": "error",
                    "message": "Column 'PRODUCT_NAME' not found in data"
                }
            
            if product_name not in df["PRODUCT_NAME"].values:
                return {
                    "status": "error",
                    "message": f"Product '{product_name}' not found in data"
                }
            
            df = df[df["PRODUCT_NAME"] == product_name]
        
        # Aggregate by date
        df_agg = df.groupby("DATE")["REVENUE"].sum().reset_index()
        df_agg = df_agg.sort_values("DATE")
        
        # Simple forecast using linear regression
        import numpy as np
        from sklearn.linear_model import LinearRegression
        
        # Convert dates to numeric (days since first date)
        first_date = df_agg["DATE"].min()
        df_agg["DAYS"] = (df_agg["DATE"] - first_date).dt.days
        
        # Fit linear regression
        X = df_agg["DAYS"].values.reshape(-1, 1)
        y = df_agg["REVENUE"].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Generate forecast
        last_date = df_agg["DATE"].max()
        forecast_dates = [last_date + pd.DateOffset(months=i+1) for i in range(periods)]
        forecast_days = [(date - first_date).days for date in forecast_dates]
        
        forecast_X = np.array(forecast_days).reshape(-1, 1)
        forecast_y = model.predict(forecast_X)
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            "DATE": forecast_dates,
            "FORECAST_REVENUE": forecast_y
        })
        
        # Prepare data for visualization
        historical_data = {
            "dates": df_agg["DATE"].dt.strftime("%Y-%m-%d").tolist(),
            "revenues": df_agg["REVENUE"].tolist(),
            "label": "Historical"
        }
        
        forecast_data = {
            "dates": forecast_df["DATE"].dt.strftime("%Y-%m-%d").tolist(),
            "revenues": forecast_df["FORECAST_REVENUE"].tolist(),
            "label": "Forecast"
        }
        
        # Prepare complete dataset for chart rendering in Streamlit
        chart_data = {
            "title": f"Sales Forecast - {product_name if product_name else 'All Products'}",
            "historical": historical_data,
            "forecast": forecast_data,
            "type": "line",
            "x_label": "Date",
            "y_label": "Revenue"
        }
        
        return {
            "status": "success",
            "product": product_name if product_name else "All Products",
            "forecast_periods": periods,
            "chart_data": chart_data,
            "historical_df": df_agg.to_dict(orient="records"),
            "forecast_df": forecast_df.to_dict(orient="records")
        }
    except Exception as e:
        logger.error(f"Error calculating sales forecast: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

@mcp_server.tool()
def identify_segment_leaders(segment: str, top_n: int = 3) -> Dict[str, Any]:
    """Identify the top companies in a specific market segment based on Snowflake data"""
    try:
        logger.info(f"Identifying top {top_n} leaders in the {segment} segment")
        
        # Connect to Snowflake
        conn = get_snowflake_connection()
        if not conn:
            return {
                "status": "error",
                "message": "Failed to connect to Snowflake database"
            }
        
        # Query Snowflake for top companies in segment
        cursor = conn.cursor()
        query = f"""
        SELECT 
            COMPANY_NAME, 
            MARKET_SHARE_PERCENTAGE,
            ANNUAL_REVENUE,
            ANNUAL_PROFIT,
            PROFIT_MARGIN
        FROM 
            MARKET_SEGMENTS
        WHERE 
            SEGMENT_NAME = '{segment}'
        ORDER BY 
            MARKET_SHARE_PERCENTAGE DESC
        LIMIT {top_n}
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        
        # Format results
        column_names = [desc[0] for desc in cursor.description]
        leaders = []
        
        for result in results:
            leader = {column_names[i]: result[i] for i in range(len(column_names))}
            leaders.append(leader)
        
        # Close connection
        cursor.close()
        conn.close()
        
        # Store in state
        state["segment_leaders"] = leaders
        
        return {
            "status": "success",
            "segment": segment,
            "leaders": leaders
        }
    except Exception as e:
        logger.error(f"Error identifying segment leaders: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

@mcp_server.tool()
def retrieve_competitor_financial_data(company_name: str, metrics: List[str] = ["revenue", "profit", "margin"], segment: str = None) -> Dict[str, Any]:
    """Retrieve financial data for a competitor from Pinecone vector database"""
    try:
        logger.info(f"Retrieving financial data for {company_name} from Pinecone")
        
        # Initialize Pinecone
        index = init_pinecone()
        if not index:
            return {
                "status": "error",
                "message": "Failed to connect to Pinecone"
            }
        
        # Create query string
        query = f"financial data {company_name} {' '.join(metrics)}"
        logger.info(f"Search query: {query}")
        
        # Generate embeddings using SentenceTransformer instead of OpenAI
        # This matches the dimension (384) of your Pinecone index
        query_embedding = sentence_model.encode(query).tolist()
        
        # Query parameters
        query_params = {
            "vector": query_embedding,
            "top_k": 5,
            "include_metadata": True
        }
        
        # Add namespace if segment is specified
        if segment:
            # Format namespace to match expected pattern (e.g., "healthcare-diagnostic")
            namespace = f"{segment.lower().replace(' ', '-')}"
            query_params["namespace"] = namespace
            logger.info(f"Using namespace: {namespace}")
        
        # Query Pinecone
        results = index.query(**query_params)
        
        # Extract and process results from the updated Pinecone API
        financial_data = {}
        for match in results.matches:
            metadata = match.metadata
            if metadata.get("company_name") == company_name:
                # Extract relevant financial metrics
                for metric in metrics:
                    if metric in metadata:
                        financial_data[metric] = metadata[metric]
                
                # Extract form 10Q data if available
                if "form_10q" in metadata:
                    financial_data["form_10q"] = metadata["form_10q"]
                
                # Extract product performance if available
                if "product_performance" in metadata:
                    financial_data["product_performance"] = metadata["product_performance"]
        
        # Log what we found
        logger.info(f"Found {len(financial_data)} financial metrics for {company_name}")
        
        # Store in state
        if company_name not in state["competitor_data"]:
            state["competitor_data"][company_name] = {}
        
        state["competitor_data"][company_name].update(financial_data)
        
        return {
            "status": "success",
            "company": company_name,
            "financial_data": financial_data,
            "matches_count": len(results.matches)
        }
    except Exception as e:
        logger.error(f"Error retrieving competitor data: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

@mcp_server.tool()
def compare_with_segment_leader(leader_company: str, metrics: List[str] = ["revenue", "units_sold"]) -> Dict[str, Any]:
    """Compare user's sales data with a segment leader based on core metrics only"""
    try:
        logger.info(f"Comparing with segment leader: {leader_company}")
        
        # Ensure user data is available
        if state["uploaded_data"] is None or not state["analysis_results"]:
            return {
                "status": "error",
                "message": "No user data available for comparison. Please upload and analyze data first."
            }
        
        # Ensure we have competitor data
        if leader_company not in state["competitor_data"]:
            # Try to retrieve it first
            retrieve_result = retrieve_competitor_financial_data(leader_company)
            if retrieve_result["status"] == "error":
                return {
                    "status": "error",
                    "message": f"No data available for {leader_company}. Please retrieve competitor data first."
                }
        
        # Get user metrics from analysis results
        user_metrics = {}
        user_metrics["revenue"] = state["analysis_results"]["total_revenue"]
        user_metrics["units_sold"] = state["analysis_results"]["total_units"]
        
        # Get competitor metrics
        competitor_metrics = state["competitor_data"][leader_company]
        
        # Perform comparison on core metrics only
        comparison = {}
        for metric in metrics:
            if metric in user_metrics and metric in competitor_metrics:
                user_value = user_metrics[metric]
                competitor_value = competitor_metrics[metric]
                
                comparison[metric] = {
                    "user_value": user_value,
                    "competitor_value": competitor_value,
                    "difference": user_value - competitor_value,
                    "difference_percentage": ((user_value - competitor_value) / competitor_value) * 100 if competitor_value else 0
                }
        
        # Product-level comparison (only revenue and units_sold, no profit)
        product_comparison = None
        if "product_performance" in state["analysis_results"] and "product_performance" in competitor_metrics:
            product_comparison = {}
            
            # Get user product data from analysis results
            user_products = state["analysis_results"]["product_performance"]
            
            # Compare with competitor products
            competitor_products = competitor_metrics["product_performance"]
            
            # Find common products
            common_products = set(user_products["REVENUE"].keys()).intersection(set(competitor_products.keys()))
            
            for product in common_products:
                product_comparison[product] = {}
                
                # Compare revenue
                user_revenue = user_products["REVENUE"][product]
                competitor_revenue = competitor_products[product].get("revenue", 0)
                
                product_comparison[product]["revenue"] = {
                    "user_value": user_revenue,
                    "competitor_value": competitor_revenue,
                    "difference": user_revenue - competitor_revenue,
                    "difference_percentage": ((user_revenue - competitor_revenue) / competitor_revenue) * 100 if competitor_revenue else 0
                }
                
                # Compare units sold if available
                if "UNITS_SOLD" in user_products and "units_sold" in competitor_products.get(product, {}):
                    user_units = user_products["UNITS_SOLD"][product]
                    competitor_units = competitor_products[product].get("units_sold", 0)
                    
                    product_comparison[product]["units_sold"] = {
                        "user_value": user_units,
                        "competitor_value": competitor_units,
                        "difference": user_units - competitor_units,
                        "difference_percentage": ((user_units - competitor_units) / competitor_units) * 100 if competitor_units else 0
                    }
        
        # Channel comparison if available
        channel_comparison = None
        if "channel_performance" in state["analysis_results"] and "channel_performance" in competitor_metrics:
            channel_comparison = {}
            
            # Get user channel data from analysis results
            user_channels = state["analysis_results"]["channel_performance"]
            
            # Compare with competitor channels
            competitor_channels = competitor_metrics["channel_performance"]
            
            # Find common channels
            common_channels = set(user_channels["REVENUE"].keys()).intersection(set(competitor_channels.keys()))
            
            for channel in common_channels:
                channel_comparison[channel] = {}
                
                # Compare revenue
                user_revenue = user_channels["REVENUE"][channel]
                competitor_revenue = competitor_channels[channel].get("revenue", 0)
                
                channel_comparison[channel]["revenue"] = {
                    "user_value": user_revenue,
                    "competitor_value": competitor_revenue,
                    "difference": user_revenue - competitor_revenue,
                    "difference_percentage": ((user_revenue - competitor_revenue) / competitor_revenue) * 100 if competitor_revenue else 0
                }
        
        # Store comparison results
        comparison_result = {
            "company_comparison": comparison,
            "product_comparison": product_comparison,
            "channel_comparison": channel_comparison
        }
        
        state["comparative_analysis"][leader_company] = comparison_result
        
        # Prepare visualization data
        comparison_chart_data = []
        for metric, values in comparison.items():
            comparison_chart_data.append({
                "metric": metric,
                "user_value": values["user_value"],
                "competitor_value": values["competitor_value"]
            })
        
        return {
            "status": "success",
            "leader_company": leader_company,
            "comparison": comparison_result,
            "chart_data": comparison_chart_data
        }
    except Exception as e:
        logger.error(f"Error comparing with segment leader: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

@mcp_server.tool()
def generate_competitive_analysis_report(segment: str, 
                                         leader_companies: List[str],
                                         focus_metrics: List[str] = ["revenue", "profit_margin", "units_sold"],
                                         llm_response_format: str = "text") -> Dict[str, Any]:
    """Generate a focused performance gap analysis between user data and segment leaders at both company and product levels"""
    try:
        logger.info(f"Generating performance gap analysis for {segment} segment")
        
        # Ensure user data is available
        if state["uploaded_data"] is None:
            return {
                "status": "error",
                "message": "No user data available for analysis. Please upload data first."
            }
        
        # Ensure we have comparison data for all specified companies
        missing_companies = []
        for company in leader_companies:
            if company not in state["comparative_analysis"]:
                # Try to compare first
                compare_result = compare_with_segment_leader(company)
                if compare_result["status"] == "error":
                    missing_companies.append(company)
        
        if missing_companies:
            return {
                "status": "error",
                "message": f"Missing comparison data for: {', '.join(missing_companies)}. Please run comparisons first."
            }
        
        # Get user's product data
        user_data = state["uploaded_data"]
        user_products = {}
        
        if "PRODUCT_NAME" in user_data.columns:
            # Group by product
            product_metrics = ["REVENUE", "UNITS_SOLD"]
            if "ESTIMATED_PROFIT" in user_data.columns:
                product_metrics.append("ESTIMATED_PROFIT")
                
            product_df = user_data.groupby("PRODUCT_NAME")[product_metrics].sum()
            
            # Calculate product-level metrics
            for product in product_df.index:
                user_products[product] = {
                    "revenue": float(product_df.loc[product, "REVENUE"]),
                    "units_sold": int(product_df.loc[product, "UNITS_SOLD"])
                }
                
                if "ESTIMATED_PROFIT" in product_metrics:
                    user_products[product]["profit"] = float(product_df.loc[product, "ESTIMATED_PROFIT"])
                    user_products[product]["profit_margin"] = (user_products[product]["profit"] / user_products[product]["revenue"]) * 100
        
        # Prepare performance gap data for LLM
        gap_analysis_data = {
            "segment": segment,
            "user_company": {
                "summary": state["analysis_results"],
                "products": user_products
            },
            "segment_leaders": {}
        }
        
        # For each leader, extract their data and calculate gaps
        for company in leader_companies:
            company_data = state["competitor_data"].get(company, {})
            comparison_data = state["comparative_analysis"].get(company, {})
            
            # Extract company-level gaps
            company_gaps = {}
            if "company_comparison" in comparison_data:
                for metric, values in comparison_data["company_comparison"].items():
                    if metric in focus_metrics:
                        company_gaps[metric] = {
                            "user_value": values["user_value"],
                            "leader_value": values["competitor_value"],
                            "absolute_gap": values["difference"],
                            "percentage_gap": values["difference_percentage"]
                        }
            
            # Extract product-level gaps
            product_gaps = {}
            if "product_comparison" in comparison_data:
                for product, metrics in comparison_data["product_comparison"].items():
                    product_gaps[product] = {}
                    
                    for metric, values in metrics.items():
                        if metric in focus_metrics:
                            product_gaps[product][metric] = {
                                "user_value": values["user_value"],
                                "leader_value": values["competitor_value"],
                                "absolute_gap": values["difference"],
                                "percentage_gap": values["difference_percentage"]
                            }
            
            # Add to gap analysis data
            gap_analysis_data["segment_leaders"][company] = {
                "financial_data": company_data,
                "company_level_gaps": company_gaps,
                "product_level_gaps": product_gaps
            }
        
        # Convert to LLM-friendly format
        analysis_json = json.dumps(gap_analysis_data, indent=2)
        
        # Generate LLM prompt focused on gap analysis
        prompt = f"""
        You are a sales analytics expert specializing in competitive analysis. You've been given data comparing a company's sales metrics with segment leaders in the {segment} market.
        
        Please focus on analyzing the performance gaps between the user's products and the segment leader's products. Specifically:
        
        1. For each key metric ({', '.join(focus_metrics)}), identify the most significant gaps at the company level
        2. For each product that appears in both the user's data and leader's data, analyze the specific performance differences
        3. Highlight the largest performance gaps that need attention
        4. For products where the user significantly lags behind leaders, suggest possible reasons and improvement strategies
        5. If there are products where the user outperforms the segment leader, identify potential competitive advantages
        
        The analysis should clearly illustrate the gap between user products and segment leader products, with concrete numbers and percentages.
        
        Data:
        {analysis_json}
        
        Format your response as {llm_response_format}.
        """
        
        # Call your LLM service (placeholder - implement your actual LLM call)
        from your_llm_service import get_llm_response
        llm_analysis = get_llm_response(prompt)
        
        # Store the analysis
        state["llm_competitive_analysis"] = llm_analysis
        
        # Prepare visualization data for gap analysis
        gap_visualization_data = []
        
        # For the biggest gaps at product level (top 5 products with largest gaps)
        product_gap_data = []
        for company in leader_companies:
            if "product_level_gaps" in gap_analysis_data["segment_leaders"][company]:
                for product, metrics in gap_analysis_data["segment_leaders"][company]["product_level_gaps"].items():
                    if "revenue" in metrics:
                        product_gap_data.append({
                            "product": product,
                            "company": company,
                            "user_revenue": metrics["revenue"]["user_value"],
                            "leader_revenue": metrics["revenue"]["leader_value"],
                            "gap_percentage": metrics["revenue"]["percentage_gap"]
                        })
        
        # Sort by absolute gap percentage
        product_gap_data = sorted(product_gap_data, key=lambda x: abs(x["gap_percentage"]), reverse=True)[:5]
        
        return {
            "status": "success",
            "segment": segment,
            "companies_analyzed": leader_companies,
            "analysis": llm_analysis,
            "gap_visualization_data": product_gap_data
        }
    except Exception as e:
        logger.error(f"Error generating performance gap analysis: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

app.mount("/mcp", mcp_server.sse_app())

# Add health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# Add this endpoint to your FastAPI app
@app.get("/")
def read_root():
    return {"message": "Welcome to the Sales Analytics MCP Server!"}

# Add this endpoint to handle favicon requests
@app.get("/favicon.ico")
def favicon():
    return RedirectResponse(url="/static/favicon.ico")

# Run the server
if __name__ == "__main__":
    port = 8005  # Sales comparision server on port 8005
    logger.info(f"Starting Sales Analytics MCP Server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)