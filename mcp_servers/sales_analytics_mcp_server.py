"""
Sales Analytics MCP Server
Provides tools for analyzing sales data and trends
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
from mcp.server.fastmcp import FastMCP
import uvicorn
from typing import Dict, Any, List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sales_analytics_mcp_server")

# Import configuration
from config.config import Config

# Create FastAPI app
app = FastAPI(title="Sales Analytics MCP Server")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create MCP server
mcp_server = FastMCP("sales_analytics")

# In-memory state for the server
state = {
    "uploaded_data": None,
    "analysis_results": {}
}

# Register MCP tools
@mcp_server.tool()
def analyze_sales_data(data: Optional[Any] = None) -> Dict[str, Any]:
    """Analyze sales data to extract key insights"""
    try:
        logger.info("Analyzing sales data")
        
        # Use provided data or previously uploaded data
        df = None
        if isinstance(data, pd.DataFrame):
            df = data
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


app.mount("/mcp", mcp_server.sse_app())

# Add health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# Run the server
if __name__ == "__main__":
    port = 8002  # Sales analytics server on port 8002
    logger.info(f"Starting Sales Analytics MCP Server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
