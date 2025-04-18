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
        
        # Create chart
        plt.figure(figsize=(10, 6))
        grouped_data = df.groupby(group_by)[metric].sum().sort_values(ascending=False)
        
        # Create bar chart
        ax = grouped_data.plot(kind="bar", color="skyblue")
        plt.title(f"{metric} by {group_by}", fontsize=14)
        plt.xlabel(group_by, fontsize=12)
        plt.ylabel(metric, fontsize=12)
        plt.grid(axis="y", alpha=0.3)
        plt.xticks(rotation=45, ha="right")
        
        # Add value labels
        for i, v in enumerate(grouped_data.values):
            ax.text(i, v + (v * 0.01), f"{v:,.0f}", ha="center", fontsize=9)
        
        plt.tight_layout()
        
        # Save to base64 for embedding
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        plt.close()
        buffer.seek(0)
        
        image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
        
        # Also save to file
        file_path = f"{metric}_by_{group_by}.png"
        plt.figure(figsize=(10, 6))
        grouped_data.plot(kind="bar", color="skyblue")
        plt.title(f"{metric} by {group_by}", fontsize=14)
        plt.xlabel(group_by, fontsize=12)
        plt.ylabel(metric, fontsize=12)
        plt.grid(axis="y", alpha=0.3)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()
        
        return {
            "status": "success",
            "title": f"{metric} by {group_by}",
            "image_base64": image_base64,
            "file_path": file_path
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
        
        # Create chart
        plt.figure(figsize=(12, 6))
        
        # Plot historical data
        plt.plot(df_agg["DATE"], df_agg["REVENUE"], marker="o", label="Historical")
        
        # Plot forecast
        plt.plot(forecast_df["DATE"], forecast_df["FORECAST_REVENUE"], marker="x", linestyle="--", label="Forecast")
        
        plt.title(f"Sales Forecast - {product_name if product_name else 'All Products'}", fontsize=14)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Revenue", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        
        # Save to base64 for embedding
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        plt.close()
        buffer.seek(0)
        
        image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
        
        # Also save to file
        file_path = f"sales_forecast_{product_name if product_name else 'all'}.png"
        plt.figure(figsize=(12, 6))
        plt.plot(df_agg["DATE"], df_agg["REVENUE"], marker="o", label="Historical")
        plt.plot(forecast_df["DATE"], forecast_df["FORECAST_REVENUE"], marker="x", linestyle="--", label="Forecast")
        plt.title(f"Sales Forecast - {product_name if product_name else 'All Products'}", fontsize=14)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Revenue", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()
        
        # Prepare forecast data
        forecast_data = []
        for i, row in forecast_df.iterrows():
            forecast_data.append({
                "date": row["DATE"].strftime("%Y-%m-%d"),
                "revenue": float(row["FORECAST_REVENUE"])
            })
        
        return {
            "status": "success",
            "product": product_name if product_name else "All Products",
            "forecast_periods": periods,
            "forecast_data": forecast_data,
            "image_base64": image_base64,
            "file_path": file_path
        }
    except Exception as e:
        logger.error(f"Error calculating sales forecast: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

# Mount MCP server to FastAPI app
# The mount_to_app method is deprecated in newer versions of FastMCP
# Instead, we'll use the FastAPI mount method
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
