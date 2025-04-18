"""
Segment-specific MCP server for MarketScope platform
This server provides tools for analyzing segment-specific sales data
and storing it in the appropriate Snowflake schema.
"""
import json
import pandas as pd
import os
import io
import base64
from typing import Dict, Any, List, Optional, Union
import asyncio
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from mcp.server.fastmcp import FastMCP
import uvicorn
from dotenv import load_dotenv
load_dotenv(override=True)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("segment_mcp_server")

# Import project utilities
from config.config import Config
from agents.custom_mcp_client import CustomMCPClient

class SegmentMCPServer:
    """
    MCP server for segment-specific sales data analysis tools.
    Creates an MCP server that uses a specified Snowflake schema
    for data operations based on the selected segment.
    """
    
    def __init__(self, segment_name: str, schema_name: str, server_name: str, port: int = 8010):
        """
        Initialize the segment-specific MCP server
        
        Args:
            segment_name: Name of the segment (e.g., "Diagnostic Segment")
            schema_name: Snowflake schema to use for this segment
            server_name: Name to use for the MCP server
            port: Port to run the server on
        """
        self.segment_name = segment_name
        self.schema_name = schema_name
        self.server_name = server_name
        self.port = port
        self.app = FastAPI(title=f"{segment_name} MCP Server")
        self.mcp_server = FastMCP(server_name)
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # State for storing data
        self.state = {
            "uploaded_data": None,
            "product_names": [],
            "product_trends": {},
            "current_analysis": None,
            "snowflake_table": None,
            "schema_name": schema_name
        }
        
        # Register MCP tools
        self._register_tools()
    
    def _register_tools(self):
        """Register MCP tools for this server"""
        # Register upload to Snowflake tool
        self._register_upload_tool()
        
        # Register product list tool
        self._register_product_list_tool()
        
        # Register visualization tool
        self._register_visualization_tool()
        
        # Register summary tool
        self._register_summary_tool()
        
        # Register product trends tool
        self._register_product_trends_tool()
        
        # Register trends visualization tools
        self._register_trends_tools()
        
    def _register_upload_tool(self):
        @self.mcp_server.tool()
        def upload_to_snowflake(csv_data: str, table_name: Optional[str] = None) -> Dict[str, Any]:
            """Upload CSV data to Snowflake database using segment-specific schema"""
            try:
                # Parse CSV into dataframe
                df = pd.read_csv(io.StringIO(csv_data))
                self.state["uploaded_data"] = df
                
                # Set a default table name if not provided, include segment in name
                if not table_name:
                    import time
                    timestamp = int(time.time())
                    table_name = f"{self.schema_name}.SALES_DATA_{timestamp}"
                # If table name doesn't include schema, add it
                elif "." not in table_name:
                    table_name = f"{self.schema_name}.{table_name}"
                
                self.state["snowflake_table"] = table_name
                
                # Initialize MCP client to connect to Snowflake server
                client = CustomMCPClient(base_url=os.getenv("MCP_SERVER_URL", "http://localhost:8000"))
                
                # Convert dataframe to CSV string
                csv_string = df.to_csv(index=False)
                
                # Upload to Snowflake using the MCP client's load_csv_to_table tool
                result = client.invoke("load_csv_to_table", {
                    "table_name": table_name,
                    "csv_data": csv_string,
                    "create_table": True
                })
                
                return {
                    "status": "success" if "successfully" in str(result) else "error",
                    "message": str(result),
                    "table_name": table_name,
                    "schema_name": self.schema_name,
                    "segment": self.segment_name,
                    "row_count": len(df)
                }
            except Exception as e:
                logger.error(f"Error in upload_to_snowflake: {str(e)}")
                return {
                    "status": "error",
                    "message": str(e)
                }

    def _register_product_list_tool(self):
        @self.mcp_server.tool()
        def get_product_list(table_name: Optional[str] = None) -> List[str]:
            """Get list of unique products from the sales data in segment-specific schema"""
            try:
                if table_name is None and self.state["snowflake_table"] is not None:
                    table_name = self.state["snowflake_table"]
                
                if self.state["uploaded_data"] is not None:
                    # Use in-memory data if available
                    df = self.state["uploaded_data"]
                    if "PRODUCT_NAME" in df.columns:
                        products = df["PRODUCT_NAME"].unique().tolist()
                        self.state["product_names"] = products
                        return products
                
                # If not in memory or table name provided, fetch from Snowflake
                if table_name:
                    client = CustomMCPClient(base_url=os.getenv("MCP_SERVER_URL", "http://localhost:8000"))
                    
                    # Ensure table name has schema
                    if "." not in table_name:
                        table_name = f"{self.schema_name}.{table_name}"
                        
                    query = f"SELECT DISTINCT PRODUCT_NAME FROM {table_name}"
                    result = client.invoke("execute_query", {"query": query})
                    
                    try:
                        # Parse the result from Snowflake query
                        if isinstance(result, str) and "rows. (Execution time:" in result:
                            json_start = result.find('[')
                            json_end = result.rfind(']') + 1
                            if json_start >= 0 and json_end > json_start:
                                data = json.loads(result[json_start:json_end])
                                products = [item.get("PRODUCT_NAME") for item in data if item.get("PRODUCT_NAME")]
                                self.state["product_names"] = products
                                return products
                    except Exception as e:
                        return [f"Error parsing query result: {str(e)}"]
                
                return [f"No product data found for segment {self.segment_name}. Please upload sales data first."]
            except Exception as e:
                logger.error(f"Error in get_product_list: {str(e)}")
                return [f"Error getting product list: {str(e)}"]

    def _register_visualization_tool(self):
        @self.mcp_server.tool()
        def create_sales_visualization(table_name: Optional[str] = None, 
                                      metric: str = "REVENUE", 
                                      group_by: str = "PRODUCT_NAME") -> Dict[str, Any]:
            """Create visualization from sales data for segment-specific schema"""
            try:
                df = None
                
                # Use in-memory data if available
                if self.state["uploaded_data"] is not None:
                    df = self.state["uploaded_data"]
                # Otherwise fetch from Snowflake
                elif table_name or self.state["snowflake_table"]:
                    table = table_name or self.state["snowflake_table"]
                    
                    # Ensure table name has schema
                    if "." not in table:
                        table = f"{self.schema_name}.{table}"
                        
                    client = CustomMCPClient(base_url=os.getenv("MCP_SERVER_URL", "http://localhost:8000"))
                    result = client.invoke("execute_query", {"query": f"SELECT * FROM {table} LIMIT 1000"})
                    
                    # Parse result
                    if isinstance(result, str) and "rows. (Execution time:" in result:
                        json_start = result.find('[')
                        json_end = result.rfind(']') + 1
                        if json_start >= 0 and json_end > json_start:
                            data = json.loads(result[json_start:json_end])
                            df = pd.DataFrame(data)
                
                if df is None or df.empty:
                    return {
                        "status": "error",
                        "message": f"No data available for visualization in {self.segment_name} segment"
                    }
                
                # Validate columns exist
                if metric not in df.columns:
                    return {
                        "status": "error", 
                        "message": f"Column '{metric}' not found in data. Available columns: {', '.join(df.columns)}"
                    }
                    
                if group_by not in df.columns:
                    return {
                        "status": "error",
                        "message": f"Column '{group_by}' not found in data. Available columns: {', '.join(df.columns)}"
                    }
                
                # Group the data
                grouped = df.groupby(group_by)[metric].sum().sort_values(ascending=False)
                
                # Convert to data format for visualization in Streamlit
                chart_data = {
                    "labels": grouped.index.tolist(),
                    "values": grouped.values.tolist(), 
                    "type": "bar",
                    "x_label": group_by.replace('_', ' ').title(),
                    "y_label": metric.replace('_', ' ').title(),
                    "title": f'{metric} by {group_by} ({self.segment_name})'
                }
                
                return {
                    "status": "success",
                    "title": f"{metric} by {group_by} ({self.segment_name})",
                    "chart_data": chart_data,
                    "data": grouped.reset_index().to_dict(orient="records"),
                    "segment": self.segment_name,
                    "schema": self.schema_name
                }
            except Exception as e:
                logger.error(f"Error in create_sales_visualization: {str(e)}")
                return {
                    "status": "error",
                    "message": f"Error creating visualization: {str(e)}"
                }

    def _register_summary_tool(self):
        @self.mcp_server.tool()
        def generate_sales_summary(table_name: Optional[str] = None) -> Dict[str, Any]:
            """Generate a comprehensive summary of the sales data for segment-specific schema"""
            try:
                df = None
                
                # Use in-memory data if available
                if self.state["uploaded_data"] is not None:
                    df = self.state["uploaded_data"]
                # Otherwise fetch from Snowflake
                elif table_name or self.state["snowflake_table"]:
                    table = table_name or self.state["snowflake_table"]
                    
                    # Ensure table name has schema
                    if "." not in table:
                        table = f"{self.schema_name}.{table}"
                        
                    client = CustomMCPClient(base_url=os.getenv("MCP_SERVER_URL", "http://localhost:8000"))
                    result = client.invoke("execute_query", {"query": f"SELECT * FROM {table} LIMIT 1000"})
                    
                    # Parse result
                    if isinstance(result, str) and "rows. (Execution time:" in result:
                        json_start = result.find('[')
                        json_end = result.rfind(']') + 1
                        if json_start >= 0 and json_end > json_start:
                            data = json.loads(result[json_start:json_end])
                            df = pd.DataFrame(data)
                
                if df is None or df.empty:
                    return {
                        "status": "error",
                        "message": f"No data available for analysis in {self.segment_name} segment"
                    }
                
                # Basic statistics
                total_revenue = df["REVENUE"].sum() if "REVENUE" in df.columns else 0
                total_units = df["UNITS_SOLD"].sum() if "UNITS_SOLD" in df.columns else 0
                total_profit = df["ESTIMATED_PROFIT"].sum() if "ESTIMATED_PROFIT" in df.columns else 0
                
                # Product performance
                product_performance = None
                if "PRODUCT_NAME" in df.columns and "REVENUE" in df.columns:
                    product_performance = df.groupby("PRODUCT_NAME").agg({
                        "UNITS_SOLD": "sum" if "UNITS_SOLD" in df.columns else "count",
                        "REVENUE": "sum",
                        "ESTIMATED_PROFIT": "sum" if "ESTIMATED_PROFIT" in df.columns else "count"
                    }).sort_values("REVENUE", ascending=False).head(5).to_dict('index')
                
                # Channel performance
                channel_performance = None
                if "SALES_CHANNEL" in df.columns and "REVENUE" in df.columns:
                    channel_performance = df.groupby("SALES_CHANNEL").agg({
                        "REVENUE": "sum"
                    }).sort_values("REVENUE", ascending=False).to_dict('index')
                
                # Marketing strategy effectiveness
                marketing_performance = None
                if "MARKETING_STRATEGY" in df.columns and "REVENUE" in df.columns:
                    marketing_performance = df.groupby("MARKETING_STRATEGY").agg({
                        "REVENUE": "sum",
                        "UNITS_SOLD": "sum" if "UNITS_SOLD" in df.columns else "count"
                    }).sort_values("REVENUE", ascending=False).to_dict('index')
                
                # Geographic performance
                geo_performance = None
                if "STATE" in df.columns and "REVENUE" in df.columns:
                    geo_performance = df.groupby("STATE").agg({
                        "REVENUE": "sum"
                    }).sort_values("REVENUE", ascending=False).head(5).to_dict('index')
                
                summary = {
                    "status": "success",
                    "segment": self.segment_name,
                    "schema": self.schema_name,
                    "total_revenue": float(total_revenue),
                    "total_units_sold": int(total_units),
                    "total_profit": float(total_profit),
                    "product_performance": product_performance,
                    "channel_performance": channel_performance,
                    "marketing_performance": marketing_performance,
                    "geographic_performance": geo_performance
                }
                
                self.state["current_analysis"] = summary
                return summary
            except Exception as e:
                logger.error(f"Error in generate_sales_summary: {str(e)}")
                return {
                    "status": "error",
                    "message": f"Error generating sales summary: {str(e)}"
                }

    def _register_product_trends_tool(self):
        @self.mcp_server.tool()
        def analyze_product_trends(product_names: Optional[List[str]] = None) -> Dict[str, Any]:
            """Analyze trends for given products using Google Shopping data via SerpAPI"""
            try:
                # Use provided product names or the ones in state
                if product_names is None:
                    product_names = self.state.get("product_names", [])
                
                if not product_names:
                    return {
                        "status": "error",
                        "message": f"No products to analyze for {self.segment_name}. Please provide product names or get them from sales data first."
                    }
                
                # Check if SerpAPI is available
                try:
                    from serpapi import GoogleSearch
                except ImportError:
                    return {
                        "status": "error",
                        "message": "SerpAPI package not available. Please install with: pip install google-search-results"
                    }
                
                # Check for API key
                api_key = os.getenv("SERP_API_KEY", Config.SERP_API_KEY if hasattr(Config, "SERP_API_KEY") else None)
                if not api_key:
                    return {
                        "status": "error",
                        "message": "SERP_API_KEY not found in environment variables or Config."
                    }
                
                # Extract best keywords from product names
                from nltk.tokenize import word_tokenize
                from sklearn.feature_extraction.text import TfidfVectorizer
                import nltk
                
                # Download NLTK data if needed
                try:
                    word_tokenize("test")
                except LookupError:
                    nltk.download('punkt')
                
                def extract_top_keyword(phrase):
                    tokens = word_tokenize(phrase.lower())
                    vectorizer = TfidfVectorizer()
                    X = vectorizer.fit_transform([" ".join(tokens)])
                    feature_names = vectorizer.get_feature_names_out()
                    tfidf_scores = X.toarray()[0]
                    top_index = tfidf_scores.argmax()
                    return feature_names[top_index]
                
                # Analyze up to 5 products (to avoid API rate limits)
                product_subset = product_names[:5]
                all_product_data = []
                
                for product_name in product_subset:
                    best_keyword = extract_top_keyword(product_name)
                    health_keyword = self.segment_name.split()[0].lower()  # First word of segment (e.g., "Diagnostic")
                    
                    params = {
                        "engine": "google_shopping",
                        "q": f"{health_keyword} {best_keyword}",
                        "hl": "en",
                        "gl": "us",
                        "api_key": api_key
                    }
                    
                    search = GoogleSearch(params)
                    results = search.get_dict()
                    products = results.get("shopping_results", [])
                    
                    for product in products:
                        all_product_data.append({
                            "product_name": product_name,
                            "search_keyword": f"{health_keyword} {best_keyword}",
                            "title": product.get("title"),
                            "price": product.get("price"),
                            "rating": product.get("rating"),
                            "reviews": product.get("reviews"),
                            "link": product.get("link")
                        })
                
                # Convert to DataFrame for analysis
                trend_df = pd.DataFrame(all_product_data)
                self.state["product_trends_df"] = trend_df
                
                return {
                    "status": "success",
                    "segment": self.segment_name,
                    "products_analyzed": product_subset,
                    "total_results": len(all_product_data),
                    "results_per_product": {p: len([item for item in all_product_data if item['product_name'] == p]) for p in product_subset},
                    "avg_rating": trend_df['rating'].mean() if 'rating' in trend_df.columns and not trend_df.empty else None,
                    "data": all_product_data[:10]  # Return first 10 items as preview
                }
                
            except Exception as e:
                logger.error(f"Error in analyze_product_trends: {str(e)}")
                return {
                    "status": "error",
                    "message": f"Error analyzing product trends: {str(e)}"
                }
    
    def _register_trends_tools(self):
        @self.mcp_server.tool()
        def fetch_product_trends(table_name: Optional[str] = None) -> Dict[str, Any]:
            """Fetch product trends from Google Shopping API based on the product names in the specified Snowflake table."""
            try:
                # Get product list from the table first
                products = []
                
                if self.state["uploaded_data"] is not None:
                    # Use in-memory data if available
                    df = self.state["uploaded_data"]
                    if "PRODUCT_NAME" in df.columns:
                        products = df["PRODUCT_NAME"].unique().tolist()
                elif table_name or self.state["snowflake_table"]:
                    # Otherwise fetch from Snowflake
                    table = table_name or self.state["snowflake_table"]
                    
                    # Ensure table name has schema
                    if "." not in table:
                        table = f"{self.schema_name}.{table}"
                        
                    client = CustomMCPClient(base_url=os.getenv("MCP_SERVER_URL", "http://localhost:8000"))
                    query = f"SELECT DISTINCT PRODUCT_NAME FROM {table}"
                    result = client.invoke("execute_query", {"query": query})
                    
                    try:
                        # Parse the result from Snowflake query
                        if isinstance(result, str) and "rows. (Execution time:" in result:
                            json_start = result.find('[')
                            json_end = result.rfind(']') + 1
                            if json_start >= 0 and json_end > json_start:
                                data = json.loads(result[json_start:json_end])
                                products = [item.get("PRODUCT_NAME") for item in data if item.get("PRODUCT_NAME")]
                    except Exception as e:
                        return {
                            "status": "error",
                            "message": f"Error parsing product query result: {str(e)}"
                        }
                
                if not products:
                    return {
                        "status": "error",
                        "message": f"No products found for segment {self.segment_name}. Please upload sales data first."
                    }
                
                # Check if SerpAPI is available
                try:
                    from serpapi import GoogleSearch
                except ImportError:
                    return {
                        "status": "error",
                        "message": "SerpAPI package not available. Please install with: pip install google-search-results"
                    }
                
                # Check for API key
                api_key = os.getenv("SERP_API_KEY", Config.SERP_API_KEY if hasattr(Config, "SERP_API_KEY") else None)
                if not api_key:
                    return {
                        "status": "error",
                        "message": "SERP_API_KEY not found in environment variables or Config."
                    }
                
                # Extract best keywords from product names
                from nltk.tokenize import word_tokenize
                from sklearn.feature_extraction.text import TfidfVectorizer
                import nltk
                
                # Download NLTK data if needed
                try:
                    word_tokenize("test")
                except LookupError:
                    nltk.download('punkt')
                
                def extract_top_keyword(phrase):
                    tokens = word_tokenize(phrase.lower())
                    vectorizer = TfidfVectorizer()
                    X = vectorizer.fit_transform([" ".join(tokens)])
                    feature_names = vectorizer.get_feature_names_out()
                    tfidf_scores = X.toarray()[0]
                    top_index = tfidf_scores.argmax()
                    return feature_names[top_index]
                
                # --- Start Product Data Scrape ---
                all_product_data = []
                
                # Limit to 5 products to avoid API rate limits
                product_subset = products[:5]
                
                for product_name in product_subset:
                    best_keyword = extract_top_keyword(product_name)
                    # Add segment keyword for context
                    segment_keyword = self.segment_name.split()[0].lower()  # e.g., "diagnostic"
                    
                    params = {
                        "engine": "google_shopping",
                        "q": f"{segment_keyword} {best_keyword}",
                        "hl": "en",
                        "gl": "us",
                        "api_key": api_key
                    }
                    
                    search = GoogleSearch(params)
                    results = search.get_dict()
                    products_found = results.get("shopping_results", [])
                    
                    for product in products_found:
                        all_product_data.append({
                            "product_name": product_name,  # original input
                            "search_keyword": f"{segment_keyword} {best_keyword}",  # used for search
                            "title": product.get("title"),
                            "price": product.get("price"),
                            "rating": product.get("rating"),
                            "reviews": product.get("reviews"),
                            "link": product.get("link")
                        })
                
                # Convert to dataframe and store in state
                trends_df = pd.DataFrame(all_product_data)
                self.state["product_trends_df"] = trends_df
                
                return {
                    "status": "success",
                    "message": f"Retrieved {len(all_product_data)} product trends for {len(product_subset)} products",
                    "segment": self.segment_name,
                    "products_analyzed": product_subset,
                    "results_count": len(all_product_data),
                    "data_preview": all_product_data[:5] if all_product_data else []
                }
                
            except Exception as e:
                logger.error(f"Error in fetch_product_trends: {str(e)}")
                return {
                    "status": "error",
                    "message": f"Error fetching product trends: {str(e)}"
                }
        
        @self.mcp_server.tool()
        def create_trends_visualization(visualization_type: str = "price_comparison") -> Dict[str, Any]:
            """Create visualizations from the product trends data fetched from Google Shopping API.
            
            Args:
                visualization_type: Type of visualization to create. Options:
                    - price_comparison: Compare prices across similar products
                    - rating_analysis: Analyze ratings and reviews
                    - price_distribution: Show price distribution
            """
            try:
                # Check if trends data is available
                if "product_trends_df" not in self.state or self.state["product_trends_df"] is None:
                    return {
                        "status": "error",
                        "message": "No product trends data available. Please run fetch_product_trends first."
                    }
                
                # Get the dataframe
                import numpy as np
                
                df = self.state["product_trends_df"]
                
                if df.empty:
                    return {
                        "status": "error",
                        "message": "Product trends dataframe is empty. No data to visualize."
                    }
                
                # Process price column to ensure numeric values
                if "price" in df.columns:
                    # Extract numeric values from price strings
                    df["price_value"] = df["price"].apply(lambda x: 
                        float(''.join(filter(
                            lambda c: c.isdigit() or c == '.', 
                            str(x).replace(',', '')
                        ))) if x else np.nan
                    )
                
                # Prepare chart data based on visualization type
                if visualization_type == "price_comparison":
                    # Group by product_name and calculate average price
                    avg_prices = df.groupby("product_name")["price_value"].mean().sort_values(ascending=False)
                    
                    # Prepare data for Streamlit visualization
                    chart_data = {
                        "type": "bar",
                        "title": f"Average Price Comparison - {self.segment_name}",
                        "x_label": "Product",
                        "y_label": "Average Price ($)",
                        "labels": avg_prices.index.tolist(),
                        "values": avg_prices.values.tolist(),
                        "color": "skyblue"
                    }
                    
                    return {
                        "status": "success",
                        "title": f"Price Comparison - {self.segment_name}",
                        "visualization_type": visualization_type,
                        "segment": self.segment_name,
                        "chart_data": chart_data,
                        "data": avg_prices.reset_index().to_dict(orient="records"),
                        "sample_size": len(df),
                        "products_analyzed": df["product_name"].nunique()
                    }
                    
                elif visualization_type == "rating_analysis":
                    # Filter out rows without ratings
                    df_with_ratings = df.dropna(subset=["rating"])
                    
                    if df_with_ratings.empty:
                        return {
                            "status": "error",
                            "message": "No rating data available for visualization"
                        }
                    
                    # Convert ratings to numeric
                    df_with_ratings["rating_value"] = pd.to_numeric(df_with_ratings["rating"], errors="coerce")
                    
                    # Group by product and calculate average rating
                    avg_ratings = df_with_ratings.groupby("product_name")["rating_value"].mean().sort_values(ascending=False)
                    
                    # Prepare data for Streamlit visualization
                    chart_data = {
                        "type": "bar",
                        "title": f"Average Rating Comparison - {self.segment_name}",
                        "x_label": "Product",
                        "y_label": "Average Rating (out of 5)",
                        "labels": avg_ratings.index.tolist(),
                        "values": avg_ratings.values.tolist(),
                        "color": "lightgreen",
                        "y_min": 0,
                        "y_max": 5.5
                    }
                    
                    return {
                        "status": "success",
                        "title": f"Rating Analysis - {self.segment_name}",
                        "visualization_type": visualization_type,
                        "segment": self.segment_name,
                        "chart_data": chart_data,
                        "data": avg_ratings.reset_index().to_dict(orient="records"),
                        "sample_size": len(df_with_ratings),
                        "products_analyzed": df_with_ratings["product_name"].nunique()
                    }
                    
                elif visualization_type == "price_distribution":
                    # Filter out missing prices
                    price_data = df.dropna(subset=["price_value"])["price_value"]
                    
                    if len(price_data) == 0:
                        return {
                            "status": "error",
                            "message": "No price data available for visualization"
                        }
                    
                    # Calculate statistics for the histogram
                    median_price = price_data.median()
                    mean_price = price_data.mean()
                    
                    # Group data into bins for histogram
                    min_price = price_data.min()
                    max_price = price_data.max()
                    bins = 20
                    bin_width = (max_price - min_price) / bins if max_price > min_price else 1
                    
                    hist, bin_edges = np.histogram(price_data, bins=bins)
                    
                    # Prepare data for Streamlit visualization
                    chart_data = {
                        "type": "histogram",
                        "title": f"Price Distribution - {self.segment_name}",
                        "x_label": "Price ($)",
                        "y_label": "Frequency",
                        "hist_values": hist.tolist(),
                        "bin_edges": bin_edges.tolist(),
                        "median": float(median_price),
                        "mean": float(mean_price),
                        "color": "salmon"
                    }
                    
                    return {
                        "status": "success",
                        "title": f"Price Distribution - {self.segment_name}",
                        "visualization_type": visualization_type,
                        "segment": self.segment_name,
                        "chart_data": chart_data,
                        "data": price_data.to_frame("price").reset_index(drop=True).to_dict(orient="records"),
                        "statistics": {
                            "median": float(median_price),
                            "mean": float(mean_price),
                            "min": float(min_price),
                            "max": float(max_price),
                        },
                        "sample_size": len(price_data)
                    }
                
                else:
                    return {
                        "status": "error",
                        "message": f"Unknown visualization type: {visualization_type}. Supported types: price_comparison, rating_analysis, price_distribution"
                    }
                
            except Exception as e:
                logger.error(f"Error in create_trends_visualization: {str(e)}")
                return {
                    "status": "error",
                    "message": f"Error creating visualization: {str(e)}"
                }
        
    def mount_and_run(self):
        """Mount the MCP server to the FastAPI app and run it"""
        # Mount MCP server - use the newer API
        # The mount_to_app method is deprecated in newer versions of FastMCP
        # Instead, we'll use the FastAPI mount method
        self.app.mount("/mcp", self.mcp_server.sse_app())
        
        # Simple routes for checking server health
        @self.app.get("/")
        def root():
            return {
                "message": f"{self.segment_name} MCP Server is running",
                "status": "healthy"
            }
            
        @self.app.get("/health")
        def health():
            return {
                "status": "healthy",
                "segment": self.segment_name,
                "schema": self.schema_name
            }
            
        # Run the server
        uvicorn.run(self.app, host="0.0.0.0", port=self.port)


def create_segment_server(segment_name: str = "Skin Care Segment"):
    """Create a segment-specific MCP server based on configuration in Config class
    
    Args:
        segment_name: Name of the segment to create a server for
        
    Returns:
        SegmentMCPServer instance
    """
    if segment_name not in Config.SEGMENT_CONFIG:
        raise ValueError(f"Unknown segment: {segment_name}")
        
    segment_config = Config.SEGMENT_CONFIG[segment_name]
    schema_name = segment_config["schema"]
    port = segment_config["port"]
    server_name = Config.MCP_SERVER_NAMES.get(segment_name, segment_name.replace(" ", "_").lower() + "_mcp_server")
    
    # Create the server
    return SegmentMCPServer(
        segment_name=segment_name,
        schema_name=schema_name,
        server_name=server_name,
        port=port
    )

def run_segment_server(segment_name: str):
    """Run a single segment MCP server based on configuration in Config class"""
    srv = create_segment_server(segment_name)
    print(f"Starting {segment_name} MCP Server on port {srv.port}")
    srv.mount_and_run()

# Create a default server instance for importing by run_all_servers.py
# Using skin care as default segment, but can be overridden when actually running
server = create_segment_server("Skin Care Segment")

# Main entry point
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python segment_mcp_server.py <segment_name>")
        print(f"Available segments: {', '.join(Config.SEGMENT_CONFIG.keys())}")
        sys.exit(1)
        
    segment_name = sys.argv[1]
    run_segment_server(segment_name)
