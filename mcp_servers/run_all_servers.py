"""
Unified MCP Server for MarketScope platform
This script starts a single unified MCP server that handles all functionalities
including Snowflake, Market Analysis, and segment-specific operations.

Note: The following files are redundant and can be safely removed:
- run_servers.py (in root directory - just a wrapper for this file)
- start_all.py (starts multiple separate MCP servers instead of this unified one)
- Various .bat files (start_api.bat, start_complete.bat, start_direct.bat, etc.) 
- test_mcp_server.py (test implementation that's been superseded by this production server)

The recommended way to run the platform is using run.py which properly starts this server,
the API, and the frontend in the correct order with proper monitoring.
"""
import os
import sys
import logging
import time

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import config
from config.config import Config
from config.paths import setup_paths
setup_paths()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("unified_mcp_server")

# Import necessary libraries
import pandas as pd
import json
import io
from mcp.server.fastmcp import FastMCP
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Create a FastAPI app
app = FastAPI(title="MarketScope MCP Server")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

# Add segments endpoint
@app.get("/segments")
async def get_segments():
    """Get available segments"""
    segments = [
        {
            "id": "skin_care",
            "name": "Skin Care Segment",
            "description": "Skincare products and treatments"
        },
        {
            "id": "pharma",
            "name": "Pharmaceutical Segment",
            "description": "Prescription and over-the-counter medications"
        },
        {
            "id": "diagnostic",
            "name": "Diagnostic Segment",
            "description": "Medical diagnostic tools and equipment"
        },
        {
            "id": "supplement",
            "name": "Supplement Segment",
            "description": "Dietary supplements and nutritional products"
        },
        {
            "id": "medical_device",
            "name": "Medical Device Segment",
            "description": "Medical devices and equipment for healthcare providers"
        }
    ]
    
    return {"segments": segments}

# Create a single unified MCP server
mcp_server = FastMCP("marketscope")

# Register Snowflake tools
@mcp_server.tool()
def execute_query(query: str) -> str:
    """Execute a SQL query on Snowflake database"""
    try:
        # Mock implementation - in a real scenario, this would connect to Snowflake
        logger.info(f"Executing query: {query}")
        
        # Parse the query to determine what to return
        if query.lower().startswith("select"):
            # For SELECT queries, return mock data based on the query
            if "product_name" in query.lower():
                # Mock product data
                data = [
                    {"PRODUCT_NAME": "HeartGuard Monitor"},
                    {"PRODUCT_NAME": "DiabeCare Sensor"},
                    {"PRODUCT_NAME": "PainEase Gel"},
                    {"PRODUCT_NAME": "Vitamin Complex"},
                    {"PRODUCT_NAME": "PediCare Drops"}
                ]
                return f"5 rows. (Execution time: 0.5s)\n{json.dumps(data)}"
            else:
                # Generic mock data
                data = [
                    {"COLUMN1": "Value1", "COLUMN2": 123},
                    {"COLUMN1": "Value2", "COLUMN2": 456}
                ]
                return f"2 rows. (Execution time: 0.3s)\n{json.dumps(data)}"
        else:
            # For non-SELECT queries, return success message
            return f"Query executed successfully. (Execution time: 0.2s)"
    except Exception as e:
        logger.error(f"Error executing query: {str(e)}")
        return f"Error: {str(e)}"

@mcp_server.tool()
def load_csv_to_table(segment_name: str, table_name: str, csv_data: str, create_table: bool = True) -> str:
    """
    Load CSV data into a Snowflake table based on segment
    
    Args:
        segment_name: Name of the segment (e.g., "Skin Care Segment")
        table_name: Name of the table to load data into
        csv_data: CSV data as a string
        create_table: Whether to create the table if it doesn't exist
        
    Returns:
        Result message
    """
    try:
        logger.info(f"Loading data for segment '{segment_name}' into table: {table_name}")
        
        # Parse CSV data
        df = pd.read_csv(io.StringIO(csv_data))
        row_count = len(df)
        
        # Use segment name to determine which table to load into
        segment_table = f"{segment_name.replace(' ', '_').lower()}_{table_name}"
        
        # Mock implementation - in a real scenario, this would load data to Snowflake
        # with segment-specific handling
        return f"Successfully loaded {row_count} rows into table {segment_table} for {segment_name}."
    except Exception as e:
        logger.error(f"Error loading CSV data: {str(e)}")
        return f"Error: {str(e)}"

@mcp_server.tool()
def get_table_schema(table_name: str) -> str:
    """Get schema information for a Snowflake table"""
    try:
        logger.info(f"Getting schema for table: {table_name}")
        
        # Mock implementation - return fake schema based on table name
        if "sales" in table_name.lower():
            schema = [
                {"COLUMN_NAME": "DATE", "DATA_TYPE": "DATE"},
                {"COLUMN_NAME": "PRODUCT_NAME", "DATA_TYPE": "VARCHAR"},
                {"COLUMN_NAME": "PRICE", "DATA_TYPE": "NUMBER"},
                {"COLUMN_NAME": "UNITS_SOLD", "DATA_TYPE": "NUMBER"},
                {"COLUMN_NAME": "REVENUE", "DATA_TYPE": "NUMBER"}
            ]
        else:
            schema = [
                {"COLUMN_NAME": "ID", "DATA_TYPE": "NUMBER"},
                {"COLUMN_NAME": "NAME", "DATA_TYPE": "VARCHAR"},
                {"COLUMN_NAME": "VALUE", "DATA_TYPE": "NUMBER"}
            ]
        
        return f"Table {table_name} Schema:\n{json.dumps(schema)}"
    except Exception as e:
        logger.error(f"Error getting table schema: {str(e)}")
        return f"Error: {str(e)}"

# Register Segment-Specific Tools for each Healthcare Segment

# Skin Care Segment Tools
@mcp_server.tool()
def analyze_skin_care_market(product_category: str = None) -> str:
    """Analyze the skin care market for specific product categories"""
    try:
        logger.info(f"Analyzing skin care market for product category: {product_category}")
        
        # Mock implementation for skin care market analysis
        categories = {
            "cleansers": {
                "market_size": "$12.5 billion",
                "growth_rate": "7.2%",
                "trends": ["Natural ingredients", "Microbiome-friendly", "Sustainable packaging"]
            },
            "moisturizers": {
                "market_size": "$18.3 billion",
                "growth_rate": "8.5%",
                "trends": ["Anti-aging formulations", "Hyaluronic acid", "CBD-infused"]
            },
            "sunscreen": {
                "market_size": "$9.8 billion",
                "growth_rate": "9.1%",
                "trends": ["Mineral-based", "Reef-safe", "Tinted options"]
            }
        }
        
        if product_category and product_category.lower() in categories:
            return json.dumps(categories[product_category.lower()])
        else:
            # Return overview of all categories
            return json.dumps({
                "total_market_size": "$45.7 billion",
                "overall_growth": "8.2%",
                "categories": list(categories.keys())
            })
    except Exception as e:
        logger.error(f"Error analyzing skin care market: {str(e)}")
        return f"Error: {str(e)}"

# Pharmaceutical Segment Tools
@mcp_server.tool()
def analyze_pharma_market(drug_class: str = None) -> str:
    """Analyze the pharmaceutical market for specific drug classes"""
    try:
        logger.info(f"Analyzing pharmaceutical market for drug class: {drug_class}")
        
        # Mock implementation for pharmaceutical market analysis
        drug_classes = {
            "antibiotics": {
                "market_size": "$42.3 billion",
                "growth_rate": "4.5%",
                "trends": ["Antibiotic resistance concerns", "Novel mechanisms", "Combination therapies"]
            },
            "anti-inflammatory": {
                "market_size": "$89.5 billion",
                "growth_rate": "6.8%",
                "trends": ["Biologics", "Targeted therapies", "JAK inhibitors"]
            },
            "cardiovascular": {
                "market_size": "$107.2 billion",
                "growth_rate": "5.2%",
                "trends": ["Preventive care", "Polypills", "Device integration"]
            }
        }
        
        if drug_class and drug_class.lower() in drug_classes:
            return json.dumps(drug_classes[drug_class.lower()])
        else:
            # Return overview of all categories
            return json.dumps({
                "total_market_size": "$1.27 trillion",
                "overall_growth": "5.5%",
                "drug_classes": list(drug_classes.keys())
            })
    except Exception as e:
        logger.error(f"Error analyzing pharmaceutical market: {str(e)}")
        return f"Error: {str(e)}"

# Diagnostic Segment Tools
@mcp_server.tool()
def analyze_diagnostic_market(test_type: str = None) -> str:
    """Analyze the diagnostic market for specific test types"""
    try:
        logger.info(f"Analyzing diagnostic market for test type: {test_type}")
        
        # Mock implementation for diagnostic market analysis
        test_types = {
            "imaging": {
                "market_size": "$28.9 billion",
                "growth_rate": "6.7%",
                "trends": ["AI integration", "Portable devices", "Cloud-based storage"]
            },
            "molecular": {
                "market_size": "$32.5 billion",
                "growth_rate": "9.3%",
                "trends": ["PCR alternatives", "CRISPR diagnostics", "Direct-to-consumer"]
            },
            "point_of_care": {
                "market_size": "$19.7 billion",
                "growth_rate": "8.8%",
                "trends": ["Smartphone connectivity", "Wearable integration", "Rural healthcare"]
            }
        }
        
        if test_type and test_type.lower() in test_types:
            return json.dumps(test_types[test_type.lower()])
        else:
            # Return overview of all categories
            return json.dumps({
                "total_market_size": "$82.3 billion",
                "overall_growth": "7.9%",
                "test_types": list(test_types.keys())
            })
    except Exception as e:
        logger.error(f"Error analyzing diagnostic market: {str(e)}")
        return f"Error: {str(e)}"

# Supplement Segment Tools
@mcp_server.tool()
def analyze_supplement_market(supplement_type: str = None) -> str:
    """Analyze the supplement market for specific supplement types"""
    try:
        logger.info(f"Analyzing supplement market for type: {supplement_type}")
        
        # Mock implementation for supplement market analysis
        supplement_types = {
            "vitamins": {
                "market_size": "$15.8 billion",
                "growth_rate": "7.2%",
                "trends": ["Gummy formats", "Subscription services", "Personalization"]
            },
            "proteins": {
                "market_size": "$18.9 billion",
                "growth_rate": "8.5%",
                "trends": ["Plant-based options", "Ready-to-drink", "Clean label"]
            },
            "probiotics": {
                "market_size": "$8.7 billion",
                "growth_rate": "10.2%",
                "trends": ["Microbiome testing", "Strain-specific benefits", "Shelf stability"]
            }
        }
        
        if supplement_type and supplement_type.lower() in supplement_types:
            return json.dumps(supplement_types[supplement_type.lower()])
        else:
            # Return overview of all categories
            return json.dumps({
                "total_market_size": "$55.2 billion",
                "overall_growth": "8.4%",
                "supplement_types": list(supplement_types.keys())
            })
    except Exception as e:
        logger.error(f"Error analyzing supplement market: {str(e)}")
        return f"Error: {str(e)}"

# Medical Device Segment Tools
@mcp_server.tool()
def analyze_medical_device_market(device_category: str = None) -> str:
    """Analyze the medical device market for specific device categories"""
    try:
        logger.info(f"Analyzing medical device market for category: {device_category}")
        
        # Mock implementation for medical device market analysis
        device_categories = {
            "monitoring": {
                "market_size": "$32.4 billion",
                "growth_rate": "6.8%",
                "trends": ["Remote patient monitoring", "AI diagnostics", "Wearables"]
            },
            "surgical": {
                "market_size": "$41.7 billion",
                "growth_rate": "5.9%",
                "trends": ["Robotic assistance", "Minimally invasive", "3D printing"]
            },
            "implantable": {
                "market_size": "$28.2 billion",
                "growth_rate": "7.3%",
                "trends": ["Biocompatible materials", "Smart implants", "Extended lifespans"]
            }
        }
        
        if device_category and device_category.lower() in device_categories:
            return json.dumps(device_categories[device_category.lower()])
        else:
            # Return overview of all categories
            return json.dumps({
                "total_market_size": "$118.6 billion",
                "overall_growth": "6.5%",
                "device_categories": list(device_categories.keys())
            })
    except Exception as e:
        logger.error(f"Error analyzing medical device market: {str(e)}")
        return f"Error: {str(e)}"

# Generic visualization tool for all segments
@mcp_server.tool()
def generate_market_visualization(segment: str, visualization_type: str, title: str) -> str:
    """Generate visualizations for market data"""
    try:
        logger.info(f"Generating {visualization_type} visualization for {segment} segment with title: {title}")
        
        # In a real implementation, this would generate and save actual visualizations
        # For now, return a mock response describing what would be generated
        
        return json.dumps({
            "status": "success",
            "description": f"Generated {visualization_type} for {segment} titled '{title}'",
            "image_path": f"/visualizations/{segment}_{visualization_type}_{int(time.time())}.png"
        })
    except Exception as e:
        logger.error(f"Error generating visualization: {str(e)}")
        return f"Error: {str(e)}"

# Run the server
if __name__ == "__main__":
    port = Config.MCP_PORT  # Use port from config
    logger.info(f"Starting Unified MCP Server on port {port}")
    
    # Mount the MCP server's SSE app to the FastAPI app
    app.mount("/", mcp_server.sse_app())
    
    # Run the FastAPI app with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
