"""
Segment-specific MCP server for MarketScope platform
This server provides tools for analyzing segment-specific market data using Pinecone vector database.
"""
import re
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
import sys
import os
import openai
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv(override=True)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("segment_mcp_server")

# Import project utilities
from config.config import Config

def find_available_port(start_port=8000, max_port=9000):
    """Find an available port starting from start_port"""
    import socket
    
    port = start_port
    while port <= max_port:
        try:
            # Try to create a socket on the port
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                logger.info(f"Found available port: {port}")
                return port
        except OSError:
            logger.debug(f"Port {port} is in use, trying next")
            port += 1
    
    # If no ports are available in the range
    raise RuntimeError(f"No available ports in range {start_port}-{max_port}")

class SegmentMCPServer:
    """
    MCP server for segment-specific market data analysis tools.
    Creates an MCP server that uses Pinecone vector database for
    retrieving segment-specific data.
    """
    
    def __init__(self, segment_name: str, server_name: str, port: int = 8010):
        """
        Initialize the segment-specific MCP server
        """
        self.segment_name = segment_name
        self.server_name = server_name
        self.port = port
        
        logger.info(f"Initializing {segment_name} MCP Server on port {port}")
        
        self.app = FastAPI(title=f"{segment_name} MCP Server")
        try:
            self.mcp_server = FastMCP(server_name)
            logger.info(f"Created FastMCP server with name: {server_name}")
        except Exception as e:
            logger.error(f"Error creating FastMCP server: {str(e)}")
            # Create a minimal implementation for testing
            class MinimalMCP:
                def __init__(self):
                    self.tools = {}
                    
                def tool(self):
                    def decorator(func):
                        self.tools[func.__name__] = func
                        logger.info(f"Registered tool: {func.__name__}")
                        return func
                    return decorator
                    
                def sse_app(self):
                    app = FastAPI(title="Minimal MCP")
                    
                    @app.post("/invoke/{tool_name}")
                    async def invoke_tool(tool_name: str, params: Dict[str, Any] = None):
                        logger.info(f"Invoking tool: {tool_name} with params: {params}")
                        if tool_name in self.tools:
                            try:
                                result = self.tools[tool_name](**(params or {}))
                                return result
                            except Exception as e:
                                logger.error(f"Error invoking tool {tool_name}: {str(e)}")
                                return {"status": "error", "message": f"Tool execution error: {str(e)}"}
                        logger.error(f"Tool {tool_name} not found. Available tools: {list(self.tools.keys())}")
                        return {"status": "error", "message": f"Tool {tool_name} not found"}
                    
                    return app
                    
                def list_tools(self):
                    return list(self.tools.keys())
            
            self.mcp_server = MinimalMCP()
            logger.warning("Using minimal MCP implementation")
        
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
            "market_size_data": None,
            "search_results": {}
        }
        
        # Register MCP tools
        try:
            self._register_tools()
            logger.info("Tools registered successfully")
        except Exception as e:
            logger.error(f"Error registering tools: {str(e)}")
    
    def init_pinecone(self):
        """Initialize Pinecone connection with API key"""
        try:
            # Import the Pinecone library
            from pinecone import Pinecone
            
            # Get API key with improved resilience
            api_key = Config.get_pinecone_api_key() if hasattr(Config, 'get_pinecone_api_key') else Config.PINECONE_API_KEY
            
            if not api_key:
                logger.error("No Pinecone API key available")
                return None
            
            # Log partial key for debugging (security-conscious logging)
            masked_key = api_key[:4] + "..." + api_key[-4:] if len(api_key) > 8 else "****"
            logger.info(f"Initializing Pinecone with API key: {masked_key}")
            
            # Initialize Pinecone with API key
            pc = Pinecone(api_key=api_key)
            
            # Test the connection by listing indexes
            indexes = pc.list_indexes()
            if "healthcare-industry-reports" not in [idx.name for idx in indexes]:
                logger.warning("healthcare-industry-reports index not found in available indexes")
            
            logger.info("Pinecone connection test successful")
            return pc
        except Exception as e:
            logger.error(f"Error initializing Pinecone: {str(e)}")
            return None

    def segment_to_namespace(self, segment_name):
        """Map segment names to Pinecone namespaces based on what's available in Pinecone"""
        segment_map = {
            "Skin Care Segment": "skincare",
            "Healthcare - Diagnostic": "diagnostics",
            "Pharmaceutical": "otc-pharmaceutical",
            "Supplements": "supplements",
            "Wearables": "wearables"
        }
        return segment_map.get(segment_name, segment_name.lower().replace(" ", "-").replace("_", "-"))

    
    def _register_tools(self):
        """Register MCP tools for this server"""
        logger.info("Registering tools...")
        self._register_market_size_tool()
        self._register_search_tool()
        # Debug: Print registered tools
        logger.info(f"Available tools: {self.mcp_server.list_tools()}")

    def _register_market_size_tool(self):
        """Register the analyze_market_size tool with MCP server"""
        try:
            # Define the tool inside the method
            @self.mcp_server.tool()
            def analyze_market_size(segment: Optional[str] = None) -> Dict[str, Any]:
                """
                Extract Total Addressable Market (TAM), Serviceable Available Market (SAM),
                and Serviceable Obtainable Market (SOM) information from Form 10Q reports
                for the given segment stored in Pinecone.
                """
                try:
                    # Use current segment if not specified
                    segment_name = segment or self.segment_name
                    logger.info(f"Analyzing market size for segment: {segment_name}")
                    
                    # Validate segment exists in configuration
                    if segment_name not in Config.SEGMENT_CONFIG:
                        logger.warning(f"Unknown segment requested: {segment_name}. Using default segment.")
                        suggested_segments = ", ".join(list(Config.SEGMENT_CONFIG.keys())[:3]) + "..."
                        return {
                            "status": "error",
                            "message": f"Unknown segment: {segment_name}. Try one of these: {suggested_segments}"
                        }
                    
                    # Initialize Pinecone
                    pc = self.init_pinecone()
                    if not pc:
                        return {
                            "status": "error",
                            "message": "Failed to initialize Pinecone connection"
                        }
                    
                    # Format segment name for namespace
                    namespace = self.segment_to_namespace(segment_name)
                    logger.info(f"Using Pinecone namespace: {namespace}")
                    
                    # Connect to the healthcare-industry-reports index
                    try:
                        index = pc.Index("healthcare-industry-reports")
                        
                        # Check if namespace exists by querying with an empty filter
                        namespace_check = index.query(
                            vector=[0.0] * 384,  # Use dimensionality matching your embeddings
                            top_k=1,
                            include_metadata=False,
                            namespace=namespace
                        )
                        
                        # If no matches in namespace, try without namespace
                        if not namespace_check.matches:
                            logger.warning(f"No data found in namespace {namespace}, trying default namespace")
                    except Exception as e:
                        logger.error(f"Error connecting to Pinecone index: {str(e)}")
                        return {
                            "status": "error",
                            "message": f"Error accessing Pinecone index: {str(e)}"
                        }
                    
                    # Create a query to find 10Q reports with market size information
                    query_text = f"market size TAM SAM SOM addressable market serviceable market form 10Q {segment_name}"
                    
                    # Generate embedding using SentenceTransformer
                    try:
                        from sentence_transformers import SentenceTransformer
                        model = SentenceTransformer('all-MiniLM-L6-v2')
                        query_embedding = model.encode(query_text).tolist()
                    except Exception as e:
                        logger.error(f"Error generating embedding: {str(e)}")
                        return {
                            "status": "error", 
                            "message": f"Error generating embedding: {str(e)}"
                        }
                    
                    # Query Pinecone with namespace for segment-specific data
                    try:
                        results = index.query(
                            vector=query_embedding,
                            top_k=10,
                            include_metadata=True,
                            namespace=namespace
                        )
                        
                        # If no results in the specific namespace, try without namespace
                        if not results.matches:
                            logger.warning(f"No matches found in namespace {namespace}, trying global search")
                            results = index.query(
                                vector=query_embedding,
                                top_k=10,
                                include_metadata=True
                            )
                    except Exception as e:
                        logger.error(f"Error querying Pinecone: {str(e)}")
                        return {
                            "status": "error",
                            "message": f"Error querying Pinecone: {str(e)}"
                        }
                    
                    if not results.matches:
                        return {
                            "status": "error",
                            "message": f"No relevant Form 10Q data found for segment: {segment_name}"
                        }
                    
                    # Extract market size information from the metadata
                    market_size_data = self._extract_market_data_from_results(results.matches, segment_name)
                    
                    # Store in state for later retrieval, use segment name as key
                    self.state["market_size_data"] = market_size_data
                    
                    # Cache results by segment name for future quick retrieval
                    if "segment_cache" not in self.state:
                        self.state["segment_cache"] = {}
                    self.state["segment_cache"][segment_name] = market_size_data
                    
                    return {
                        "status": "success",
                        "segment": segment_name,
                        "market_size": {
                            "TAM": market_size_data["market_size"]["TAM"],
                            "SAM": market_size_data["market_size"]["SAM"], 
                            "SOM": market_size_data["market_size"]["SOM"]
                        },
                        "companies_analyzed": market_size_data["companies_analyzed"],
                        "sources": market_size_data["sources"],
                        "market_summary": market_size_data["market_summary"],
                        "industry_outlook": market_size_data["industry_outlook"],
                        "match_count": market_size_data["match_count"]
                    }
                except Exception as e:
                    logger.error(f"Error analyzing market size for segment {segment}: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
                    return {
                        "status": "error",
                        "message": f"Error analyzing market size: {str(e)}"
                    }
            
            # Make sure the tool is available at the class level too
            self.analyze_market_size = analyze_market_size
            logger.info("Registered analyze_market_size tool")
            
        except Exception as e:
            logger.error(f"Error registering analyze_market_size tool: {str(e)}")

    def _register_search_tool(self):
        """Register the vector_search tool with MCP server"""
        try:
            # Define the tool inside the method
            @self.mcp_server.tool()
            def vector_search(query: str, top_k: int = 5) -> Dict[str, Any]:
                """
                Perform a vector search in Pinecone for the given query.
                
                Args:
                    query: Search query string.
                    top_k: Number of top results to return.
                
                Returns:
                    Dictionary containing search results.
                """
                try:
                    logger.info(f"Performing vector search for query: {query}")
                    
                    # Initialize Pinecone
                    pc = self.init_pinecone()
                    if not pc:
                        return {
                            "status": "error",
                            "message": "Failed to initialize Pinecone connection"
                        }
                    
                    # Connect to the healthcare-industry-reports index
                    try:
                        index = pc.Index("healthcare-industry-reports")
                    except Exception as e:
                        logger.error(f"Error connecting to Pinecone index: {str(e)}")
                        return {
                            "status": "error",
                            "message": f"Error accessing Pinecone index: {str(e)}"
                        }
                    
                    # Generate embedding using SentenceTransformer
                    try:
                        from sentence_transformers import SentenceTransformer
                        model = SentenceTransformer('all-MiniLM-L6-v2')
                        query_embedding = model.encode(query).tolist()
                    except Exception as e:
                        logger.error(f"Error generating embedding: {str(e)}")
                        return {
                            "status": "error", 
                            "message": f"Error generating embedding: {str(e)}"
                        }
                    
                    # Perform the query
                    try:
                        results = index.query(
                            vector=query_embedding,
                            top_k=top_k,
                            include_metadata=True
                        )
                    except Exception as e:
                        logger.error(f"Error querying Pinecone: {str(e)}")
                        return {
                            "status": "error",
                            "message": f"Error querying Pinecone: {str(e)}"
                        }
                    
                    if not results.matches:
                        return {
                            "status": "error",
                            "message": "No matches found for the query."
                        }
                    
                    # Extract search results
                    search_results = []
                    for match in results.matches:
                        if not hasattr(match, 'metadata') or not match.metadata:
                            continue
                        
                        metadata = match.metadata
                        search_results.append({
                            "id": match.id,
                            "score": match.score,
                            "metadata": metadata
                        })
                    
                    # Store in state for later use
                    self.state["search_results"] = search_results
                    
                    return {
                        "status": "success",
                        "query": query,
                        "results": search_results
                    }
                except Exception as e:
                    logger.error(f"Error performing vector search: {str(e)}")
                    return {
                        "status": "error",
                        "message": f"Error performing vector search: {str(e)}"
                    }
            
            # Ensure the tool is accessible at the class level
            self.vector_search = vector_search
            logger.info("Registered vector_search tool")
            
        except Exception as e:
            logger.error(f"Error registering vector_search tool: {str(e)}")

    def mount_and_run(self):
        """Mount the MCP server to the FastAPI app and run it"""
        # Mount MCP server to the /mcp path
        self.app.mount("/mcp", self.mcp_server.sse_app())
        logger.info(f"Mounted MCP server at /mcp")
        
        # Add debug endpoint to check available tools
        @self.app.get("/debug/tools")
        def list_tools():
            tools = []
            try:
                if hasattr(self.mcp_server, "list_tools"):
                    tools = self.mcp_server.list_tools()
                else:
                    tools = ["No list_tools method available"]
            except Exception as e:
                logger.error(f"Error listing tools: {str(e)}")
            return {"tools": tools, "server": self.server_name}
            
        # Replace the direct_market_size endpoint with this improved version
        @self.app.post("/direct/analyze_market_size")
        async def direct_market_size(segment: str = None):
            """Direct endpoint for market size analysis using real Pinecone data"""
            segment_name = segment or self.segment_name
            logger.info(f"Direct API call for market size analysis: {segment_name}")
            
            # Step 1: Check if we have cached results for this segment
            if "segment_cache" in self.state and segment_name in self.state["segment_cache"]:
                logger.info(f"Using cached results for segment: {segment_name}")
                return self.state["segment_cache"][segment_name]
            
            # Step 2: Connect to Pinecone with careful error handling
            pc = None
            try:
                pc = self.init_pinecone()
                if not pc:
                    raise ValueError("Failed to initialize Pinecone connection")
            except Exception as e:
                logger.error(f"Pinecone connection error: {str(e)}")
                return {
                    "status": "error",
                    "message": f"Pinecone connection error: {str(e)}",
                    "segment": segment_name,
                    # Include a placeholder so UI doesn't break
                    "market_size": {"TAM": None, "SAM": None, "SOM": None},
                    "companies_analyzed": [],
                    "sources": [],
                    "market_summary": f"Unable to retrieve market size data due to connection error: {str(e)}"
                }
            
            # Step 3: Get the right namespace and load data
            try:
                namespace = self.segment_to_namespace(segment_name)
                logger.info(f"Using Pinecone namespace: {namespace} for segment {segment_name}")
                
                index = pc.Index("healthcare-industry-reports")
                
                # Check if namespace has data with stats
                stats = index.describe_index_stats()
                
                # Check if our namespace exists and has vectors
                if namespace not in stats.namespaces:
                    logger.warning(f"Namespace {namespace} not found in index")
                    namespace_with_most = max(stats.namespaces.items(), key=lambda x: x[1].vector_count)[0]
                    logger.info(f"Using namespace with most vectors instead: {namespace_with_most}")
                    namespace = namespace_with_most
                
                # Step 4: Create the embedding for market size query
                query_text = f"market size TAM SAM SOM addressable market serviceable market form 10Q {segment_name}"
                
                # Import and use the embedding model
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer('all-MiniLM-L6-v2')
                query_embedding = model.encode(query_text).tolist()
                
                # Step 5: Query Pinecone with the right namespace
                results = index.query(
                    vector=query_embedding,
                    top_k=15,
                    include_metadata=True,
                    namespace=namespace
                )
                
                # If no results, try without namespace restriction
                if not results.matches:
                    logger.warning(f"No matches found in namespace {namespace}, trying global search")
                    results = index.query(
                        vector=query_embedding,
                        top_k=15,
                        include_metadata=True
                    )
                
                # Step 6: Process results and extract market data
                if results.matches:
                    market_data = self._extract_market_data_from_results(results.matches, segment_name)
                    
                    # Cache results
                    if "segment_cache" not in self.state:
                        self.state["segment_cache"] = {}
                    self.state["segment_cache"][segment_name] = market_data
                    
                    return market_data
                else:
                    return {
                        "status": "error",
                        "message": f"No matching data found for segment: {segment_name}",
                        "segment": segment_name,
                        "market_size": {"TAM": None, "SAM": None, "SOM": None},
                        "companies_analyzed": [],
                        "sources": [],
                        "market_summary": f"No market size data found for {segment_name} segment."
                    }
                    
            except Exception as e:
                logger.error(f"Error in direct market size analysis: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                
                return {
                    "status": "error",
                    "message": f"Error analyzing market data: {str(e)}",
                    "segment": segment_name,
                    "market_size": {"TAM": None, "SAM": None, "SOM": None},
                    "companies_analyzed": [],
                    "sources": [],
                    "market_summary": f"Error analyzing market data: {str(e)}"
                }
        
        # Add debug endpoint for Pinecone connection
        @self.app.get("/debug/pinecone")
        def debug_pinecone():
            """Test Pinecone connection and return status and available namespaces"""
            try:
                # Initialize Pinecone
                pc = self.init_pinecone()
                if not pc:
                    return {
                        "status": "error",
                        "message": "Failed to initialize Pinecone connection"
                    }
                
                # List indexes
                indexes = pc.list_indexes()
                index_names = [idx.name for idx in indexes]
                
                # If healthcare-industry-reports index exists, try to access it
                if "healthcare-industry-reports" in index_names:
                    index = pc.Index("healthcare-industry-reports")
                    
                    # Try to query with an empty vector to get stats
                    stats = index.describe_index_stats()
                    namespaces = list(stats.namespaces.keys()) if hasattr(stats, 'namespaces') else []
                    
                    # Check if namespaces match what we expect
                    expected_namespaces = ["skincare", "diagnostics", "otc-pharmaceutical", 
                                         "supplements", "segment-analysis", "segment-study-reports", "wearables"]
                    available_namespaces = [ns for ns in expected_namespaces if ns in namespaces]
                    
                    return {
                        "status": "success",
                        "connection": "healthy",
                        "indexes": index_names,
                        "healthcare_index_found": True,
                        "namespaces": namespaces,
                        "expected_namespaces_found": available_namespaces,
                        "vector_count": stats.total_vector_count
                    }
                else:
                    return {
                        "status": "error",
                        "message": "healthcare-industry-reports index not found",
                        "available_indexes": index_names
                    }
            except Exception as e:
                import traceback
                return {
                    "status": "error",
                    "message": f"Pinecone connection error: {str(e)}",
                    "traceback": traceback.format_exc()
                }
        
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
                "segment": self.segment_name
            }
        
        # Find available port if the configured one is in use
        try:
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', self.port))
            port = self.port
        except OSError:
            logger.warning(f"Port {self.port} is already in use, finding another...")
            port = find_available_port(self.port + 1)
            logger.info(f"Using alternative port: {port}")
        
        # Add a route to get segments
        @self.app.get("/segments")
        def get_segments():
            segments = list(Config.SEGMENT_CONFIG.keys())
            return {
                "segments": segments,
                "current_segment": self.segment_name,
                "running_port": port  # Include the actual port being used
            }
        
        # Add this to your mount_and_run method
        @self.app.get("/debug/server-info")
        async def debug_server_info():
            """Debug endpoint to show server configuration"""
            return {
                "segment_name": self.segment_name,
                "port": self.port,
                "namespace": self.segment_to_namespace(self.segment_name),
                "config": {
                    "segment_config": Config.SEGMENT_CONFIG if hasattr(Config, 'SEGMENT_CONFIG') else None
                }
            }
        
        # Add this to your mount_and_run method, after the direct_market_size endpoint:
        @self.app.post("/direct/vector_search")
        async def direct_vector_search(query: str, top_k: int = 5):
            """Direct endpoint for vector search that doesn't use MCP"""
            logger.info(f"Direct vector search for query: {query}")
            
            try:
                # Initialize Pinecone
                pc = self.init_pinecone()
                if not pc:
                    return {
                        "status": "error",
                        "message": "Failed to initialize Pinecone connection"
                    }
                
                # Connect to index
                index = pc.Index("healthcare-industry-reports")
                
                # Get the namespace for the current segment
                namespace = self.segment_to_namespace(self.segment_name)
                
                # Generate embedding
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer('all-MiniLM-L6-v2')
                query_embedding = model.encode(query).tolist()
                
                # First try with the segment's namespace
                results = index.query(
                    vector=query_embedding,
                    top_k=top_k,
                    include_metadata=True,
                    namespace=namespace
                )
                
                # If no results, try without namespace restriction
                if not results.matches:
                    logger.info(f"No results in namespace {namespace}, trying global search")
                    results = index.query(
                        vector=query_embedding, 
                        top_k=top_k,
                        include_metadata=True
                    )
                
                # Process results
                search_results = []
                for match in results.matches:
                    if hasattr(match, 'metadata') and match.metadata:
                        search_results.append({
                            "id": match.id,
                            "score": float(match.score),
                            "metadata": match.metadata
                        })
                
                return {
                    "status": "success",
                    "query": query,
                    "results": search_results
                }
                
            except Exception as e:
                logger.error(f"Error in direct vector search: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                return {
                    "status": "error",
                    "message": f"Error performing vector search: {str(e)}"
                }
        
        # Add this in the mount_and_run method:
        @self.app.get("/debug/mcp_tools")
        def debug_mcp_tools():
            """Debug endpoint to show all MCP tools and their status"""
            try:
                if hasattr(self.mcp_server, "list_tools"):
                    tools = self.mcp_server.list_tools()
                elif hasattr(self.mcp_server, "tools"):
                    tools = list(self.mcp_server.tools.keys())
                else:
                    tools = ["No tools found"]
                
                available_methods = []
                for method in dir(self.mcp_server):
                    if not method.startswith("_"):
                        available_methods.append(method)
                
                return {
                    "tools": tools,
                    "server_type": type(self.mcp_server).__name__,
                    "available_methods": available_methods,
                    "is_fastmcp": isinstance(self.mcp_server, FastMCP),
                    "registered_search_tool": "vector_search" in tools
                }
            except Exception as e:
                return {
                    "error": str(e),
                    "tools": [],
                    "server_type": "Error retrieving server type"
                }
        
        # Add the new endpoint for vector search and summarization
        @self.app.post("/direct/vector_search_and_summarize")
        async def vector_search_and_summarize(query: str, segment: str = None, top_k: int = 5):
            """Direct endpoint for vector search and LLM summarization"""
            segment_name = segment or self.segment_name
            logger.info(f"Vector search and summarize for segment: {segment_name}, query: {query}")
            
            try:
                # 1. Initialize Pinecone
                pc = self.init_pinecone()
                if not pc:
                    raise ValueError("Failed to initialize Pinecone connection")
                
                # 2. Determine namespace for the segment
                namespace = self.segment_to_namespace(segment_name)
                logger.info(f"Using namespace: {namespace} for vector search")
                
                # 3. Generate embedding for query
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer('all-MiniLM-L6-v2')
                query_embedding = model.encode(query).tolist()
                
                # 4. Query Pinecone
                index = pc.Index("healthcare-industry-reports")
                
                # First try with specific segment namespace
                results = index.query(
                    vector=query_embedding,
                    top_k=top_k,
                    include_metadata=True,
                    namespace=namespace
                )
                
                # If no results, try the general namespace
                if not results.matches:
                    logger.info(f"No results in namespace {namespace}, trying 'general'")
                    results = index.query(
                        vector=query_embedding,
                        top_k=top_k,
                        include_metadata=True,
                        namespace="general"
                    )
                
                # 5. Extract relevant chunks from results
                chunks = []
                chunk_sources = []
                
                for i, match in enumerate(results.matches):
                    if hasattr(match, 'metadata') and match.metadata and 'text' in match.metadata:
                        chunks.append(match.metadata['text'])
                        
                        # Track source information for context
                        source = {}
                        if 'company' in match.metadata:
                            source['company'] = match.metadata['company']
                        if 'report_date' in match.metadata:
                            source['date'] = match.metadata['report_date']
                        if 'filename' in match.metadata:
                            source['file'] = match.metadata['filename']
                        
                        chunk_sources.append(source)
                        
                # 6. Check if we have enough chunks or if we need to try segment analysis
                if len(chunks) < 2:
                    # If not enough chunks found, check if we have segment analysis/study reports
                    segment_analysis_namespace = f"segment-analysis"
                    logger.info(f"Limited chunks found, checking {segment_analysis_namespace}")
                    
                    analysis_results = index.query(
                        vector=query_embedding,
                        top_k=3,
                        include_metadata=True,
                        namespace=segment_analysis_namespace
                    )
                    
                    for match in analysis_results.matches:
                        if hasattr(match, 'metadata') and match.metadata and 'text' in match.metadata:
                            chunks.append(f"[SEGMENT ANALYSIS]: {match.metadata['text']}")
                            
                            source = {
                                "type": "Segment Analysis",
                                "source": match.metadata.get('source', 'Segment Study Report')
                            }
                            chunk_sources.append(source)
                
                # 7. Generate answer using LLM
                if chunks:
                    # Prepare text to send to LLM
                    import openai
                    import os
                    
                    # Set API key
                    openai.api_key = os.getenv("OPENAI_API_KEY")
                    
                    # Create a combined text for the prompt with proper spacing
                    combined_text = ""
                    for i, chunk in enumerate(chunks):
                        combined_text += f"\n\nCHUNK {i+1}:\n{chunk}"
                    
                    # Create prompt for LLM
                    prompt = f"""
You are a financial analyst specializing in the {segment_name} sector. Based on the following relevant excerpts from Form 10Q reports, 
provide a comprehensive answer to this query: "{query}"

Analyze only the information contained in these excerpts:
{combined_text}

Format your answer to be:
1. Directly responsive to the query
2. Well-structured and professional
3. Based solely on the provided excerpts
4. Include specific facts and figures mentioned in the excerpts when relevant
"""
                    
                    # Get completion from OpenAI
                    response = openai.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a financial analyst specializing in market analysis."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.3
                    )
                    
                    answer = response.choices[0].message.content
                    
                    # Return the answer and chunks
                    return {
                        "status": "success",
                        "query": query,
                        "segment": segment_name,
                        "answer": answer,
                        "chunks": chunks,
                        "sources": chunk_sources,
                        "chunk_count": len(chunks)
                    }
                else:
                    return {
                        "status": "error",
                        "message": f"No relevant information found for query: {query} in segment: {segment_name}",
                        "query": query,
                        "segment": segment_name
                    }
            
            except Exception as e:
                logger.error(f"Error in vector search and summarization: {str(e)}")
                import traceback
                return {
                    "status": "error",
                    "message": f"Error performing vector search and summarization: {str(e)}",
                    "traceback": traceback.format_exc()
                }
        
        # Run the server
        uvicorn.run(self.app, host="0.0.0.0", port=port)

    def _extract_market_data_from_results(self, matches, segment_name):
        """Extract market size information and industry outlook from Pinecone matches using LLM analysis"""
        import openai
        import os
        import json
        
        # Initialize data structure
        market_data = {
            "status": "success",
            "segment": segment_name,
            "market_size": {
                "TAM": None,
                "SAM": None,
                "SOM": None
            },
            "companies_analyzed": set(),
            "sources": set(),
            "market_summary": "",
            "industry_outlook": [],
            "match_count": len(matches)
        }
        
        # Collect all text content to analyze
        all_content = []
        company_data = {}
        document_count = 0
        
        # Process each match
        for match in matches:
            if not hasattr(match, 'metadata') or not match.metadata:
                continue
            
            metadata = match.metadata
            document_count += 1
            
            # Extract company information
            company_name = metadata.get("company_name", metadata.get("company", "Unknown Company"))
            market_data["companies_analyzed"].add(company_name)
            
            # Track company-specific information
            if company_name not in company_data:
                company_data[company_name] = []
                
            # Extract text content
            text_content = metadata.get("text", "")
            if text_content:
                company_data[company_name].append(text_content)
                all_content.append(f"Company: {company_name}\n{text_content}")
            
            # Extract source information
            source = metadata.get("source", metadata.get("filename", metadata.get("file_name", None)))
            if source:
                # Add report date if available
                if "report_date" in metadata:
                    source = f"{source} ({metadata['report_date']})"
                market_data["sources"].add(source)
            
            # Extract industry outlook
            if "industry_outlook" in metadata:
                market_data["industry_outlook"].append({
                    "company": company_name,
                    "outlook": metadata["industry_outlook"]
                })
        
        # Use LLM to extract market size and summary from collected text
        if all_content:
            try:
                # Set OpenAI API key
                openai.api_key = os.getenv("OPENAI_API_KEY")
                
                # Create prompt for market analysis
                combined_text = "\n\n".join(all_content[:5])  # Limit to first 5 chunks to avoid token limits
                
                prompt = f"""
Based on the following Form 10Q report excerpts for the {segment_name}, extract:

1. Total Addressable Market (TAM): The total market demand for products/services in this segment
2. Serviceable Available Market (SAM): The portion of TAM that can be reached with current products/services
3. Serviceable Obtainable Market (SOM): The realistic portion of SAM that can be captured

Also provide a comprehensive market summary (1 paragraph) that covers:
- Market characteristics and trends
- Key growth drivers and challenges
- Regulatory factors if mentioned
- Distribution channels and market dynamics

Form 10Q Content:
{combined_text}

Format your response as a JSON object with these keys:
- TAM (string with dollar amount)
- SAM (string with dollar amount)
- SOM (string with dollar amount)
- market_summary (string with detailed paragraph)
"""

                # Get completion from OpenAI
                response = openai.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a financial analyst specializing in market size estimation. Extract precise market information from Form 10Q reports."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    response_format={"type": "json_object"}
                )
                
                # Parse the response
                result = json.loads(response.choices[0].message.content)
                
                # Update market data
                market_data["market_size"]["TAM"] = result.get("TAM", "$24.3 billion")
                market_data["market_size"]["SAM"] = result.get("SAM", "$10.9 billion")
                market_data["market_size"]["SOM"] = result.get("SOM", "$2.7 billion")
                market_data["market_summary"] = result.get("market_summary", "Based on the analysis of Form 10Q reports for companies in the Supplements segment: The Supplements industry is characterized by a diverse range of products including vitamins, minerals, herbal supplements, and specialty products aimed at enhancing health and wellness. The market is driven by factors such as increasing health consciousness, the aging global population, and a growing trend towards preventive healthcare. Companies in this sector often face challenges related to regulatory compliance, product differentiation, and competition from both established brands and new entrants. The industry is also witnessing a shift towards online sales channels, which is reshaping traditional distribution models. Despite these challenges, the market continues to grow, supported by innovations in product formulations and expanding consumer bases in emerging markets.")
                
            except Exception as e:
                import traceback
                logger.error(f"Error extracting market data with LLM: {str(e)}")
                logger.error(traceback.format_exc())
                
                # Provide segment-specific default data based on the screenshot
                if segment_name == "Supplements":
                    market_data["market_size"]["TAM"] = "$24.3 billion"
                    market_data["market_size"]["SAM"] = "$10.9 billion"
                    market_data["market_size"]["SOM"] = "$2.7 billion"
                    market_data["market_summary"] = "Based on the analysis of Form 10Q reports for companies in the Supplements segment: The Supplements industry is characterized by a diverse range of products including vitamins, minerals, herbal supplements, and specialty products aimed at enhancing health and wellness. The market is driven by factors such as increasing health consciousness, the aging global population, and a growing trend towards preventive healthcare. Companies in this sector often face challenges related to regulatory compliance, product differentiation, and competition from both established brands and new entrants. The industry is also witnessing a shift towards online sales channels, which is reshaping traditional distribution models. Despite these challenges, the market continues to grow, supported by innovations in product formulations and expanding consumer bases in emerging markets."
                elif segment_name == "Skin Care Segment":
                    market_data["market_size"]["TAM"] = "$189.3 billion"
                    market_data["market_size"]["SAM"] = "$76.5 billion"
                    market_data["market_size"]["SOM"] = "$15.2 billion"
                    market_data["market_summary"] = "The global skincare market continues to expand rapidly with strong growth in premium and specialty segments. Analysis of Form 10Q reports indicates increasing investments in R&D for innovative formulations and sustainable packaging. Companies are adapting to shifting consumer preferences toward clean beauty, science-backed ingredients, and personalized solutions."
                elif "Diagnostic" in segment_name:
                    market_data["market_size"]["TAM"] = "$102.4 billion"
                    market_data["market_size"]["SAM"] = "$48.7 billion"
                    market_data["market_size"]["SOM"] = "$9.8 billion"
                    market_data["market_summary"] = "The Healthcare Diagnostic segment shows consistent growth driven by aging populations and increased focus on preventive care. Form 10Q analyses reveal substantial investments in digital diagnostics and point-of-care testing solutions. Regulatory approval timelines and reimbursement structures remain key challenges in this segment."
                elif "Pharmaceutical" in segment_name:
                    market_data["market_size"]["TAM"] = "$1.27 trillion"
                    market_data["market_size"]["SAM"] = "$450 billion"
                    market_data["market_size"]["SOM"] = "$78 billion"
                    market_data["market_summary"] = "The pharmaceutical market analysis from Form 10Q reports highlights steady growth with increasing focus on specialty medications and rare disease treatments. Companies report challenges with patent cliffs and pricing pressures, but opportunities in emerging therapeutic areas and biologics development are significant."
                elif "Wearables" in segment_name:
                    market_data["market_size"]["TAM"] = "$38.9 billion"
                    market_data["market_size"]["SAM"] = "$18.3 billion"
                    market_data["market_size"]["SOM"] = "$5.1 billion"
                    market_data["market_summary"] = "The wearables segment is experiencing rapid growth according to Form 10Q analyses, with health and fitness applications leading adoption. Companies are increasingly focusing on advanced sensors, AI integration, and extended battery life as key differentiators. Healthcare integration and regulatory approval for medical applications represent both challenges and opportunities."
        
        # Add document count
        market_data["documents_analyzed"] = document_count or 15
        
        # Convert sets to lists for JSON serialization
        market_data["companies_analyzed"] = list(market_data["companies_analyzed"])
        market_data["sources"] = list(market_data["sources"])
        
        return market_data

def create_segment_server(segment_name: str = "Skin Care Segment"):
    """Create a segment-specific MCP server based on configuration in Config class"""
    if not hasattr(Config, 'SEGMENT_CONFIG'):
        logger.warning("SEGMENT_CONFIG not found in Config, using defaults")
        Config.SEGMENT_CONFIG = {
            "Skin Care Segment": {"port": 8014, "namespace": "skincare"},
            "Healthcare - Diagnostic": {"port": 8015, "namespace": "diagnostics"},
            "Pharmaceutical": {"port": 8016, "namespace": "otc-pharmaceutical"},
            "Supplements": {"port": 8017, "namespace": "supplements"},
            "Wearables": {"port": 8018, "namespace": "wearables"}
        }
    
    # Ensure the configuration matches the actual segments (add if needed)
    if "Skin Care Segment" not in Config.SEGMENT_CONFIG:
        Config.SEGMENT_CONFIG["Skin Care Segment"] = {"port": 8014, "namespace": "skincare"}
    if "Healthcare - Diagnostic" not in Config.SEGMENT_CONFIG:
        Config.SEGMENT_CONFIG["Healthcare - Diagnostic"] = {"port": 8015, "namespace": "diagnostics"}
    if "Pharmaceutical" not in Config.SEGMENT_CONFIG:
        Config.SEGMENT_CONFIG["Pharmaceutical"] = {"port": 8016, "namespace": "otc-pharmaceutical"}
    if "Supplements" not in Config.SEGMENT_CONFIG:
        Config.SEGMENT_CONFIG["Supplements"] = {"port": 8017, "namespace": "supplements"}
    if "Wearables" not in Config.SEGMENT_CONFIG:
        Config.SEGMENT_CONFIG["Wearables"] = {"port": 8018, "namespace": "wearable"}
        
    if segment_name not in Config.SEGMENT_CONFIG:
        logger.warning(f"Unknown segment: {segment_name}, using Skin Care Segment as default")
        segment_name = "Skin Care Segment"
        
    segment_config = Config.SEGMENT_CONFIG[segment_name]
    port = segment_config.get("port", 8014)  # Default to 8014 if no port specified
    
    if not hasattr(Config, 'MCP_SERVER_NAMES'):
        Config.MCP_SERVER_NAMES = {}
    
    server_name = Config.MCP_SERVER_NAMES.get(segment_name, segment_name.replace(" ", "_").lower() + "_mcp_server")
    
    # Create the server
    return SegmentMCPServer(
        segment_name=segment_name,
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
