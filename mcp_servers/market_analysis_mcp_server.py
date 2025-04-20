import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)
"""
Market Analysis MCP Server
Provides tools for market analysis and segment strategies using
RAG with Philip Kotler's Marketing Management book
"""
import json
import boto3
import logging
from fastapi import FastAPI, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from mcp.server.fastmcp import FastMCP
import uvicorn
from typing import Dict, Any, List, Optional, Union
from langsmith import traceable
from pydantic import BaseModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("market_analysis_mcp_server")

# Handle OpenAI import - try both new and old style
try:
    # Try new OpenAI API (v1.0.0+)
    from openai import OpenAI
    logger.info("Using new OpenAI API style (v1.0.0+)")
    has_new_openai = True
except ImportError:
    # Fall back to old OpenAI API
    import openai
    logger.info("Using old OpenAI API style (pre-v1.0.0)")
    has_new_openai = False
    
# Import configuration
from config.config import Config

# Get the OpenAI API key directly from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not found in environment variables! Using Config fallback.")
    OPENAI_API_KEY = Config.OPENAI_API_KEY

# Log the API key status (partial key for security)
if OPENAI_API_KEY:
    masked_key = OPENAI_API_KEY[:4] + "*****" + OPENAI_API_KEY[-4:] if len(OPENAI_API_KEY) > 8 else "*****"
    logger.info(f"Using OpenAI API key: {masked_key}")
else:
    logger.error("No OpenAI API key found! API calls will fail.")
    
# Initialize OpenAI client with explicit API key
if has_new_openai:
    # New API style
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
else:
    # Old API style
    openai.api_key = OPENAI_API_KEY
    openai_client = openai

# Create FastAPI app
app = FastAPI(title="Market Analysis MCP Server")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create MCP server
mcp_server = FastMCP("market_analysis")

# Global state for RAG operations
rag_state = {
    "retrieved_chunks": [],
    "last_query": None,
    "current_segment": None,
    "analysis_results": {}
}

@traceable(name="pinecone_search")
@mcp_server.tool()
def pinecone_search(query: str, top_k: int = 3) -> Union[List[str], Dict[str, str]]:
    """Search for relevant chunks in Pinecone using query embeddings."""
    try:
        logger.info(f"Searching Pinecone for: {query}")
        
        # Get embeddings for the query - handle both old and new OpenAI API styles
        try:
            if has_new_openai:
                # New API style
                response = openai_client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=[query]
                )
                query_embedding = response.data[0].embedding
            else:
                # Old API style
                response = openai_client.Embedding.create(
                    model="text-embedding-ada-002",
                    input=[query]
                )
                query_embedding = response["data"][0]["embedding"]
        except Exception as embed_error:
            logger.error(f"Error creating embeddings: {str(embed_error)}")
            return ["No relevant chunks found. Please try a different query."]
        
        # Use updated Pinecone import
        from pinecone import Pinecone
        
        # Initialize Pinecone with the updated API
        pc = Pinecone(api_key=Config.PINECONE_API_KEY)
        index_name = "healthcare-product-analytics"
        
        # Print debug info
        logger.info(f"Searching Pinecone index: {index_name}")
        
        # Get the index (assuming it already exists)
        index = pc.Index(index_name)
        
        # Try different namespaces - often "book-kotler" might not be available in all environments
        namespaces_to_try = ["book-kotler", "kotler", "marketing"]
        results = None
        
        for namespace in namespaces_to_try:
            try:
                logger.info(f"Trying namespace: {namespace}")
                # Search Pinecone with the current namespace
                results = index.query(
                    vector=query_embedding,
                    top_k=top_k,
                    include_metadata=True,
                    namespace=namespace
                )
                
                if results and results.matches and len(results.matches) > 0:
                    logger.info(f"Found {len(results.matches)} matches in namespace {namespace}")
                    break
            except Exception as ns_error:
                logger.warning(f"Error with namespace {namespace}: {str(ns_error)}")
                continue
        
        # If all namespaces failed, try without namespace
        if not results or not results.matches or len(results.matches) == 0:
            try:
                logger.info("Trying without namespace")
                results = index.query(
                    vector=query_embedding,
                    top_k=top_k,
                    include_metadata=True
                )
            except Exception as e:
                logger.warning(f"Error querying without namespace: {str(e)}")
        
        # Update state
        rag_state["last_query"] = query
        rag_state["retrieved_chunks"] = []
        
        # Get matches and their scores
        matches = []
        for match in results.matches:
            chunk_id = match.id
            score = match.score
            matches.append(f"{chunk_id} (score: {score:.4f})")
            
        # Print debug info about what we found
        logger.info(f"Found {len(results.matches)} matches: {matches}")
        
        # Return chunk IDs directly
        if not results.matches:
            return ["No relevant chunks found. Please try a different query."]
        return [match.id for match in results.matches]
    
    except Exception as e:
        logger.error(f"Error in pinecone_search: {str(e)}")
        return {"error": f"Error in pinecone_search: {str(e)}"}

@traceable(name="fetch_s3_chunk")
@mcp_server.tool()
def fetch_s3_chunk(chunk_id: str) -> str:
    """Fetch a specific chunk from the S3 chunks file."""
    logger.info(f"Fetching chunk: {chunk_id}")
    
    try:
        # Initialize S3 client with credentials
        s3_client = boto3.client(
            's3',
            aws_access_key_id=Config.AWS_SERVER_PUBLIC_KEY,
            aws_secret_access_key=Config.AWS_SERVER_SECRET_KEY,
            region_name=Config.AWS_REGION
        )
        
        # Get bucket and key from config
        bucket_name = Config.BUCKET_NAME
        key = Config.S3_CHUNKS_PATH + Config.S3_CHUNKS_FILE
        
        logger.info(f"Fetching from S3: bucket={bucket_name}, key={key}")
        
        # Get the JSON file from S3
        response = s3_client.get_object(Bucket=bucket_name, Key=key)
        chunks_data = json.loads(response['Body'].read().decode('utf-8'))
        
        if "chunks" in chunks_data and chunk_id in chunks_data["chunks"]:
            chunk = chunks_data["chunks"][chunk_id]
            chunk_text = chunk.get("text", "")
            
            # Ensure chunk_text is a string
            if not isinstance(chunk_text, str):
                chunk_text = str(chunk_text)
            
            # Add to retrieved chunks
            rag_state["retrieved_chunks"].append({
                "chunk_id": chunk_id,
                "content": chunk_text
            })
            
            logger.info(f"Successfully retrieved chunk {chunk_id}")
            return chunk_text
        else:
            return f"Error fetching chunk {chunk_id}: Chunk not found in chunks file."
            
    except Exception as e:
        logger.error(f"Error fetching chunk {chunk_id}: {str(e)}")
        return f"Error fetching chunk {chunk_id}: {str(e)}"

# Metadata and Aggregation Tools
@traceable(name="get_chunks_metadata")
@mcp_server.tool()
def get_chunks_metadata() -> Dict[str, Any]:
    """Get metadata about available chunks."""
    try:
        return {
            "retrieved_count": len(rag_state["retrieved_chunks"]),
            "last_query": rag_state["last_query"],
            "current_segment": rag_state["current_segment"]
        }
    except Exception as e:
        logger.error(f"Error getting chunks metadata: {str(e)}")
        return {"error": f"Error getting chunks metadata: {str(e)}"}

@traceable(name="get_all_retrieved_chunks")
@mcp_server.tool()
def get_all_retrieved_chunks() -> List[Dict[str, str]]:
    """Get all chunks that have been retrieved in this session."""
    return rag_state["retrieved_chunks"]

# Marketing Analysis Tools
@traceable(name="analyze_market_segment")
@mcp_server.tool()
def analyze_market_segment(segment_name: str, market_type: str = "healthcare") -> Dict[str, Any]:
    """Retrieve and aggregate relevant content for a market segment from Kotler's book."""
    try:
        logger.info(f"Analyzing market segment: {segment_name} in {market_type}")
        
        rag_state["current_segment"] = segment_name
        search_query = f"marketing segmentation strategy for {segment_name} in {market_type}"
        chunk_ids = pinecone_search(search_query, top_k=5)
        
        if isinstance(chunk_ids, list) and chunk_ids and not chunk_ids[0].startswith("Error"):
            chunk_contents = [fetch_s3_chunk(chunk_id) for chunk_id in chunk_ids]
            chunk_contents = [c for c in chunk_contents if not c.startswith("Error") and not c.startswith("Chunk")]
            
            if chunk_contents:
                result = {
                    "segment_name": segment_name,
                    "market_type": market_type,
                    "chunks": chunk_contents[:3],
                    "sources": chunk_ids[:3]
                }
                
                # Store in the analysis results
                rag_state["analysis_results"][segment_name] = result
                
                return result
                
        # If no chunks found
        return {
            "status": "error",
            "message": f"No relevant content found for segment: {segment_name} in {market_type}"
        }
        
    except Exception as e:
        logger.error(f"Error analyzing market segment: {str(e)}")
        return {"error": f"Error analyzing market segment: {str(e)}"}

@traceable(name="generate_segment_strategy")
@mcp_server.tool()
def generate_segment_strategy(segment_name: str, product_type: str, competitive_position: str = "challenger") -> Dict[str, Any]:
    """Generate marketing strategy for a segment based on Kotler's book and provide relevant quotes."""
    try:
        logger.info(f"Generating strategy for {segment_name}, product: {product_type}")
        
        # First check if we already have analysis results
        if segment_name not in rag_state["analysis_results"]:
            # If not, run the analysis
            analysis_result = analyze_market_segment(segment_name)
            if isinstance(analysis_result, dict) and "error" in analysis_result:
                return analysis_result
        
        # Get the analysis results
        segment_analysis = rag_state["analysis_results"].get(segment_name, {})
        
        # Check if we have chunks from Kotler's book
        if "chunks" in segment_analysis and segment_analysis["chunks"]:
            chunk_content = segment_analysis["chunks"]
            chunk_sources = segment_analysis.get("sources", [])
            
            return {
                "status": "success",
                "segment_name": segment_name,
                "product_type": product_type,
                "competitive_position": competitive_position,
                "segment_analysis": chunk_content,
                "sources": chunk_sources
            }
        else:
            return {
                "status": "error",
                "message": f"No strategy data available for segment: {segment_name}"
            }
    except Exception as e:
        logger.error(f"Error generating segment strategy: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

@traceable(name="query_marketing_book")
@mcp_server.tool()
def query_marketing_book(query: str, top_k: int = 3) -> Dict[str, Any]:
    """Query Philip Kotler's Marketing Management book and return relevant chunks."""
    try:
        logger.info(f"Querying marketing book for: {query}")
        
        # First, search for relevant chunks
        chunk_ids = pinecone_search(query, top_k=top_k)
        
        # Check if we got valid results
        if not isinstance(chunk_ids, list) or not chunk_ids or chunk_ids[0].startswith("Error"):
            return {
                "status": "error",
                "message": "No relevant content found in the marketing book."
            }
        
        # Fetch each chunk
        chunks = []
        for chunk_id in chunk_ids:
            chunk_content = fetch_s3_chunk(chunk_id)
            if not chunk_content.startswith("Error"):
                chunks.append({
                    "chunk_id": chunk_id, 
                    "content": chunk_content
                })
        
        # Return the chunks with their IDs
        return {
            "status": "success",
            "query": query,
            "chunks": chunks,
            "chunks_found": len(chunks)
        }
    except Exception as e:
        logger.error(f"Error querying marketing book: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

# Mount MCP server to FastAPI app
app.mount("/mcp", mcp_server.sse_app())

# Add direct endpoints for critical tools to ensure accessibility
class ToolRequest(BaseModel):
    name: str
    parameters: Dict[str, Any]

@app.post("/tools/query_marketing_book/invoke")
async def direct_query_marketing_book(request: Request):
    """Direct endpoint for query_marketing_book tool to ensure accessibility"""
    try:
        # Parse the JSON request body manually
        json_data = await request.json()
        
        # Extract parameters from request
        parameters = json_data.get("parameters", {})
        query = parameters.get("query", "")
        top_k = parameters.get("top_k", 3)
        
        logger.info(f"Direct endpoint received query: {query}, top_k: {top_k}")
        
        # Call the actual tool function
        result = query_marketing_book(query, top_k)
        
        # Return in the expected format
        return {"content": result}
    except Exception as e:
        logger.error(f"Error in direct_query_marketing_book endpoint: {str(e)}", exc_info=True)
        return {"content": {"status": "error", "message": str(e)}}

# Add an explicit endpoint at the MCP server path to ensure both endpoint patterns work
@app.post("/mcp/tools/query_marketing_book/invoke")
async def mcp_direct_query_marketing_book(request: Request):
    """MCP path direct endpoint for query_marketing_book to match client expectations"""
    # Reuse the same implementation as the non-mcp path
    return await direct_query_marketing_book(request)

@app.post("/direct/query_marketing_content")
async def direct_query_marketing_content(query: str, top_k: int = 3):
    """Direct endpoint for querying marketing content with OpenAI embeddings"""
    try:
        logger.info(f"Query marketing content for: {query}")
        
        # Get embeddings for the query - handle both OpenAI API styles
        query_embedding = None
        try:
            if has_new_openai:
                # New API style
                response = openai_client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=[query]
                )
                query_embedding = response.data[0].embedding
            else:
                # Old API style
                response = openai_client.Embedding.create(
                    model="text-embedding-ada-002",
                    input=[query]
                )
                query_embedding = response["data"][0]["embedding"]
        except Exception as embed_error:
            logger.error(f"Error creating embeddings: {str(embed_error)}")
            return {"status": "error", "message": "Error with embedding creation. Please try again."}
        
        # Use updated Pinecone import
        from pinecone import Pinecone
        
        # Initialize Pinecone with the updated API
        pc = Pinecone(api_key=Config.PINECONE_API_KEY)
        index_name = "healthcare-product-analytics"
        
        logger.info(f"Querying Pinecone index: {index_name} for marketing content")
        
        # Get the index
        index = pc.Index(index_name)
        
        # Search Pinecone - try multiple namespaces
        namespaces_to_try = ["book-kotler", "kotler", "marketing", None] # None for default namespace
        results = None
        
        for namespace in namespaces_to_try:
            try:
                namespace_str = namespace if namespace else "default"
                logger.info(f"Trying namespace: {namespace_str}")
                
                # If namespace is None, don't include it in the query
                if namespace is None:
                    results = index.query(
                        vector=query_embedding,
                        top_k=top_k,
                        include_metadata=True
                    )
                else:
                    results = index.query(
                        vector=query_embedding,
                        top_k=top_k,
                        include_metadata=True,
                        namespace=namespace
                    )
                
                if results and hasattr(results, 'matches') and len(results.matches) > 0:
                    logger.info(f"Found {len(results.matches)} matches in namespace {namespace_str}")
                    break
            except Exception as ns_error:
                logger.warning(f"Error with namespace {namespace_str}: {str(ns_error)}")
                continue
        
        # Check if we have results
        if not results or not hasattr(results, 'matches') or len(results.matches) == 0:
            logger.warning(f"No marketing content found for query: {query}")
            return {
                "status": "error",
                "message": "No relevant marketing content found. Please try a different query."
            }
        
        # Extract and process results
        content_chunks = []
        for match in results.matches:
            if hasattr(match, 'metadata') and match.metadata:
                # Get content from the chunk
                content = match.metadata.get('text', 'No content available')
                source = match.id  # Using chunk ID as source reference
                content_chunks.append({
                    "content": content,
                    "source": source,
                    "score": float(match.score)
                })
        
        # Prepare response
        return {
            "status": "success",
            "query": query,
            "chunks": content_chunks,
            "chunks_found": len(content_chunks)
        }
    except Exception as e:
        logger.error(f"Error querying marketing content: {str(e)}")
        return {
            "status": "error",
            "message": f"Error retrieving marketing content: {str(e)}"
        }

# Add a simple direct API endpoint for marketing book queries that won't time out
@app.get("/api/marketing/query")
async def simple_marketing_query(query: str, top_k: int = 5):
    """
    Simple, direct endpoint for marketing book queries that returns plaintext results.
    This endpoint is designed for maximum reliability and avoids timeout issues.
    """
    try:
        logger.info(f"Simple marketing query: {query}, top_k: {top_k}")
        
        # First, search for relevant chunks using the existing function
        chunk_ids = pinecone_search(query, top_k=top_k)
        
        if not isinstance(chunk_ids, list) or not chunk_ids or chunk_ids[0].startswith("Error"):
            return {"status": "error", "message": "No relevant content found"}
            
        # Fetch chunks one by one
        chunks = []
        for chunk_id in chunk_ids:
            try:
                chunk_content = fetch_s3_chunk(chunk_id)
                if not chunk_content.startswith("Error"):
                    chunks.append({
                        "id": chunk_id,
                        "content": chunk_content[:500] + "..." if len(chunk_content) > 500 else chunk_content
                    })
            except Exception as chunk_error:
                logger.error(f"Error fetching chunk {chunk_id}: {str(chunk_error)}")
                continue
                
        if not chunks:
            return {"status": "error", "message": "Failed to fetch chunks"}
            
        # Return a simplified response to avoid any serialization issues
        return {
            "status": "success",
            "query": query,
            "results": chunks,
            "count": len(chunks)
        }
        
    except Exception as e:
        logger.error(f"Error in simple marketing query: {str(e)}")
        return {"status": "error", "message": f"Server error: {str(e)}"}  

# Add health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# Run the server
if __name__ == "__main__":
    port = 8001  # Market analysis server on port 8001
    logger.info(f"Starting Market Analysis MCP Server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
