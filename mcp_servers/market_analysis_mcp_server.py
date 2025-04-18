"""
Market Analysis MCP Server
Provides tools for market analysis and segment strategies using
RAG with Philip Kotler's Marketing Management book
"""
import json
import boto3
from openai import OpenAI
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mcp.server.fastmcp import FastMCP
import uvicorn
from typing import Dict, Any, List, Optional, Union
from langsmith import traceable

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("market_analysis_mcp_server")

# Import configuration
from config.config import Config

# Initialize OpenAI client
openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)

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

# Core Search & Retrieval Tools
@traceable(name="pinecone_search")
@mcp_server.tool()
def pinecone_search(query: str, top_k: int = 3) -> Union[List[str], Dict[str, str]]:
    """Search for relevant chunks in Pinecone using query embeddings."""
    try:
        logger.info(f"Searching Pinecone for: {query}")
        
        # Get embeddings for the query
        response = openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=[query]
        )
        query_embedding = response.data[0].embedding
        
        # Use updated Pinecone import
        from pinecone import Pinecone
        
        # Initialize Pinecone with the updated API
        pc = Pinecone(api_key=Config.PINECONE_API_KEY)
        index_name = Config.PINECONE_INDEX_NAME
        
        # Print debug info
        logger.info(f"Searching Pinecone index: {index_name}")
        
        # Get the index (assuming it already exists)
        index = pc.Index(index_name)
        
        # Search Pinecone - use the "book-kotler" namespace
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace="book-kotler"
        )
        
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

# Add health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# Run the server
if __name__ == "__main__":
    port = 8001  # Market analysis server on port 8001
    logger.info(f"Starting Market Analysis MCP Server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
