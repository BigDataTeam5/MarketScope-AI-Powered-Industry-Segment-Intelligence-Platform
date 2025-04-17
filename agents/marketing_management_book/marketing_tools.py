"""
Marketing Management Book Tools for MCP Server
These tools provide access to marketing knowledge from Philip Kotler's Marketing Management book.
"""
import json
import boto3
from openai import OpenAI
from langsmith import traceable
from typing import Dict, Any, List, Union

# Import consolidated config
from config.config import Config

# Initialize OpenAI client
openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)

# Global state for RAG operations
rag_state = {
    "retrieved_chunks": [],
    "last_query": None,
    "current_segment": None,
    "analysis_results": {}
}

# Core Search & Retrieval Tools
@traceable(name="pinecone_search")
def pinecone_search(query: str, top_k: int = 3) -> Union[List[str], Dict[str, str]]:
    """Search for relevant chunks in Pinecone using query embeddings."""
    try:
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
        print(f"Searching Pinecone index: {index_name}")
        
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
        print(f"Found {len(results.matches)} matches: {matches}")
        
        # Return chunk IDs directly
        if not results.matches:
            return ["No relevant chunks found. Please try a different query."]
        return [match.id for match in results.matches]
    
    except Exception as e:
        return {"error": f"Error in pinecone_search: {str(e)}"}

@traceable(name="fetch_s3_chunk")
def fetch_s3_chunk(chunk_id: str) -> str:
    """Fetch a specific chunk from the S3 chunks file."""
    print(f"Fetching chunk: {chunk_id}")
    
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
        
        print(f"Fetching from S3: bucket={bucket_name}, key={key}")
        
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
            
            print(f"Successfully retrieved chunk {chunk_id}")
            return chunk_text
        else:
            return f"Error fetching chunk {chunk_id}: Chunk not found in chunks file."
            
    except Exception as e:
        return f"Error fetching chunk {chunk_id}: {str(e)}"

# Metadata and Aggregation Tools
@traceable(name="get_chunks_metadata")
def get_chunks_metadata() -> Dict[str, Any]:
    """Get metadata about available chunks."""
    try:
        return {
            "retrieved_count": len(rag_state["retrieved_chunks"]),
            "last_query": rag_state["last_query"],
            "current_segment": rag_state["current_segment"]
        }
    except Exception as e:
        return {"error": f"Error getting chunks metadata: {str(e)}"}

@traceable(name="get_all_retrieved_chunks")
def get_all_retrieved_chunks() -> List[str]:
    """Get all chunks that have been retrieved in this session."""
    return rag_state["retrieved_chunks"]

# Marketing Analysis Tools
@traceable(name="analyze_market_segment")
def analyze_market_segment(segment_name: str, market_type: str = "healthcare") -> Dict[str, Any]:
    """Retrieve and aggregate relevant content for a market segment (no LLM call)."""
    try:
        rag_state["current_segment"] = segment_name
        search_query = f"marketing segmentation strategy for {segment_name} in {market_type}"
        chunk_ids = pinecone_search(search_query, top_k=5)
        
        if isinstance(chunk_ids, list) and chunk_ids and not chunk_ids[0].startswith("Error"):
            chunk_contents = [fetch_s3_chunk(chunk_id) for chunk_id in chunk_ids]
            chunk_contents = [c for c in chunk_contents if not c.startswith("Error") and not c.startswith("Chunk")]
            
            if chunk_contents:
                return {
                    "segment_name": segment_name,
                    "market_type": market_type,
                    "chunks": chunk_contents[:3],
                    "sources": chunk_ids[:3]
                }
        
        return {"error": "No relevant content found in the marketing literature."}
    except Exception as e:
        return {"error": f"Error analyzing market segment: {str(e)}"}

@traceable(name="generate_segment_strategy")
def generate_segment_strategy(segment_name: str, product_type: str, competitive_position: str = "challenger") -> Dict[str, Any]:
    """Aggregate data for strategy generation (no LLM call)."""
    try:
        if segment_name not in rag_state["analysis_results"]:
            analysis_result = analyze_market_segment(segment_name)
            if isinstance(analysis_result, dict) and "error" in analysis_result:
                return analysis_result
            rag_state["analysis_results"][segment_name] = analysis_result

        segment_analysis = rag_state["analysis_results"].get(segment_name, {}).get("chunks", [])
        if not segment_analysis:
            return {"error": f"Could not find or generate analysis for the {segment_name} segment."}
        
        return {
            "segment_name": segment_name,
            "product_type": product_type,
            "competitive_position": competitive_position,
            "segment_analysis": segment_analysis
        }
    except Exception as e:
        return {"error": f"Error generating segment strategy: {str(e)}"}

# A dictionary mapping tool names to functions
tool_functions = {
    "pinecone_search": pinecone_search,
    "fetch_s3_chunk": fetch_s3_chunk,
    "get_chunks_metadata": get_chunks_metadata,
    "get_all_retrieved_chunks": get_all_retrieved_chunks,
    "analyze_market_segment": analyze_market_segment,
    "generate_segment_strategy": generate_segment_strategy
}