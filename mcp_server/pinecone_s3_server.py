import os
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
from openai import OpenAI

from config import (
    OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENV, PINECONE_INDEX_NAME,
    S3_BUCKET_NAME, S3_CHUNKS_PATH, S3_CHUNKS_FILE,
    AWS_SERVER_PUBLIC_KEY, AWS_SERVER_SECRET_KEY, AWS_REGION,
    setup_langsmith
)
setup_langsmith()
from langsmith import traceable
# Create MCP server instance
mcp = FastMCP("PineconeS3")

# Create OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Simple RAG state to track retrieved chunks
rag_state = {
    "retrieved_chunks": [],
    "last_query": None
}

# Tools
@mcp.tool()
@traceable(name="pinecone_search", run_type="chain")
def pinecone_search(query: str, top_k: int = 3):
    """Search for relevant chunks in Pinecone using query embeddings."""
    try:
        # Use updated Pinecone import
        from pinecone import Pinecone
        
        # Initialize Pinecone with the updated API
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # Get the index (assuming it already exists)
        index = pc.Index(PINECONE_INDEX_NAME)
        
        # Generate embedding
        response = openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=query
        )
        vector = response.data[0].embedding
        
        # Search Pinecone - use the "book-kotler" namespace
        results = index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True,
            namespace="book-kotler"  # Match the namespace from your screenshot
        )
        
        # Update state
        rag_state["last_query"] = query
        rag_state["retrieved_chunks"] = []
        
        # Get matches and their scores
        matches = []
        for match in results.matches:
            # Extract the actual ID from the metadata (which matches your S3 chunks)
            chunk_id = match.id  # This should be something like "kotler_chunk_0"
            score = match.score
            matches.append(f"{chunk_id} (score: {score:.4f})")
            
        # Print debug info about what we found
        print(f"Found {len(results.matches)} matches: {matches}")
        
        # Return chunk IDs directly
        return [match.id for match in results.matches]
    except Exception as e:
        # Return a clear error message
        error_msg = f"Error in pinecone_search: {str(e)}"
        print(error_msg)
        return [error_msg]
    
@mcp.tool()
@traceable(name="fetch_s3_chunk", run_type="chain")
def fetch_s3_chunk(chunk_id: str):
    """Fetch a specific chunk from the S3 chunks file."""
    import boto3
    import json
    
    print(f"Fetching chunk: {chunk_id}")
    
    # Get AWS credentials
    access_key = os.getenv("AWS_SERVER_PUBLIC_KEY", "")
    secret_key = os.getenv("AWS_SERVER_SECRET_KEY", "")
    region = os.getenv("AWS_REGION", "us-east-1")
    
    # Initialize S3 client
    s3_client = boto3.client(
        's3',
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region
    )
    
    try:
        # Construct object key for the single chunks file
        object_key = f"{S3_CHUNKS_PATH}{S3_CHUNKS_FILE}"
        
        # Get object from S3
        response = s3_client.get_object(
            Bucket=S3_BUCKET_NAME,
            Key=object_key
        )
        
        # Parse JSON
        content = response['Body'].read().decode('utf-8')
        chunks_data = json.loads(content)
        
        # Extract the specific chunk - the ID from Pinecone is directly usable
        if "chunks" in chunks_data and chunk_id in chunks_data["chunks"]:
            chunk = chunks_data["chunks"][chunk_id]
            chunk_text = chunk.get("text", "")
            
            # Add to retrieved chunks
            rag_state["retrieved_chunks"].append({
                "chunk_id": chunk_id,
                "content": chunk_text
            })
            
            # Print preview of the chunk
            preview = chunk_text[:100] + "..." if len(chunk_text) > 100 else chunk_text
            print(f"Retrieved chunk {chunk_id}: {preview}")
            
            return chunk_text
        else:
            error_msg = f"Chunk {chunk_id} not found in chunks file."
            print(error_msg)
            return error_msg
        
    except Exception as e:
        error_msg = f"Error fetching chunk {chunk_id}: {str(e)}"
        print(error_msg)
        return error_msg
    
@mcp.tool()
@traceable(name="get_chunks_metadata", run_type="chain")
def get_chunks_metadata():
    """Get metadata about available chunks."""
    import boto3
    import json
    
    # Get AWS credentials
    access_key = os.getenv("AWS_SERVER_PUBLIC_KEY", "")
    secret_key = os.getenv("AWS_SERVER_SECRET_KEY", "")
    region = os.getenv("AWS_REGION", "us-east-1")
    
    # Initialize S3 client
    s3_client = boto3.client(
        's3',
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region
    )
    
    try:
        # Construct object key
        object_key = f"{S3_CHUNKS_PATH}{S3_CHUNKS_FILE}"
        
        # Get object from S3
        response = s3_client.get_object(
            Bucket=S3_BUCKET_NAME,
            Key=object_key
        )
        
        # Parse JSON
        content = response['Body'].read().decode('utf-8')
        chunks_data = json.loads(content)
        
        return {
            "book_name": chunks_data.get("book_name", "Unknown"),
            "total_chunks": chunks_data.get("total_chunks", 0),
            "created_at": chunks_data.get("created_at", "Unknown"),
            "chunk_ids": list(chunks_data.get("chunks", {}).keys())
        }
        
    except Exception as e:
        return f"Error retrieving chunks metadata: {str(e)}"

@mcp.tool()
@traceable(name="get_all_retrieved_chunks", run_type="chain")
def get_all_retrieved_chunks():
    """Get all chunks that have been retrieved in this session."""
    if not rag_state["retrieved_chunks"]:
        return "No chunks have been retrieved yet."
    
    # Format all chunks as a single text
    chunks_text = "\n\n".join([
        f"CHUNK {i+1} (ID: {chunk['chunk_id']}):\n{chunk['content']}"
        for i, chunk in enumerate(rag_state["retrieved_chunks"])
    ])
    
    return chunks_text

if __name__ == "__main__":
    # Run SSE transport
    mcp.run(transport="sse")