"""
Marketing Management Book Tools for MCP Server
These tools provide access to marketing knowledge from Philip Kotler's Marketing Management book.
"""
import json
import boto3
from openai import OpenAI
from langsmith import traceable

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
@traceable(name="pinecone_search", run_type="chain")
def pinecone_search(query: str, top_k: int = 3):
    """Search for relevant chunks in Pinecone using query embeddings."""
    try:
        # Use updated Pinecone import
        from pinecone import Pinecone
        
        # Initialize Pinecone with the updated API
        pc = Pinecone(api_key=Config.PINECONE_API_KEY)
        index_name = Config.PINECONE_INDEX_NAME
        
        # Print debug info
        print(f"Searching Pinecone index: {index_name}")
        
        # Get the index (assuming it already exists)
        index = pc.Index(index_name)
        
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
        # Return a clear error message
        error_msg = f"Error in pinecone_search: {str(e)}"
        print(error_msg)
        return [error_msg]
    
@traceable(name="fetch_s3_chunk", run_type="chain")
def fetch_s3_chunk(chunk_id: str):
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
            error_msg = f"Chunk {chunk_id} not found in chunks file."
            print(error_msg)
            return error_msg
            
    except Exception as e:
        error_msg = f"Error fetching chunk: {str(e)}"
        print(error_msg)
        return error_msg

# Metadata and Aggregation Tools
@traceable(name="get_chunks_metadata", run_type="chain")
def get_chunks_metadata():
    """Get metadata about available chunks."""
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
        
        print(f"Fetching metadata from S3: bucket={bucket_name}, key={key}")
        
        # Get the JSON file from S3
        response = s3_client.get_object(Bucket=bucket_name, Key=key)
        chunks_data = json.loads(response['Body'].read().decode('utf-8'))
        
        metadata = {
            "book_name": chunks_data.get("book_name", "Unknown"),
            "total_chunks": chunks_data.get("total_chunks", 0),
            "created_at": chunks_data.get("created_at", "Unknown"),
            "chunk_ids": list(chunks_data.get("chunks", {}).keys())
        }
        
        print(f"Successfully retrieved metadata with {metadata['total_chunks']} chunks")
        return metadata
        
    except Exception as e:
        error_msg = f"Error retrieving chunks metadata: {str(e)}"
        print(error_msg)
        return error_msg

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

# Marketing Analysis Tools
@traceable(name="analyze_market_segment", run_type="chain")
def analyze_market_segment(segment_name: str, market_type: str = "healthcare"):
    """Analyze a specific market segment using the marketing book knowledge."""
    try:
        # Update state
        rag_state["current_segment"] = segment_name
        
        # First search for relevant chunks about this segment
        search_query = f"marketing segmentation strategy for {segment_name} in {market_type}"
        chunk_ids = pinecone_search(search_query, top_k=5)
        
        if isinstance(chunk_ids, list) and chunk_ids and not chunk_ids[0].startswith("Error"):
            # Retrieve chunks
            chunk_contents = [fetch_s3_chunk(chunk_id) for chunk_id in chunk_ids]
            chunk_contents = [c for c in chunk_contents if not c.startswith("Error") and not c.startswith("Chunk")]
            
            if chunk_contents:
                # Generate analysis using OpenAI
                analysis_prompt = f"""
                Based on Philip Kotler's Marketing Management principles, analyze the {segment_name} segment 
                in the {market_type} market. Use the following relevant information from the textbook:
                
                {' '.join(chunk_contents[:3])}
                
                Provide a structured analysis including:
                1. Key characteristics of this segment
                2. Recommended positioning strategies
                3. Potential marketing mix adaptations
                """
                
                response = openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": analysis_prompt}]
                )
                
                analysis = response.choices[0].message.content
                
                # Store in state
                rag_state["analysis_results"][segment_name] = {
                    "analysis": analysis,
                    "sources": chunk_ids[:3]
                }
                
                return analysis
            else:
                return "Could not find relevant content about this segment in the marketing literature."
        else:
            return "Could not find relevant segments in the marketing knowledge base."
    
    except Exception as e:
        return f"Error analyzing market segment: {str(e)}"

@traceable(name="generate_segment_strategy", run_type="chain")
def generate_segment_strategy(segment_name: str, product_type: str, competitive_position: str = "challenger"):
    """Generate a marketing strategy for a specific product in a segment."""
    try:
        # First get or create segment analysis
        if segment_name not in rag_state["analysis_results"]:
            analyze_market_segment(segment_name)
            
        segment_analysis = rag_state["analysis_results"].get(segment_name, {}).get("analysis", "")
        
        if not segment_analysis:
            return f"Could not find or generate analysis for the {segment_name} segment."
            
        # Generate strategy
        strategy_prompt = f"""
        Based on Philip Kotler's Marketing Management principles and the segment analysis below,
        create a specific marketing strategy for a {competitive_position} company 
        selling {product_type} in the {segment_name} segment.
        
        Segment Analysis:
        {segment_analysis[:1000]}
        
        Include:
        1. Value proposition
        2. Pricing strategy
        3. Distribution channels
        4. Promotion tactics
        5. Key performance indicators
        """
        
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": strategy_prompt}]
        )
        
        strategy = response.choices[0].message.content
        return strategy
        
    except Exception as e:
        return f"Error generating segment strategy: {str(e)}"

# A dictionary mapping tool names to functions
# This will be used by the MCP server to register tools
tool_functions = {
    "pinecone_search": pinecone_search,
    "fetch_s3_chunk": fetch_s3_chunk,
    "get_chunks_metadata": get_chunks_metadata,
    "get_all_retrieved_chunks": get_all_retrieved_chunks,
    "analyze_market_segment": analyze_market_segment,
    "generate_segment_strategy": generate_segment_strategy
}