import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

def setup_pinecone_index():
    """
    Create Pinecone index with correct configuration for OpenAI embeddings.
    Run this script before starting the Airflow pipeline.
    """
    # Load environment variables
    load_dotenv()
    
    # Initialize Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    # Configuration
    index_name = os.getenv("PINECONE_INDEX_NAME")
    
    try:
        # Check if index already exists
        existing_indexes = pc.list_indexes().names()
        if index_name in existing_indexes:
            print(f"✅ Index '{index_name}' already exists")
            
            # Verify index configuration
            index_info = pc.describe_index(index_name)
            if index_info.dimension != 1536:
                raise ValueError(f"Existing index has wrong dimension: {index_info.dimension} (expected 1536)")
            if index_info.metric != "cosine":
                raise ValueError(f"Existing index has wrong metric: {index_info.metric} (expected cosine)")
                
            print("✅ Index configuration verified")
            return
        
        # Create new index
        print(f"Creating Pinecone index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=1536,  # OpenAI ada-002 dimensions
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        
        # Wait for index to be ready
        print("Waiting for index to be ready...")
        while True:
            try:
                index_info = pc.describe_index(index_name)
                if index_info.status['ready']:
                    break
            except Exception as e:
                print(f"Still waiting... ({str(e)})")
            time.sleep(10)
        
        print("✅ Index is ready!")
        
        # Print index details
        print("\nIndex Configuration:")
        print(f"- Name: {index_name}")
        print(f"- Dimension: {index_info.dimension}")
        print(f"- Metric: {index_info.metric}")
        print(f"- Region: {index_info.host}")
        
    except Exception as e:
        print(f"❌ Error setting up Pinecone index: {str(e)}")
        raise

if __name__ == "__main__":
    setup_pinecone_index()