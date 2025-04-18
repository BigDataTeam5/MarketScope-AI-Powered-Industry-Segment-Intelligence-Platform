"""
MCP Pinecone Integration Test - Using the modern Pinecone API
"""
import os
import logging
import traceback
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("mcp_pinecone_test")

# Get Pinecone API key from environment or input
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    PINECONE_API_KEY = input("Enter your Pinecone API key: ")

# Initialize Pinecone with the new API (as shown in your error message)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Initialize the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

def test_pinecone_connection():
    """Test basic Pinecone connection"""
    logger.info("Testing Pinecone connection...")
    
    try:
        # List indexes
        indexes = pc.list_indexes()
        index_names = [index.name for index in indexes]
        logger.info(f"Available indexes: {index_names}")
        
        # Check for healthcare index
        target_index = "healthcare-industry-reports"
        if target_index in index_names:
            logger.info(f"✅ Found target index '{target_index}'")
            return True
        else:
            logger.warning(f"❌ Target index '{target_index}' not found")
            return False
    except Exception as e:
        logger.error(f"❌ Error connecting to Pinecone: {e}")
        logger.error(traceback.format_exc())
        return False

def test_mcp_integration():
    """Test Pinecone integration with MCP server"""
    logger.info("Testing Pinecone integration with MCP...")
    
    # Step 1: Get a healthcare-related embedding
    logger.info("Generating test embedding...")
    query_text = "healthcare industry company financial metrics"
    embedding = model.encode(query_text).tolist()
    logger.info(f"Generated embedding with dimension {len(embedding)}")
    
    # Step 2: Try to query Pinecone
    try:
        index_name = "healthcare-industry-reports"
        index = pc.Index(index_name)
        
        # Test basic stats
        try:
            stats = index.describe_index_stats()
            logger.info(f"Index stats: {stats}")
        except Exception as e:
            logger.warning(f"Could not get index stats: {e}")
        
        # Test query
        logger.info("Querying index...")
        results = index.query(
            vector=embedding,
            top_k=3,
            include_metadata=True
        )
        
        # Check results
        match_count = len(results.matches)
        logger.info(f"✅ Query returned {match_count} matches")
        
        if match_count > 0:
            # Display first match
            match = results.matches[0]
            logger.info(f"Top match score: {match.score:.4f}")
            
            # Check for metadata
            if hasattr(match, 'metadata') and match.metadata:
                logger.info(f"Metadata keys: {list(match.metadata.keys())}")
                
                # Check for company_name
                if 'company_name' in match.metadata:
                    logger.info(f"Company: {match.metadata['company_name']}")
                
                # Check for financial data
                financial_keys = [k for k in match.metadata.keys() if k in 
                                 ['revenue', 'profit', 'profit_margin', 'market_share']]
                if financial_keys:
                    logger.info(f"Financial data keys: {financial_keys}")
                
                # Check for product_performance
                if 'product_performance' in match.metadata:
                    logger.info("Product performance data available")
            
            return True
        else:
            logger.warning("No matches found in query results")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error querying Pinecone: {e}")
        logger.error(traceback.format_exc())
        return False

def check_mcp_compatibility():
    """Check if this version of Pinecone is compatible with your MCP code"""
    logger.info("Checking MCP compatibility...")
    
    try:
        # Create a sample Python file to test imports
        with open("pinecone_mcp_check.py", "w") as f:
            f.write("""
# Test imports for MCP compatibility
from pinecone import Pinecone
import os

# Initialize client
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY", "test-key"))

# Define function that matches your MCP code structure
def retrieve_competitor_financial_data(company_name, metrics=["revenue", "profit", "margin"]):
    # This mimics your MCP function but with the new SDK
    try:
        # Get the index
        index = pc.Index("healthcare-industry-reports")
        
        # Create query embedding (using your SentenceTransformer in reality)
        query_embedding = [0.1] * 384  # Dummy embedding
        
        # Query
        results = index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True
        )
        
        return {
            "status": "success",
            "matches_found": len(results.matches)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

# Print function definition for reference
print("Function defined successfully")
""")
        
        # Try to run the test file
        import subprocess
        result = subprocess.run(
            ["python", "pinecone_mcp_check.py"], 
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logger.info("✅ MCP compatibility check passed!")
            return True
        else:
            logger.error(f"❌ MCP compatibility check failed: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"Error checking MCP compatibility: {e}")
        return False
    finally:
        # Cleanup test file
        try:
            import os
            os.remove("pinecone_mcp_check.py")
        except:
            pass

def provide_fix_recommendations():
    """Provide recommendations to fix the MCP server code"""
    logger.info("\n--- MCP Server Fix Recommendations ---\n")
    logger.info("Based on your error message, you need to update your MCP server code to use the new Pinecone SDK.")
    logger.info("Here are the changes you need to make:\n")
    
    logger.info("1. Change your imports from:")
    logger.info("   import pinecone")
    logger.info("To:")
    logger.info("   from pinecone import Pinecone\n")
    
    logger.info("2. Change your initialization from:")
    logger.info("   pinecone.init(api_key=Config.PINECONE_API_KEY, environment=\"us-east-1-gcp\")")
    logger.info("   index = pinecone.Index(\"healthcare-industry-reports\")")
    logger.info("To:")
    logger.info("   pc = Pinecone(api_key=Config.PINECONE_API_KEY)")
    logger.info("   index = pc.Index(\"healthcare-industry-reports\")\n")
    
    logger.info("3. Update any query response processing to work with the new object format:")
    logger.info("   Old: results['matches']")
    logger.info("   New: results.matches\n")
    
    logger.info("4. Update metadata access:")
    logger.info("   Old: match.get('metadata', {}).get('company_name')")
    logger.info("   New: match.metadata.get('company_name')\n")
    
    logger.info("These changes should make your MCP server compatible with your current Pinecone SDK.")

def main():
    """Run all tests"""
    logger.info("Starting MCP Pinecone integration tests...")
    
    # Test connection
    if test_pinecone_connection():
        # Test integration
        test_mcp_integration()
        
        # Check compatibility with MCP
        check_mcp_compatibility()
    
    # Always provide recommendations
    provide_fix_recommendations()
    
    logger.info("\nTests completed.")

if __name__ == "__main__":
    main()