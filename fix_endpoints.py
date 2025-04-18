#!/usr/bin/env python
"""
Endpoint Fix Script for MarketScope

This script ensures all API endpoints in the MarketScope project are properly configured.
It fixes issues with:
- Direct endpoint paths in MCP servers
- OpenAI API key loading
- Timeouts and connection errors
"""
import os
import sys
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

def fix_market_analysis_server():
    """Fix the market_analysis_mcp_server.py file"""
    filepath = "mcp_servers/market_analysis_mcp_server.py"
    print(f"Fixing {filepath}...")
    
    # Make sure the file exists
    if not os.path.exists(filepath):
        print(f"  ⚠ File not found: {filepath}")
        return False
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fixes to apply:
    
    # 1. Fix OpenAI API key loading
    modified = False
    if "import os" not in content:
        content = "import os\nfrom dotenv import load_dotenv\n\n# Load environment variables from .env file\nload_dotenv(override=True)\n" + content
        modified = True
        print("  ✅ Added dotenv imports and loading")
        
    # 2. Add OpenAI API key handling
    if "OPENAI_API_KEY = os.getenv" not in content:
        api_key_code = """
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
"""
        
        # Find position after imports but before OpenAI client initialization
        import_section_end = content.find("# Initialize OpenAI client")
        if import_section_end != -1:
            content = content[:import_section_end] + api_key_code + content[import_section_end:]
            # Update the client initialization to use our API key variable
            content = content.replace("api_key=Config.OPENAI_API_KEY", "api_key=OPENAI_API_KEY")
            modified = True
            print("  ✅ Added OpenAI API key handling")
    
    # 3. Add the MCP tools endpoint
    if "@app.post(\"/mcp/tools/query_marketing_book/invoke\")" not in content:
        # Find the position after the direct endpoint but before health check
        direct_endpoint_pos = content.find("@app.post(\"/tools/query_marketing_book/invoke\")")
        if direct_endpoint_pos != -1:
            health_check_pos = content.find("# Add health check endpoint", direct_endpoint_pos)
            if health_check_pos != -1:
                mcp_endpoint_code = """
# Add an explicit endpoint at the MCP server path to ensure both endpoint patterns work
@app.post("/mcp/tools/query_marketing_book/invoke")
async def mcp_direct_query_marketing_book(request: Request):
    # MCP path direct endpoint for query_marketing_book to match client expectations
    # Reuse the same implementation as the non-mcp path
    return await direct_query_marketing_book(request)
"""
                content = content[:health_check_pos] + mcp_endpoint_code + content[health_check_pos:]
                modified = True
                print("  ✅ Added MCP tools endpoint")
    
    # Write back the fixed content if changes were made
    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✅ Successfully updated {filepath}")
        return True
    else:
        print(f"  ✓ No changes needed for {filepath}")
        return False

def fix_custom_mcp_client():
    """Fix the custom_mcp_client.py file to handle endpoint issues better"""
    filepath = "agents/custom_mcp_client.py"
    print(f"Fixing {filepath}...")
    
    # Make sure the file exists
    if not os.path.exists(filepath):
        print(f"  ⚠ File not found: {filepath}")
        return False
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fixes to apply:
    
    # Increase timeout for query_marketing_book calls
    modified = False
    if "timeout=30" in content:
        content = content.replace("timeout=30", "timeout=60")  # Increase timeout to 60 seconds
        modified = True
        print("  ✅ Increased timeout for API calls")
    
    # Improve error handling for _invoke_marketing_book_directly
    if "logger.error(\"All direct invocation attempts for marketing book query failed\")" in content:
        # Find the position of the error message
        error_pos = content.find("logger.error(\"All direct invocation attempts for marketing book query failed\")")
        if error_pos != -1:
            # Find the return statement position
            return_pos = content.find("return {", error_pos)
            if return_pos != -1:
                # Add retry logic before the return statement
                retry_code = """
            # Try one more approach - direct URL for content
            try:
                logger.info(f"Trying direct content URL as last resort")
                direct_url = f"{self.service_url}/direct/query_marketing_content"
                response = requests.post(
                    direct_url,
                    json={"query": parameters.get("query", ""), "top_k": parameters.get("top_k", 3)},
                    timeout=60,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if "chunks" in result and len(result["chunks"]) > 0:
                        # Transform to expected format
                        chunks = [{"chunk_id": chunk["source"], "content": chunk["content"]} for chunk in result["chunks"]]
                        return {
                            "status": "success",
                            "chunks": chunks,
                            "chunks_found": len(chunks),
                            "query": parameters.get("query", "")
                        }
            except Exception as e:
                logger.warning(f"Error with direct content URL approach: {str(e)}")
            """
                content = content[:return_pos] + retry_code + content[return_pos:]
                modified = True
                print("  ✅ Added retry logic for marketing book queries")
    
    # Write back the fixed content if changes were made
    if modified:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✅ Successfully updated {filepath}")
        return True
    else:
        print(f"  ✓ No changes needed for {filepath}")
        return False

def main():
    """Main function to fix all endpoint issues"""
    print("Fixing MarketScope API endpoint issues...")
    
    # Fix market analysis server
    fix_market_analysis_server()
    
    # Fix custom MCP client
    fix_custom_mcp_client()
    
    print("\nFixes applied. Please restart your servers for the changes to take effect.")
    print("Ensure your .env file contains a valid OPENAI_API_KEY.")

if __name__ == "__main__":
    main()
