import requests
import json
import logging

logger = logging.getLogger('custom_mcp_client')

class CustomMCPClient:
    """Custom MCP client implementation for communicating with Snowflake_server.py"""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        logger.info(f"Initialized CustomMCPClient with base_url: {base_url}")
    
    def get(self, path):
        """Make a GET request to the specified path"""
        url = f"{self.base_url}{path}"
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error making GET request to {url}: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def invoke(self, name, inputs=None):
        """Invoke a tool by name with the given inputs"""
        if inputs is None:
            inputs = {}
        
        url = f"{self.base_url}/invoke"
        payload = {
            "name": name,
            "inputs": inputs
        }
        
        try:
            logger.info(f"Invoking {name} with inputs: {inputs}")
            response = self.session.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error invoking {name}: {str(e)}")
            return str(e)
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error invoking {name}. Is the MCP server running? Error: {str(e)}")
            return f"Connection error: Is the MCP server running at {self.base_url}?"
        except Exception as e:
            logger.error(f"Error invoking {name}: {str(e)}")
            return str(e)