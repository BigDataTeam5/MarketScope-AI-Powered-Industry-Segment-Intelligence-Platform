"""
Configuration module for MarketScope
"""
import os
import importlib
from typing import Dict, Any, List, Type

class Config:
    """Unified configuration for the MarketScope application"""
    
    # AWS Credentials
    AWS_SERVER_PUBLIC_KEY = os.getenv("AWS_SERVER_PUBLIC_KEY")
    AWS_SERVER_SECRET_KEY = os.getenv("AWS_SERVER_SECRET_KEY")
    AWS_REGION = os.getenv("AWS_REGION", "us-east-2")
    
    # S3 Configuration
    BUCKET_NAME = os.getenv("BUCKET_NAME", "finalproject-product")
    S3_CHUNKS_PATH = os.getenv("S3_CHUNKS_PATH", "book-content/chunks/")
    S3_CHUNKS_FILE = os.getenv("S3_CHUNKS_FILE", "S3D7W4_Marketing_Management_chunks.json")
    
    # Pinecone Configuration
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "marketscope-index")
    
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GROK_API_KEY = os.getenv("GROK_API_KEY")
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    
    # Model Configuration
    DEFAULT_MODEL = "gpt-4o"
    DEFAULT_TEMPERATURE = 0.3
    
    AVAILABLE_MODELS = {
        "gpt-4o": {"name": "GPT-4 Optimized", "provider": "openai"},
        "claude-3-haiku-20240307": {"name": "Claude 3 Haiku", "provider": "anthropic"},
        "gemini-pro": {"name": "Gemini Pro", "provider": "google"},
        "deepseek-chat": {"name": "DeepSeek Chat", "provider": "deepseek"},
        "grok-1": {"name": "Grok-1", "provider": "grok"}
    }
    
    # Server Configuration
    MCP_PORT = int(os.getenv("MCP_PORT", "8000"))
    API_PORT = int(os.getenv("API_PORT", "8001"))
    MCP_URL = f"http://localhost:{MCP_PORT}/sse"
    
    # Tool Configuration
    ENABLED_TOOLS = [
        "pinecone_search",
        "fetch_s3_chunk",
        "get_chunks_metadata",
        "get_all_retrieved_chunks",
        "analyze_market_segment",
        "generate_segment_strategy"
    ]
    
    # Agent Configurations
    AGENT_CONFIGS: Dict[str, Dict[str, Any]] = {
        "marketing": {
            "agent_class": "agents.marketing_management_book.agent.MarketingManagementAgent",
            "tool_modules": ["agents.marketing_management_book.marketing_tools"],
            "description": "Marketing Management Agent"
        }
    }
    
    @classmethod
    def get_agent_class(cls, agent_name: str) -> Type:
        """Get the agent class from the fully qualified class name."""
        try:
            module_path, class_name = cls.AGENT_CONFIGS[agent_name]["agent_class"].rsplit(".", 1)
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except (KeyError, ValueError, ImportError, AttributeError) as e:
            raise ValueError(f"Could not load agent class for {agent_name}: {str(e)}")

    @classmethod
    def validate_config(cls) -> List[str]:
        """Validate that required configuration is present
        
        Returns:
            List of missing configuration variables
        """
        required_vars = [
            "AWS_SERVER_PUBLIC_KEY", 
            "AWS_SERVER_SECRET_KEY",
            "PINECONE_API_KEY",
            "OPENAI_API_KEY"
        ]
        
        missing = [var for var in required_vars if not getattr(cls, var)]
        
        return missing

    @classmethod
    def get_model_config(cls, model_name: str = None) -> Dict[str, Any]:
        """Get the configuration for a specific model"""
        model_name = model_name or cls.DEFAULT_MODEL
        return cls.AVAILABLE_MODELS.get(model_name, {"name": model_name, "provider": "unknown"})

